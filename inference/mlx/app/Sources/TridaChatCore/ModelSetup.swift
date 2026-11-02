import Foundation

/// A checkpoint the app knows how to fetch and convert.
public struct ModelRecipe: Identifiable, Hashable, Sendable {
    public var repo: String          // Hugging Face repo id
    public var note: String
    public var id: String { repo }
    public var name: String { String(repo.split(separator: "/").last ?? Substring(repo)) }

    public init(repo: String, note: String) {
        self.repo = repo
        self.note = note
    }

    public static let known: [ModelRecipe] = [
        ModelRecipe(repo: "trillionlabs/Zoro-1", note: "Recommended. Agent + vision, ~10 GB download"),
        ModelRecipe(repo: "trillionlabs/Trida-2.0-4B-1006", note: "Vision-language Trida, ~10 GB download"),
        ModelRecipe(repo: "trillionlabs/Trida2.0-4B", note: "Text-only Trida, ~8 GB download"),
    ]

    /// Folder names under the models directory.
    public func rawFolder() -> String { name }
    public func convertedFolder(bits: Int) -> String { bits >= 16 ? name : "\(name)-mlx-q\(bits)" }
}

/// Terminal-style log: `\r` rewrites the current line (tqdm progress bars), `\n` starts a new one.
public struct TerminalLog: Equatable, Sendable {
    public private(set) var lines: [String] = [""]
    public var maxLines: Int

    public init(maxLines: Int = 400) {
        self.maxLines = maxLines
    }

    public mutating func append(_ s: String) {
        for ch in s {
            switch ch {
            case "\n", "\r\n":
                lines.append("")
            case "\r":
                lines[lines.count - 1] = ""
            default:
                lines[lines.count - 1].append(ch)
            }
        }
        if lines.count > maxLines { lines.removeFirst(lines.count - maxLines) }
    }

    /// The last line with text in it (the live progress bar, usually).
    public var current: String {
        lines.last(where: { !$0.trimmingCharacters(in: .whitespaces).isEmpty }) ?? ""
    }

    public var text: String { lines.joined(separator: "\n") }
}

/// Resolves a console script of the MLX project: its `.venv/bin/<tool>` after `uv sync`,
/// otherwise `uv run <tool>` through a login shell (so Homebrew / ~/.local/bin are on PATH).
public enum ProjectTool {
    public static func command(_ tool: String, project: URL, args: [String]) -> (executable: URL, arguments: [String]) {
        let venvBin = project.appendingPathComponent(".venv/bin/\(tool)")
        if FileManager.default.isExecutableFile(atPath: venvBin.path) {
            return (venvBin, args)
        }
        let line = "cd \(ServerProcess.shellQuote(project.path)) && exec uv run \(tool) "
            + args.map(ServerProcess.shellQuote).joined(separator: " ")
        return (URL(fileURLWithPath: "/bin/zsh"), ["-lc", line])
    }

    /// True when a Hugging Face token is available to the CLI (logged in, or HF_TOKEN set).
    public static func huggingFaceLoggedIn() -> Bool {
        let env = ProcessInfo.processInfo.environment
        if let t = env["HF_TOKEN"], !t.isEmpty { return true }
        let home = env["HF_HOME"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface")
        return FileManager.default.fileExists(atPath: home.appendingPathComponent("token").path)
    }

    /// Free space (bytes) on the volume holding `url`, or nil if unknown.
    public static func freeSpace(at url: URL) -> Int64? {
        var probe = url
        while !FileManager.default.fileExists(atPath: probe.path), probe.pathComponents.count > 1 {
            probe.deleteLastPathComponent()
        }
        let v = try? probe.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return v?.volumeAvailableCapacityForImportantUsage
    }
}

/// Runs one command at a time, streaming its output; `cancel()` terminates it.
public final class CommandRunner: @unchecked Sendable {
    private let lock = NSLock()
    private var process: Process?

    public init() {}

    public func run(_ executable: URL, _ arguments: [String], cwd: URL, extraEnv: [String: String] = [:],
                    onOutput: @escaping @Sendable (String) -> Void) async throws -> Int32 {
        let p = Process()
        p.executableURL = executable
        p.arguments = arguments
        p.currentDirectoryURL = cwd
        var env = ProcessInfo.processInfo.environment
        env["PYTHONUNBUFFERED"] = "1"
        env.removeValue(forKey: "VIRTUAL_ENV")
        env.removeValue(forKey: "PYTHONHOME")
        env.removeValue(forKey: "PYTHONPATH")
        for (k, v) in extraEnv { env[k] = v }
        p.environment = env
        p.standardInput = FileHandle.nullDevice
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        pipe.fileHandleForReading.readabilityHandler = { h in
            let d = h.availableData
            if d.isEmpty {
                h.readabilityHandler = nil
            } else {
                onOutput(String(decoding: d, as: UTF8.self))
            }
        }
        onOutput("$ \(executable.lastPathComponent) \(arguments.joined(separator: " "))\n")
        let status: Int32 = try await withCheckedThrowingContinuation { cont in
            p.terminationHandler = { proc in cont.resume(returning: proc.terminationStatus) }
            do {
                try p.run()
            } catch {
                p.terminationHandler = nil
                cont.resume(throwing: error)
                return
            }
            lock.lock()
            process = p
            lock.unlock()
        }
        lock.withLock { process = nil }
        return status
    }

    public func cancel() {
        lock.lock()
        let p = process
        lock.unlock()
        if let p, p.isRunning { p.terminate() }
    }
}
