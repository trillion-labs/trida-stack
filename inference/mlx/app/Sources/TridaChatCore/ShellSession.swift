import Foundation

/// The agent's terminal: run_shell commands of one chat share a working directory and exported environment,
/// like one terminal tab. Each command runs in a fresh `zsh -l`; before it, the session's saved directory and
/// exports are restored, and after it (also on `exit`) the new ones are saved. Aliases, functions and
/// non-exported variables do not carry over.
public final class ShellSession: @unchecked Sendable {
    private static let lock = NSLock()
    private static var sessions: [String: ShellSession] = [:]

    /// The session for `id` (a chat), created in `root` the first time.
    public static func session(_ id: String, root: URL) -> ShellSession {
        lock.lock()
        defer { lock.unlock() }
        if let s = sessions[id] { return s }
        let s = ShellSession(root: root)
        sessions[id] = s
        return s
    }

    public static let defaultTimeout: TimeInterval = 300
    public static let maxTimeout: TimeInterval = 1800
    /// Output longer than head + tail characters keeps its start and end.
    public static let headChars = 4000, tailChars = 8000

    private let dir: URL
    private let cwdFile: URL
    private let envFile: URL

    init(root: URL) {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("zoro-shell-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        cwdFile = dir.appendingPathComponent("cwd")
        envFile = dir.appendingPathComponent("env")
        try? root.path.write(to: cwdFile, atomically: true, encoding: .utf8)
    }

    /// The directory the next command starts in.
    public var cwd: String {
        (try? String(contentsOf: cwdFile, encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }

    static func quote(_ s: String) -> String { "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'" }

    /// Runs `command`; returns its combined output (start and end kept when long) and the exit status.
    public func run(_ command: String, timeout: TimeInterval = defaultTimeout) async throws -> String {
        let t = min(max(timeout, 1), Self.maxTimeout)
        let out = dir.appendingPathComponent("out-\(UUID().uuidString).log")
        FileManager.default.createFile(atPath: out.path, contents: nil)
        defer { try? FileManager.default.removeItem(at: out) }
        let cwd = Self.quote(self.cwd), cf = Self.quote(cwdFile.path), ef = Self.quote(envFile.path)
        // Restore, run, save (the trap also saves after `exit`).
        let script = """
            [ -f \(ef) ] && source \(ef) 2>/dev/null
            cd \(cwd) 2>/dev/null || cd ~
            trap 'pwd > \(cf); export -p > \(ef)' EXIT
            \(command)
            """
        let handle = try FileHandle(forWritingTo: out)
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/zsh")
        p.arguments = ["-lc", script]
        p.standardOutput = handle
        p.standardError = handle
        p.standardInput = FileHandle.nullDevice
        final class Flag: @unchecked Sendable { var on = false }
        let timedOut = Flag()
        let status: Int32 = try await withCheckedThrowingContinuation { cont in
            p.terminationHandler = { proc in cont.resume(returning: proc.terminationStatus) }
            do {
                try p.run()
            } catch {
                p.terminationHandler = nil
                cont.resume(throwing: error)
                return
            }
            DispatchQueue.global().asyncAfter(deadline: .now() + t) {
                if p.isRunning { timedOut.on = true; p.terminate() }
            }
        }
        try? handle.close()
        let text = (try? String(contentsOf: out, encoding: .utf8)) ?? ""
        var body = Self.clip(text)
        if timedOut.on { body += (body.isEmpty ? "" : "\n") + "(stopped after \(Int(t)) s; pass a larger timeout, up to \(Int(Self.maxTimeout)))" }
        let tail = "(exit \(status); cwd \(self.cwd))"
        return body.isEmpty ? tail : body + "\n" + tail
    }

    /// The first headChars and last tailChars of long output, with a note of what was left out.
    public static func clip(_ s: String) -> String {
        guard s.count > headChars + tailChars else { return s }
        let omitted = s.count - headChars - tailChars
        return String(s.prefix(headChars)) + "\n… \(omitted) characters omitted …\n" + String(s.suffix(tailChars))
    }

    // MARK: - read-only commands

    static let readOnlyCommands: Set<String> = [
        "ls", "pwd", "cat", "head", "tail", "wc", "grep", "egrep", "fgrep", "rg", "which", "whoami", "date", "echo",
        "file", "stat", "du", "df", "tree", "uname", "printenv", "sw_vers", "cd", "less", "more", "sort", "uniq",
        "cut", "tr", "nl", "basename", "dirname", "realpath", "readlink", "id", "hostname", "uptime", "ps", "top",
        "diff", "cmp", "md5", "shasum", "md5sum", "sha256sum", "type", "true", "jq", "column", "fd",
    ]
    static let readOnlyGit: Set<String> = [
        "status", "log", "diff", "show", "branch", "ls-files", "rev-parse", "blame", "describe", "shortlog", "tag",
        "remote", "config", "grep", "reflog", "ls-tree", "cat-file",
    ]

    /// Whether `command` only reads: every piped / chained part starts with a known read-only command, with
    /// no redirection to files, command substitution, background jobs or sudo. Such commands run without
    /// asking when the user allows it (Settings → Agent).
    public static func isReadOnly(_ command: String) -> Bool {
        let c = command.trimmingCharacters(in: .whitespacesAndNewlines)
        if c.isEmpty || c.contains("\n") { return false }
        for bad in [">", "<(", "$(", "`", "sudo", "eval ", "exec ", "xargs"] where c.contains(bad) {
            return false
        }
        // split on pipes and chains
        var parts = [c]
        for sep in ["&&", "||", ";", "|"] { parts = parts.flatMap { $0.components(separatedBy: sep) } }
        for part in parts {
            if part.contains("&") { return false }          // a background job (`&&` was split off above)
            let words = part.split(whereSeparator: { $0 == " " || $0 == "\t" }).map(String.init)
            guard let first = words.first else { return false }
            if first.contains("=") { return false }        // VAR=... cmd
            let name = (first as NSString).lastPathComponent
            if name == "git" {
                guard words.count > 1, readOnlyGit.contains(words[1]) else { return false }
                if ["branch", "tag", "remote", "config"].contains(words[1]),
                   words.dropFirst(2).contains(where: { !$0.hasPrefix("-") || ["-d", "-D", "-m", "-M", "--delete", "--unset", "--add"].contains($0) }),
                   !(words[1] == "config" && words.dropFirst(2).allSatisfy { $0.hasPrefix("--get") || $0 == "--list" || $0 == "-l" }) {
                    return false                           // e.g. `git branch new`, `git config user.name x`
                }
                continue
            }
            if name == "find" {
                if words.contains(where: { ["-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fls"].contains($0) }) { return false }
                continue
            }
            if words.contains("--version") || words.contains("-V") || words.contains("--help") { continue }
            guard readOnlyCommands.contains(name) else { return false }
        }
        return true
    }
}
