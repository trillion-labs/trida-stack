import Foundation

/// A converted (or raw) checkpoint directory under the models folder.
public struct LocalModel: Identifiable, Hashable, Sendable {
    public var url: URL
    public var name: String { url.lastPathComponent }
    /// e.g. "q8 g64", or "bf16" for an unquantized checkpoint.
    public var precision: String
    public var vision: Bool
    public var bytes: Int64
    public var id: String { url.path }

    public var sizeText: String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
}

public enum ModelCatalog {
    /// Every subdirectory of `dir` that holds a config.json and safetensors weights.
    public static func scan(_ dir: URL) -> [LocalModel] {
        let fm = FileManager.default
        guard let entries = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.isDirectoryKey],
                                                       options: [.skipsHiddenFiles]) else { return [] }
        var out: [LocalModel] = []
        for e in entries {
            guard (try? e.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true,
                  let data = try? Data(contentsOf: e.appendingPathComponent("config.json")),
                  let cfg = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let files = try? fm.contentsOfDirectory(atPath: e.path) else { continue }
            let weights = files.filter { $0.hasSuffix(".safetensors") }
            if weights.isEmpty { continue }
            var bytes: Int64 = 0
            for w in weights {
                let attrs = try? fm.attributesOfItem(atPath: e.appendingPathComponent(w).path)
                bytes += (attrs?[.size] as? NSNumber)?.int64Value ?? 0
            }
            let text = cfg["text_config"] as? [String: Any] ?? [:]
            let q = (cfg["quantization"] as? [String: Any]) ?? (text["quantization"] as? [String: Any])
            var precision = "bf16"
            if let q, let bits = q["bits"] {
                precision = "q\(bits)"
                if let g = q["group_size"] { precision += " g\(g)" }
            }
            out.append(LocalModel(url: e, precision: precision, vision: cfg["vision_config"] != nil, bytes: bytes))
        }
        // quantized first, then by name
        return out.sorted { a, b in
            let qa = a.precision != "bf16", qb = b.precision != "bf16"
            return qa != qb ? qa : a.name.localizedStandardCompare(b.name) == .orderedAscending
        }
    }

    /// Zoro-1 q8 if present, else the first quantized model, else anything.
    public static func preferred(_ models: [LocalModel]) -> LocalModel? {
        models.first { $0.name.lowercased().contains("zoro") && $0.precision != "bf16" }
            ?? models.first { $0.precision != "bf16" }
            ?? models.first
    }
}

/// Runs `trida-mlx-server` as a child process.
public final class ServerProcess: @unchecked Sendable {
    public struct Config: Sendable {
        public var runtime: PythonRuntime
        public var model: URL
        public var port: Int
        public var servedName: String
        public var extraArgs: [String]

        public init(runtime: PythonRuntime, model: URL, port: Int, servedName: String, extraArgs: [String] = []) {
            self.runtime = runtime
            self.model = model
            self.port = port
            self.servedName = servedName
            self.extraArgs = extraArgs
        }

        public var arguments: [String] {
            ["--model", model.path, "--host", "127.0.0.1", "--port", String(port),
             "--served-model-name", servedName] + extraArgs
        }
    }

    private let lock = NSLock()
    private var process: Process?
    /// Called (on a background queue) with each chunk of stdout/stderr.
    public var onOutput: (@Sendable (String) -> Void)?
    /// Called (on a background queue) when the process exits.
    public var onExit: (@Sendable (Int32) -> Void)?

    public init() {}

    public var isRunning: Bool {
        lock.lock(); defer { lock.unlock() }
        return process?.isRunning ?? false
    }

    static func shellQuote(_ s: String) -> String {
        "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    public func start(_ c: Config) throws {
        stop()
        try c.runtime.validate()
        let (exe, args) = c.runtime.server(c.arguments)
        let p = Process()
        p.executableURL = exe
        p.arguments = args
        p.currentDirectoryURL = c.runtime.workingDirectory
        var env = ProcessInfo.processInfo.environment
        env.removeValue(forKey: "VIRTUAL_ENV")
        env.removeValue(forKey: "PYTHONHOME")
        env.removeValue(forKey: "PYTHONPATH")
        for (k, v) in c.runtime.environment { env[k] = v }
        p.environment = env
        p.standardInput = FileHandle.nullDevice
        let pipe = Pipe()
        p.standardOutput = pipe
        p.standardError = pipe
        let out = onOutput
        pipe.fileHandleForReading.readabilityHandler = { h in
            let d = h.availableData
            if d.isEmpty {
                h.readabilityHandler = nil
            } else {
                out?(String(decoding: d, as: UTF8.self))
            }
        }
        let exited = onExit
        p.terminationHandler = { proc in exited?(proc.terminationStatus) }
        out?("[\(c.runtime.label)]\n$ \(exe.path) \(args.joined(separator: " "))\n")
        try p.run()
        lock.lock()
        process = p
        lock.unlock()
    }

    /// SIGTERM, then SIGKILL if the server has not exited after `grace` seconds.
    public func stop(grace: TimeInterval = 3) {
        lock.lock()
        let p = process
        process = nil
        lock.unlock()
        guard let p, p.isRunning else { return }
        p.terminate()
        let deadline = Date().addingTimeInterval(grace)
        while p.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.05)
        }
        if p.isRunning {
            kill(p.processIdentifier, SIGKILL)
        }
    }
}
