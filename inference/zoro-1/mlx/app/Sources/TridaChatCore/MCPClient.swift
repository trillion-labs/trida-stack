import Foundation

/// A Model Context Protocol server run as a child process and spoken to over stdio: newline-delimited JSON-RPC 2.0
/// (MCP 2025-06-18). Enough of the protocol for tools: `initialize`, `tools/list` (with paging) and `tools/call`;
/// the server's own `ping` is answered and its other requests are refused.
///
/// The command runs through a login shell (`/bin/zsh -lc`), so `npx`, `uvx` and `deno` resolve the way they do in
/// the user's Terminal (Homebrew, nvm, ~/.local/bin). Anything a profile prints on stdout is skipped: only lines
/// that parse as JSON-RPC are read.
public final class MCPClient: @unchecked Sendable {
    public struct Tool: Sendable, Equatable {
        public let name: String
        public let description: String
        /// The tool's JSON Schema for its arguments (`inputSchema`), as JSON.
        public let schema: Data

        public var schemaObject: [String: Any] {
            (try? JSONSerialization.jsonObject(with: schema) as? [String: Any]) ?? ["type": "object", "properties": [:]]
        }
    }

    public enum MCPError: LocalizedError {
        case notRunning
        case exited(String)
        case timeout(String)
        case rpc(String)

        public var errorDescription: String? {
            switch self {
            case .notRunning: return "the MCP server is not running"
            case let .exited(tail): return "the MCP server exited" + (tail.isEmpty ? "" : ": " + tail)
            case let .timeout(method): return "the MCP server did not answer \(method) in time"
            case let .rpc(message): return message
            }
        }
    }

    public static let protocolVersion = "2025-06-18"

    private let lock = NSLock()
    private var process: Process?
    private var stdin: FileHandle?
    private var pending: [Int: CheckedContinuation<Data, Error>] = [:]
    private var nextID = 1
    private var buffer = Data()
    private var errTail = ""

    public init() {}

    public var isRunning: Bool { lock.withLock { process?.isRunning ?? false } }
    /// The last few KB the server wrote on stderr (its own log; shown when it fails).
    public var stderrTail: String { lock.withLock { errTail } }

    /// Starts `command` (a shell command line) with `environment` added to the app's.
    public func start(command: String, environment: [String: String], cwd: URL) throws {
        signal(SIGPIPE, SIG_IGN)   // a server that exits mid-write must not take the app with it
        stop()
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/zsh")
        p.arguments = ["-lc", "exec " + command]
        p.currentDirectoryURL = cwd
        var env = ProcessInfo.processInfo.environment
        for key in ["VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH"] { env.removeValue(forKey: key) }
        for (k, v) in environment { env[k] = v }
        p.environment = env
        let inPipe = Pipe(), outPipe = Pipe(), errPipe = Pipe()
        p.standardInput = inPipe
        p.standardOutput = outPipe
        p.standardError = errPipe
        outPipe.fileHandleForReading.readabilityHandler = { [weak self] h in
            let d = h.availableData
            if d.isEmpty { h.readabilityHandler = nil } else { self?.received(d) }
        }
        errPipe.fileHandleForReading.readabilityHandler = { [weak self] h in
            let d = h.availableData
            if d.isEmpty { h.readabilityHandler = nil; return }
            guard let self else { return }
            self.lock.withLock {
                self.errTail += String(decoding: d, as: UTF8.self)
                if self.errTail.count > 6000 { self.errTail = String(self.errTail.suffix(4000)) }
            }
        }
        p.terminationHandler = { [weak self] _ in self?.failAll(MCPError.exited(self?.lastErrorLine ?? "")) }
        try p.run()
        lock.withLock {
            process = p
            stdin = inPipe.fileHandleForWriting
            buffer = Data()
        }
    }

    public func stop() {
        let p: Process? = lock.withLock {
            let p = process
            process = nil
            stdin = nil
            return p
        }
        if let p, p.isRunning { p.terminate() }
        failAll(MCPError.notRunning)
    }

    /// The MCP handshake. Returns the server's "name version".
    public func initialize(timeout: Double = 120) async throws -> String {
        let result = try await request("initialize", [
            "protocolVersion": Self.protocolVersion,
            "capabilities": [String: Any](),
            "clientInfo": ["name": "Zoro-1 Chat", "version": "1.0"],
        ], timeout: timeout)
        try notify("notifications/initialized")
        let info = (try? JSONSerialization.jsonObject(with: result) as? [String: Any])?["serverInfo"] as? [String: Any]
        return [info?["name"] as? String, info?["version"] as? String].compactMap { $0 }.joined(separator: " ")
    }

    public func listTools(timeout: Double = 60) async throws -> [Tool] {
        var tools: [Tool] = []
        var cursor: String? = nil
        repeat {
            let params: [String: Any] = cursor.map { ["cursor": $0] } ?? [:]
            let data = try await request("tools/list", params, timeout: timeout)
            let obj = (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
            for t in obj["tools"] as? [[String: Any]] ?? [] {
                guard let name = t["name"] as? String else { continue }
                let schema = (t["inputSchema"] as? [String: Any]) ?? ["type": "object", "properties": [String: Any]()]
                tools.append(Tool(name: name, description: (t["description"] as? String ?? "")
                                     .trimmingCharacters(in: .whitespacesAndNewlines),
                                  schema: (try? JSONSerialization.data(withJSONObject: schema)) ?? Data("{}".utf8)))
            }
            cursor = obj["nextCursor"] as? String
        } while cursor != nil && tools.count < 500
        return tools
    }

    /// Calls a tool. The result's text parts are joined; images and other parts are named, not inlined.
    public func callTool(_ name: String, arguments: [String: Any], timeout: Double = 120) async throws
        -> (text: String, isError: Bool) {
        let data = try await request("tools/call", ["name": name, "arguments": arguments], timeout: timeout)
        let obj = (try? JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
        var parts: [String] = []
        for c in obj["content"] as? [[String: Any]] ?? [] {
            switch c["type"] as? String {
            case "text": parts.append(c["text"] as? String ?? "")
            case "image": parts.append("[image: \(c["mimeType"] as? String ?? "image")]")
            case "resource":
                let r = c["resource"] as? [String: Any]
                parts.append(r?["text"] as? String ?? "[resource: \(r?["uri"] as? String ?? "?")]")
            case "resource_link": parts.append("[link: \(c["uri"] as? String ?? "?")]")
            default: break
            }
        }
        if parts.isEmpty, let structured = obj["structuredContent"],
           let d = try? JSONSerialization.data(withJSONObject: structured, options: [.sortedKeys]) {
            parts.append(String(decoding: d, as: UTF8.self))
        }
        return (parts.joined(separator: "\n"), obj["isError"] as? Bool ?? false)
    }

    // MARK: - JSON-RPC

    private func request(_ method: String, _ params: [String: Any], timeout: Double) async throws -> Data {
        let id: Int = lock.withLock {
            defer { nextID += 1 }
            return nextID
        }
        let line = try JSONSerialization.data(withJSONObject: ["jsonrpc": "2.0", "id": id, "method": method,
                                                               "params": params]) + Data([0x0A])
        let timer = Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(timeout * 1_000_000_000))
            if !Task.isCancelled { self?.fail(id, MCPError.timeout(method)) }
        }
        defer { timer.cancel() }
        return try await withCheckedThrowingContinuation { cont in
            let handle: FileHandle? = lock.withLock {
                pending[id] = cont
                return stdin
            }
            do {
                guard let handle else { throw MCPError.notRunning }
                try handle.write(contentsOf: line)
            } catch {
                fail(id, error)
            }
        }
    }

    private func notify(_ method: String) throws {
        let line = try JSONSerialization.data(withJSONObject: ["jsonrpc": "2.0", "method": method]) + Data([0x0A])
        guard let handle = lock.withLock({ stdin }) else { throw MCPError.notRunning }
        try handle.write(contentsOf: line)
    }

    private func respond(_ id: Any, result: [String: Any]? = nil, error: (Int, String)? = nil) {
        var msg: [String: Any] = ["jsonrpc": "2.0", "id": id]
        if let error { msg["error"] = ["code": error.0, "message": error.1] } else { msg["result"] = result ?? [:] }
        guard let data = try? JSONSerialization.data(withJSONObject: msg),
              let handle = lock.withLock({ stdin }) else { return }
        try? handle.write(contentsOf: data + Data([0x0A]))
    }

    private func received(_ data: Data) {
        var lines: [Data] = []
        lock.withLock {
            buffer.append(data)
            while let nl = buffer.firstIndex(of: 0x0A) {
                lines.append(buffer[buffer.startIndex..<nl])
                buffer.removeSubrange(buffer.startIndex...nl)
            }
        }
        for line in lines {
            guard let obj = try? JSONSerialization.jsonObject(with: line) as? [String: Any] else { continue }
            if let method = obj["method"] as? String {
                guard let id = obj["id"] else { continue }   // a notification: nothing to do
                if method == "ping" { respond(id) } else { respond(id, error: (-32601, "not supported: \(method)")) }
                continue
            }
            guard let id = (obj["id"] as? NSNumber)?.intValue else { continue }
            let cont = lock.withLock { pending.removeValue(forKey: id) }
            if let err = obj["error"] as? [String: Any] {
                cont?.resume(throwing: MCPError.rpc(err["message"] as? String ?? "error \(err["code"] ?? "")"))
            } else {
                let result = obj["result"] ?? [String: Any]()
                cont?.resume(returning: (try? JSONSerialization.data(withJSONObject: result)) ?? Data("{}".utf8))
            }
        }
    }

    private func fail(_ id: Int, _ error: Error) {
        lock.withLock { pending.removeValue(forKey: id) }?.resume(throwing: error)
    }

    private func failAll(_ error: Error) {
        let all = lock.withLock {
            let p = pending
            pending = [:]
            return p
        }
        for c in all.values { c.resume(throwing: error) }
    }

    private var lastErrorLine: String {
        stderrTail.split(separator: "\n").last.map { String($0.prefix(300)) } ?? ""
    }
}
