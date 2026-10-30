import AppKit
import Foundation
import Network
import TridaChatCore

/// A local control API so a program (trida-mlx-simuser: an LLM playing the user) can use the app like a
/// person: open a chat, send a message, wait for the whole turn (tool routing, tools, reply check, rewrite),
/// and read what happened. Everything goes through the normal app path, so chats and request logs record it
/// as usual, and approvals still show the normal sheet.
///
/// Off by default (Settings → Agent). Listens on 127.0.0.1 only and needs the token stored in
/// ~/Library/Application Support/Zoro-1 Chat/automation-token (readable by your user only):
///   GET  /status                       {ready, model, generating, agent_mode}
///   POST /chats   {"agent_mode"?: bool} → {id}       a new chat (selected in the window)
///   POST /chats/<id>/messages {"text"} → {messages}  sends it and returns every message the turn added
///   GET  /chats/<id>                   → {title, messages}
final class AutomationServer: @unchecked Sendable {
    static let port: UInt16 = 8765
    static let tokenFile = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("Zoro-1 Chat/automation-token")

    private var listener: NWListener?
    private let queue = DispatchQueue(label: "zoro.automation")
    private weak var state: AppState?
    private let token: String

    init(state: AppState) {
        self.state = state
        token = Self.loadOrCreateToken()
    }

    static func loadOrCreateToken() -> String {
        if let t = try? String(contentsOf: tokenFile, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines),
           t.count >= 32 { return t }
        let t = (0..<4).map { _ in UUID().uuidString.replacingOccurrences(of: "-", with: "") }.joined()
        try? FileManager.default.createDirectory(at: tokenFile.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? t.write(to: tokenFile, atomically: true, encoding: .utf8)
        try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: tokenFile.path)
        return t
    }

    func start() throws {
        guard listener == nil else { return }
        let params = NWParameters.tcp
        params.requiredLocalEndpoint = NWEndpoint.hostPort(host: "127.0.0.1", port: NWEndpoint.Port(rawValue: Self.port)!)
        let l = try NWListener(using: params)
        l.newConnectionHandler = { [weak self] c in self?.accept(c) }
        l.start(queue: queue)
        listener = l
    }

    func stop() {
        listener?.cancel()
        listener = nil
    }

    // MARK: - HTTP

    private func accept(_ c: NWConnection) {
        c.start(queue: queue)
        read(c, Data())
    }

    private func read(_ c: NWConnection, _ buf: Data) {
        c.receive(minimumIncompleteLength: 1, maximumLength: 1 << 20) { [weak self] data, _, done, err in
            guard let self else { return }
            var b = buf
            if let data { b.append(data) }
            if let req = Request(b) {
                self.handle(req) { status, obj in self.respond(c, status, obj) }
            } else if done || err != nil || b.count > 32 << 20 {
                c.cancel()
            } else {
                self.read(c, b)
            }
        }
    }

    private func respond(_ c: NWConnection, _ status: Int, _ obj: Any) {
        let body = (try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys])) ?? Data("{}".utf8)
        let reason = [200: "OK", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found", 409: "Conflict",
                      503: "Service Unavailable"][status] ?? "Error"
        var head = "HTTP/1.1 \(status) \(reason)\r\nContent-Type: application/json\r\n"
        head += "Content-Length: \(body.count)\r\nConnection: close\r\n\r\n"
        c.send(content: Data(head.utf8) + body, completion: .contentProcessed { _ in c.cancel() })
    }

    struct Request {
        var method: String
        var path: String
        var headers: [String: String]
        var body: Data

        /// Parses a complete request, or nil while bytes are still missing.
        init?(_ d: Data) {
            guard let end = d.range(of: Data("\r\n\r\n".utf8)) else { return nil }
            let head = String(decoding: d[..<end.lowerBound], as: UTF8.self).split(separator: "\r\n").map(String.init)
            guard let first = head.first else { return nil }
            let parts = first.split(separator: " ").map(String.init)
            guard parts.count >= 2 else { return nil }
            method = parts[0]
            path = parts[1]
            var h: [String: String] = [:]
            for line in head.dropFirst() {
                if let i = line.firstIndex(of: ":") {
                    h[line[..<i].lowercased()] = line[line.index(after: i)...].trimmingCharacters(in: .whitespaces)
                }
            }
            headers = h
            let n = Int(h["content-length"] ?? "0") ?? 0
            let start = end.upperBound
            guard d.count - start >= n else { return nil }
            body = d[start..<(start + n)]
        }

        var json: [String: Any] { (try? JSONSerialization.jsonObject(with: body) as? [String: Any]) ?? [:] }
    }

    private func handle(_ r: Request, _ reply: @escaping @Sendable (Int, Any) -> Void) {
        guard r.headers["authorization"] == "Bearer \(token)" else {
            return reply(401, ["error": "missing or wrong token (\(Self.tokenFile.path))"])
        }
        let parts = r.path.split(separator: "/").map(String.init)
        let body = r.json
        let text = body["text"] as? String
        let agent = body["agent_mode"] as? Bool
        Task { @MainActor [weak self] in
            guard let state = self?.state else { return reply(503, ["error": "app is closing"]) }
            switch (r.method, parts.count) {
            case ("GET", 1) where parts[0] == "status":
                reply(200, state.automationStatus())
            case ("POST", 1) where parts[0] == "chats":
                reply(200, ["id": state.automationNewChat(agentMode: agent).uuidString])
            case ("GET", 2) where parts[0] == "chats":
                guard let id = UUID(uuidString: parts[1]), let chat = state.chats.first(where: { $0.id == id }) else {
                    return reply(404, ["error": "no such chat"])
                }
                reply(200, ["title": chat.title, "messages": chat.messages.map(AppState.automationJSON)] as [String: Any])
            case ("POST", 3) where parts[0] == "chats" && parts[2] == "messages":
                guard let id = UUID(uuidString: parts[1]) else { return reply(404, ["error": "no such chat"]) }
                guard let text, !text.isEmpty else { return reply(400, ["error": "missing text"]) }
                let (code, obj) = await state.automationSend(id, text)
                reply(code, obj)
            default:
                reply(404, ["error": "unknown endpoint \(r.method) \(r.path)"])
            }
        }
    }
}

extension AppState {
    func automationStatus() -> [String: Any] {
        ["ready": status.isUsable, "model": servedName, "generating": isGenerating,
         "agent_mode": settings.agentMode, "decision": supportsDecision]
    }

    func automationNewChat(agentMode: Bool?) -> Chat.ID {
        if let agentMode { settings.agentMode = agentMode }
        let c = Chat()
        chats.insert(c, at: 0)
        selectedChatID = c.id
        return c.id
    }

    /// Sends `text` in chat `id` the way the composer does and waits for the whole turn to finish.
    func automationSend(_ id: Chat.ID, _ text: String) async -> (Int, Any) {
        guard let ci = chats.firstIndex(where: { $0.id == id }) else { return (404, ["error": "no such chat"]) }
        guard status.isUsable else { return (503, ["error": "the model server is not ready"]) }
        // wait for anything still running (another turn, a summary)
        for _ in 0..<6000 where isGenerating { try? await Task.sleep(nanoseconds: 100_000_000) }
        guard !isGenerating else { return (409, ["error": "the app is busy"]) }
        selectedChatID = id
        let before = Set(chats[ci].messages.map(\.id))
        send(text, images: [])
        // the turn starts on the next run-loop pass; then wait until it is done (up to an hour)
        try? await Task.sleep(nanoseconds: 300_000_000)
        for _ in 0..<36000 where isGenerating { try? await Task.sleep(nanoseconds: 100_000_000) }
        guard let chat = chats.first(where: { $0.id == id }) else { return (404, ["error": "chat was deleted"]) }
        let added = chat.messages.filter { !before.contains($0.id) }.map(Self.automationJSON)
        let result: [String: Any] = ["messages": added, "busy": isGenerating, "decide_flow_open": decideFlow?.chatID == id]
        return (200, result)
    }

    static func automationJSON(_ m: ChatMessage) -> [String: Any] {
        var o: [String: Any] = ["id": m.id.uuidString, "role": m.role.rawValue, "content": m.content]
        if !m.reasoning.isEmpty { o["reasoning"] = m.reasoning }
        if let r = m.route { o["route"] = r }
        if let c = m.check { o["check"] = c }
        if m.superseded == true { o["superseded"] = true }
        if let e = m.error { o["error"] = e }
        if m.decide != nil { o["app_notice"] = true }
        if !m.images.isEmpty { o["images"] = m.images.count }
        if let n = m.toolName { o["tool_name"] = n }
        if !m.toolRuns.isEmpty {
            o["tool_runs"] = m.toolRuns.map { r -> [String: Any] in
                ["name": r.call.name, "arguments": r.call.arguments, "status": "\(r.status)",
                 "result": String(r.result.prefix(4000))]
            }
        }
        return o
    }
}
