import Foundation

/// Events produced while streaming one `/v1/chat/completions` response.
public enum StreamEvent: Equatable, Sendable {
    case prefill(done: Int, total: Int)   // `: prefill n/m` keep-alive comments during long prompts
    case content(String)
    case reasoning(String)
    case toolCalls([ToolCall])
    case finish(String)
    case usage(GenStats)
}

public enum ChatClientError: LocalizedError {
    case http(Int, String)
    case badURL

    public var errorDescription: String? {
        switch self {
        case let .http(code, body):
            if let data = body.data(using: .utf8),
               let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let err = obj["error"] as? [String: Any], let msg = err["message"] as? String {
                return "Server error \(code): \(msg)"
            }
            return "Server error \(code): \(body.prefix(500))"
        case .badURL:
            return "Bad server URL"
        }
    }
}

/// Parses the server's SSE stream one line at a time. Pure, so it is unit-tested.
public struct SSEParser {
    public init() {}

    /// Returns the events carried by one line; `nil` signals the end of the stream (`data: [DONE]`).
    public mutating func parse(line raw: String) -> [StreamEvent]? {
        let line = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if line.isEmpty { return [] }
        if line.hasPrefix(":") {
            // ": prefill 512/14000"
            let body = line.dropFirst().trimmingCharacters(in: .whitespaces)
            if body.hasPrefix("prefill ") {
                let nums = body.dropFirst("prefill ".count).split(separator: "/")
                if nums.count == 2, let d = Int(nums[0]), let t = Int(nums[1]) {
                    return [.prefill(done: d, total: t)]
                }
            }
            return []
        }
        guard line.hasPrefix("data:") else { return [] }
        let payload = line.dropFirst("data:".count).trimmingCharacters(in: .whitespaces)
        if payload == "[DONE]" { return nil }
        guard let data = payload.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return [] }
        var events: [StreamEvent] = []
        if let usage = obj["usage"] as? [String: Any] {
            events.append(.usage(GenStats(usage: usage)))
        }
        for choice in obj["choices"] as? [[String: Any]] ?? [] {
            let delta = choice["delta"] as? [String: Any] ?? [:]
            if let r = delta["reasoning_content"] as? String, !r.isEmpty { events.append(.reasoning(r)) }
            if let c = delta["content"] as? String, !c.isEmpty { events.append(.content(c)) }
            if let tcs = delta["tool_calls"] as? [[String: Any]], !tcs.isEmpty {
                let calls = tcs.compactMap { tc -> ToolCall? in
                    guard let fn = tc["function"] as? [String: Any], let name = fn["name"] as? String else { return nil }
                    let args: String
                    if let s = fn["arguments"] as? String {
                        args = s
                    } else if let o = fn["arguments"], JSONSerialization.isValidJSONObject(o),
                              let d = try? JSONSerialization.data(withJSONObject: o) {
                        args = String(decoding: d, as: UTF8.self)
                    } else {
                        args = "{}"
                    }
                    return ToolCall(id: tc["id"] as? String ?? "call_\(UUID().uuidString.prefix(8))", name: name, arguments: args)
                }
                if !calls.isEmpty { events.append(.toolCalls(calls)) }
            }
            if let f = choice["finish_reason"] as? String { events.append(.finish(f)) }
        }
        return events
    }
}

/// Minimal OpenAI-compatible streaming client for trida_mlx/server.py.
public struct ChatClient: Sendable {
    public var baseURL: URL   // e.g. http://127.0.0.1:8080/v1

    public init(baseURL: URL) {
        self.baseURL = baseURL
    }

    public static func requestBody(messages: [[String: Any]], tools: [[String: Any]]?,
                                   options: GenerationOptions, model: String?) -> [String: Any] {
        var body: [String: Any] = [
            "messages": messages,
            "stream": true,
            "stream_options": ["include_usage": true],
            "temperature": options.temperature,
            "top_k": options.topK,
            "top_p": options.topP,
            "max_tokens": options.maxTokens,
            "trida_mode": options.selfSpec ? "self-spec" : "causal",
            "chat_template_kwargs": ["enable_thinking": options.thinking],
        ]
        if let model { body["model"] = model }
        if let tools, !tools.isEmpty { body["tools"] = tools }
        return body
    }

    /// JSON for a request body with sorted keys. Swift dictionaries have no stable key order, and
    /// the chat template prints the tool schemas exactly as received, so unsorted keys would make
    /// every prompt differ at the tools block and defeat the server's prompt cache.
    public static func encode(_ body: [String: Any]) throws -> Data {
        try JSONSerialization.data(withJSONObject: body, options: [.sortedKeys])
    }

    /// Streams one completion. Cancelling the consuming task closes the connection, which
    /// stops generation on the server.
    public func stream(body: [String: Any]) -> AsyncThrowingStream<StreamEvent, Error> {
        let url = baseURL.appendingPathComponent("chat/completions")
        let payload: Data
        do {
            payload = try Self.encode(body)
        } catch {
            return AsyncThrowingStream { $0.finish(throwing: error) }
        }
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var req = URLRequest(url: url)
                    req.httpMethod = "POST"
                    req.setValue("application/json", forHTTPHeaderField: "Content-Type")
                    req.setValue("text/event-stream", forHTTPHeaderField: "Accept")
                    req.timeoutInterval = 3600
                    req.httpBody = payload
                    let (bytes, response) = try await URLSession.shared.bytes(for: req)
                    if let http = response as? HTTPURLResponse, http.statusCode != 200 {
                        var text = ""
                        for try await line in bytes.lines { text += line + "\n" }
                        throw ChatClientError.http(http.statusCode, text)
                    }
                    var parser = SSEParser()
                    for try await line in bytes.lines {
                        guard let events = parser.parse(line: line) else { break }
                        for e in events { continuation.yield(e) }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// GET /v1/models — the served model card, or nil if the server is not up.
    public func modelCard() async -> [String: Any]? {
        var req = URLRequest(url: baseURL.appendingPathComponent("models"))
        req.timeoutInterval = 2
        guard let result = try? await URLSession.shared.data(for: req),
              (result.1 as? HTTPURLResponse)?.statusCode == 200,
              let obj = try? JSONSerialization.jsonObject(with: result.0) as? [String: Any],
              let list = obj["data"] as? [[String: Any]] else { return nil }
        return list.first
    }
}

/// POST /v1/decide: typed decisions through the checkpoint's pointer head (trida_mlx/decision.py).
/// JSON whose object keys keep their order. Decision requests need it: the state is rendered for the model
/// in key order, and a choice question's options are its criteria in order (Swift dictionaries have none).
public indirect enum OrderedJSON: Sendable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case null
    case array([OrderedJSON])
    case object([(String, OrderedJSON)])

    public static func strings(_ xs: [String]) -> OrderedJSON { .array(xs.map { .string($0) }) }

    public func write(into out: inout String) {
        switch self {
        case let .string(s): out += Self.quote(s)
        case let .number(d): out += d == d.rounded() && abs(d) < 1e15 ? String(Int64(d)) : String(d)
        case let .bool(b): out += b ? "true" : "false"
        case .null: out += "null"
        case let .array(xs):
            out += "["
            for (i, x) in xs.enumerated() {
                if i > 0 { out += "," }
                x.write(into: &out)
            }
            out += "]"
        case let .object(kv):
            out += "{"
            for (i, (k, v)) in kv.enumerated() {
                if i > 0 { out += "," }
                out += Self.quote(k) + ":"
                v.write(into: &out)
            }
            out += "}"
        }
    }

    public var data: Data {
        var s = ""
        write(into: &s)
        return Data(s.utf8)
    }

    static func quote(_ s: String) -> String {
        var o = "\""
        for u in s.unicodeScalars {
            switch u {
            case "\"": o += "\\\""
            case "\\": o += "\\\\"
            case "\n": o += "\\n"
            case "\r": o += "\\r"
            case "\t": o += "\\t"
            default:
                if u.value < 0x20 { o += String(format: "\\u%04x", u.value) } else { o.unicodeScalars.append(u) }
            }
        }
        return o + "\""
    }
}

public extension ChatClient {
    /// POST /v1/decide. Throws ChatClientError on a non-200 reply (400: no decision head, bad request).
    func decide(_ body: OrderedJSON) async throws -> [String: Any] {
        var req = URLRequest(url: baseURL.appendingPathComponent("decide"))
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.timeoutInterval = 900
        req.httpBody = body.data
        let (data, resp) = try await URLSession.shared.data(for: req)
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        guard code == 200 else { throw ChatClientError.http(code, String(decoding: data, as: UTF8.self)) }
        guard let obj = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw ChatClientError.http(code, "not a JSON object")
        }
        return obj
    }
}

/// One choice question's result from /v1/decide, flattened for display.
public struct ChoiceDecision: Sendable {
    public var choice: String
    /// (option key, probability), most likely first.
    public var ranked: [(String, Double)]
    public var escalated: Bool
    public var reason: String?
    public var reasoning: String?
    public var cut: String?
    public var timingMs: Double

    public var top: Double { ranked.first?.1 ?? 0 }

    public init?(response r: [String: Any], question: String) {
        guard let answers = r["answers"] as? [String: Any], let a = answers[question] as? [String: Any],
              let choice = a["choice"] as? String,
              let probs = a["probabilities"] as? [String: Any] else { return nil }
        self.choice = choice
        ranked = probs.compactMap { k, v in GenStats.double(v).map { (k, $0) } }.sorted { $0.1 > $1.1 }
        escalated = r["escalated"] as? Bool ?? false
        reason = r["reason"] as? String
        reasoning = r["reasoning"] as? String
        cut = r["cut"] as? String
        let direct = r["direct"] as? [String: Any]
        let reasoned = r["reasoned"] as? [String: Any]
        timingMs = (GenStats.double(direct?["timing_ms"]) ?? 0) + (GenStats.double(reasoned?["timing_ms"]) ?? 0)
    }

    /// From a parsed result (the streamed path).
    public init?(result r: DecisionResult, question: String) {
        guard let a = r.answers[question], a.type == .choice else { return nil }
        choice = a.pick
        ranked = a.options.map { ($0.key, $0.p) }.sorted { $0.1 > $1.1 }
        escalated = r.escalated
        reason = r.reason
        reasoning = r.reasoning
        cut = r.cut
        timingMs = r.timingMs
    }

    /// e.g. "p 0.83" or "p 0.91 after reasoning (direct top 0.55)".
    public var summary: String {
        var s = String(format: "p %.2f", top)
        if escalated {
            s += cut == nil ? " after reasoning" : " (reasoning cut off; direct answer used)"
            if let reason { s += " — \(reason)" }
        }
        return s
    }
}
