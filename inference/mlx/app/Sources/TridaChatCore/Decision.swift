import Foundation

/// Typed decisions (POST /v1/decide, trida_mlx/decision.py): build requests, stream them, read the answers.
/// One question type per kind of answer: `choice` (one of named options), `noul` (yes / no), `score`
/// (a level on an ordered scale).
public enum DecisionType: String, CaseIterable, Sendable {
    case choice, noul, score

    public var label: String {
        switch self {
        case .choice: return "Choice"
        case .noul: return "Yes / no"
        case .score: return "Score"
        }
    }
}

public struct DecisionQuestion: Sendable, Codable {
    public var key: String
    public var type: DecisionType
    public var instructions: String
    /// choice: (key, description) in order; score: the levels in order (keys unused); noul: unused.
    public var options: [(String, String)]

    private enum CodingKeys: String, CodingKey { case key, type, instructions, options }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        key = try c.decode(String.self, forKey: .key)
        type = DecisionType(rawValue: try c.decode(String.self, forKey: .type)) ?? .choice
        instructions = try c.decode(String.self, forKey: .instructions)
        options = try c.decode([[String]].self, forKey: .options).map { ($0.first ?? "", $0.count > 1 ? $0[1] : "") }
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(key, forKey: .key)
        try c.encode(type.rawValue, forKey: .type)
        try c.encode(instructions, forKey: .instructions)
        try c.encode(options.map { [$0.0, $0.1] }, forKey: .options)
    }

    public init(key: String, type: DecisionType, instructions: String, options: [(String, String)] = []) {
        self.key = key
        self.type = type
        self.instructions = instructions
        self.options = options
    }

    public static func choice(_ key: String, _ instructions: String, _ options: [(String, String)]) -> DecisionQuestion {
        DecisionQuestion(key: key, type: .choice, instructions: instructions, options: options)
    }

    public static func yesNo(_ key: String, _ instructions: String) -> DecisionQuestion {
        DecisionQuestion(key: key, type: .noul, instructions: instructions)
    }

    public static func score(_ key: String, _ instructions: String, levels: [String]) -> DecisionQuestion {
        DecisionQuestion(key: key, type: .score, instructions: instructions, options: levels.map { ($0, $0) })
    }

    /// Option lines as typed in the Decide panel: "key: description" or just "key" (choice), one level per
    /// line (score). Blank lines are skipped.
    public static func parseOptions(_ text: String, type: DecisionType) -> [(String, String)] {
        let lines = text.split(separator: "\n").map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        if type != .choice { return lines.map { ($0, $0) } }
        return lines.map { line in
            if let r = line.range(of: ": ") {
                return (String(line[..<r.lowerBound]).trimmingCharacters(in: .whitespaces),
                        String(line[r.upperBound...]).trimmingCharacters(in: .whitespaces))
            }
            return (line, "")
        }
    }

    /// Option keys the server reports probabilities under, in option order.
    public var answerKeys: [String] {
        switch type {
        case .choice: return options.map(\.0)
        case .noul: return ["false", "true"]
        case .score: return options.indices.map(String.init)
        }
    }

    public var json: OrderedJSON {
        var q: [(String, OrderedJSON)] = [("type", .string(type.rawValue)), ("instructions", .string(instructions))]
        switch type {
        case .choice: q.append(("criteria", .object(options.map { ($0.0, OrderedJSON.string($0.1)) })))
        case .score: q.append(("criteria", .strings(options.map(\.0))))
        case .noul: break
        }
        return .object(q)
    }
}

public struct DecisionSettings: Sendable {
    /// "auto": reason when the top probability is below `threshold`; "always"; "never".
    public var reasoning = "auto"
    public var threshold = 0.7
    public var maxReasoningTokens = 8192
    public var selfSpec = true
    public var maxImagePixels = 448 * 448

    public init(reasoning: String = "auto", threshold: Double = 0.7, maxReasoningTokens: Int = 8192, selfSpec: Bool = true) {
        self.reasoning = reasoning
        self.threshold = threshold
        self.maxReasoningTokens = maxReasoningTokens
        self.selfSpec = selfSpec
    }
}

public enum Decision {
    public static func body(state: OrderedJSON, questions: [DecisionQuestion], image: Data? = nil,
                            settings: DecisionSettings, stream: Bool = false) -> OrderedJSON {
        var b: [(String, OrderedJSON)] = [
            ("state", state),
            ("questions", .object(questions.map { ($0.key, $0.json) })),
            ("reasoning", .string(settings.reasoning)),
            ("threshold", .number(settings.threshold)),
            ("max_reasoning_tokens", .number(Double(settings.maxReasoningTokens))),
            ("trida_mode", .string(settings.selfSpec ? "self-spec" : "causal")),
        ]
        if let image {
            b.append(("image", .string("data:image/png;base64," + image.base64EncodedString())))
            b.append(("max_image_pixels", .number(Double(settings.maxImagePixels))))
        }
        if stream { b.append(("stream", .bool(true))) }
        return .object(b)
    }
}

/// One question's answer, with the probabilities in option order.
public struct DecisionAnswer: Sendable {
    public var type: DecisionType
    /// (option key, label shown, probability), in option order.
    public var options: [(key: String, label: String, p: Double)]
    /// choice: the picked key; noul: "yes" / "no"; score: the nearest level.
    public var pick: String
    /// noul: P(yes); score: the expected level index; choice: the top probability.
    public var value: Double

    /// The probability of the most likely option (what escalation compares to the threshold).
    public var top: Double { options.map(\.p).max() ?? 0 }

    public init?(_ a: [String: Any], question q: DecisionQuestion) {
        type = q.type
        switch q.type {
        case .noul:
            guard let y = GenStats.double(a["noul"]) else { return nil }
            options = [("false", "no", 1 - y), ("true", "yes", y)]
            pick = y >= 0.5 ? "yes" : "no"
            value = y
        case .choice:
            guard let probs = a["probabilities"] as? [String: Any] else { return nil }
            options = q.options.map { ($0.0, $0.1.isEmpty ? $0.0 : "\($0.0): \($0.1)", GenStats.double(probs[$0.0]) ?? 0) }
            pick = a["choice"] as? String ?? (options.max { $0.p < $1.p }?.key ?? "")
            value = options.map(\.p).max() ?? 0
        case .score:
            guard let probs = a["probabilities"] as? [String: Any] else { return nil }
            options = q.options.enumerated().map { (String($0.offset), $0.element.0, GenStats.double(probs[String($0.offset)]) ?? 0) }
            value = GenStats.double(a["score"]) ?? 0
            let i = min(max(Int(value.rounded()), 0), max(options.count - 1, 0))
            pick = options.isEmpty ? "" : options[i].label
        }
    }

    /// "billing (0.82)", "yes (0.91)", "medium (score 1.2 of 0–2)".
    public var summary: String {
        switch type {
        case .score: return "\(pick) (score \(String(format: "%.2f", value)) of 0–\(max(options.count - 1, 0)))"
        default: return "\(pick) (\(String(format: "%.2f", options.first { $0.label == pick || $0.key == pick }?.p ?? top)))"
        }
    }
}

/// A whole /v1/decide response.
public struct DecisionResult: Sendable {
    public var answers: [String: DecisionAnswer]
    public var direct: [String: DecisionAnswer]
    public var escalated: Bool
    public var reason: String?
    public var reasoning: String?
    public var cut: String?
    public var timingMs: Double
    /// The server's JSON for this result (saved with the chat; `init?(raw:questions:)` rebuilds it).
    public var raw: Data

    public init?(raw: Data, questions: [DecisionQuestion]) {
        guard let r = try? JSONSerialization.jsonObject(with: raw) as? [String: Any] else { return nil }
        self.init(r, questions: questions)
    }

    public init?(_ r: [String: Any], questions: [DecisionQuestion]) {
        raw = (try? JSONSerialization.data(withJSONObject: r)) ?? Data()
        func read(_ obj: Any?) -> [String: DecisionAnswer] {
            guard let answers = (obj as? [String: Any]) else { return [:] }
            var out: [String: DecisionAnswer] = [:]
            for q in questions {
                if let a = answers[q.key] as? [String: Any], let d = DecisionAnswer(a, question: q) { out[q.key] = d }
            }
            return out
        }
        answers = read(r["answers"])
        guard answers.count == questions.count else { return nil }
        let directObj = r["direct"] as? [String: Any]
        direct = read(directObj?["answers"])
        escalated = r["escalated"] as? Bool ?? false
        reason = r["reason"] as? String
        reasoning = r["reasoning"] as? String
        cut = r["cut"] as? String
        timingMs = (GenStats.double(directObj?["timing_ms"]) ?? 0)
            + (GenStats.double((r["reasoned"] as? [String: Any])?["timing_ms"]) ?? 0)
    }

    /// For tool results: how the answer came about.
    public var provenance: String {
        guard escalated else { return "direct decision" }
        if cut != nil { return "reasoning was cut off; direct decision used" }
        return "decided after reasoning (the direct decision was unsure)"
    }
}

/// Events of a streamed decision.
public enum DecisionEvent: Sendable {
    case direct(reason: String?)     // the direct decision is in; `reason` set when it escalates
    case reasoning(String)           // a piece of the reasoning text
    case result(DecisionResult)
}

public extension ChatClient {
    /// POST /v1/decide with "stream": true. Cancelling the consumer closes the connection, which stops the
    /// reasoning on the server.
    func decideStream(_ body: OrderedJSON, questions: [DecisionQuestion]) -> AsyncThrowingStream<DecisionEvent, Error> {
        let url = baseURL.appendingPathComponent("decide")
        // Always ask for SSE: a body built without "stream": true gets one plain JSON reply, which has no
        // "data:" lines, so the stream would end without a result.
        var streamed = body
        if case let .object(kv) = body {
            streamed = .object(kv.filter { $0.0 != "stream" } + [("stream", .bool(true))])
        }
        let payload = streamed.data
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
                    for try await raw in bytes.lines {
                        let line = raw.trimmingCharacters(in: .whitespaces)
                        guard line.hasPrefix("data:") else { continue }
                        let p = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
                        if p == "[DONE]" { break }
                        guard let obj = try? JSONSerialization.jsonObject(with: Data(p.utf8)) as? [String: Any] else { continue }
                        switch obj["type"] as? String {
                        case "direct": continuation.yield(.direct(reason: obj["reason"] as? String))
                        case "reasoning": continuation.yield(.reasoning(obj["text"] as? String ?? ""))
                        case "result":
                            guard let r = DecisionResult(obj, questions: questions) else {
                                throw ChatClientError.http(200, "unexpected decision result")
                            }
                            continuation.yield(.result(r))
                        case "error": throw ChatClientError.http(500, obj["message"] as? String ?? "decision failed")
                        default: break
                        }
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    /// Non-streamed decision, parsed.
    func decide(_ body: OrderedJSON, questions: [DecisionQuestion]) async throws -> DecisionResult {
        let r = try await decide(body)
        guard let d = DecisionResult(r, questions: questions) else {
            throw ChatClientError.http(200, "unexpected decision response")
        }
        return d
    }
}
