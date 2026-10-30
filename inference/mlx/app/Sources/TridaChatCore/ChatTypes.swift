import Foundation

/// One tool call requested by the model (OpenAI `tool_calls[]` entry).
public struct ToolCall: Identifiable, Equatable, Sendable, Codable {
    public var id: String
    public var name: String
    /// JSON-encoded arguments, exactly as the server sent them.
    public var arguments: String

    public init(id: String, name: String, arguments: String) {
        self.id = id
        self.name = name
        self.arguments = arguments
    }

    /// Arguments decoded into a dictionary (empty when they are not a JSON object).
    public var argumentDict: [String: Any] {
        guard let data = arguments.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return [:] }
        return obj
    }
}

/// Decode statistics the server returns in `usage.trida` (see trida_mlx/decode.py DecodeStats).
public struct GenStats: Equatable, Sendable, Codable {
    public var mode: String = ""
    public var promptTokens: Int = 0
    public var cachedTokens: Int = 0
    public var newTokens: Int = 0
    public var forwards: Int = 0
    public var tokensPerForward: Double = 0
    public var prefillSeconds: Double = 0
    public var decodeSeconds: Double = 0
    public var decodeTokensPerSecond: Double = 0
    public var finishReason: String = ""

    public init() {}

    public init(usage: [String: Any]) {
        let t = usage["trida"] as? [String: Any] ?? [:]
        mode = t["mode"] as? String ?? ""
        promptTokens = Self.int(t["prompt_tokens"]) ?? Self.int(usage["prompt_tokens"]) ?? 0
        cachedTokens = Self.int(t["reused_tokens"]) ?? 0
        newTokens = Self.int(t["new_tokens"]) ?? Self.int(usage["completion_tokens"]) ?? 0
        forwards = Self.int(t["forwards"]) ?? 0
        tokensPerForward = Self.double(t["tokens_per_forward"]) ?? 0
        prefillSeconds = Self.double(t["prefill_s"]) ?? 0
        decodeSeconds = Self.double(t["decode_s"]) ?? 0
        decodeTokensPerSecond = Self.double(t["decode_tok_s"]) ?? 0
        finishReason = t["finish_reason"] as? String ?? ""
    }

    /// e.g. "self-spec(N=4) · 52.4 tok/s · 2.40 tok/fwd · 300 tok · prefill 0.14 s (4,812 cached)"
    public var summary: String {
        var parts: [String] = []
        if !mode.isEmpty { parts.append(mode) }
        parts.append(String(format: "%.1f tok/s", decodeTokensPerSecond))
        if tokensPerForward > 0 { parts.append(String(format: "%.2f tok/fwd", tokensPerForward)) }
        parts.append("\(newTokens) tok")
        var pre = String(format: "prefill %.2f s", prefillSeconds)
        if cachedTokens > 0 { pre += " (\(cachedTokens.formatted()) of \(promptTokens.formatted()) cached)" }
        parts.append(pre)
        return parts.joined(separator: " · ")
    }

    static func int(_ v: Any?) -> Int? {
        if let i = v as? Int { return i }
        if let d = v as? Double { return Int(d) }
        if let n = v as? NSNumber { return n.intValue }
        return nil
    }

    static func double(_ v: Any?) -> Double? {
        if let d = v as? Double { return d }
        if let i = v as? Int { return Double(i) }
        if let n = v as? NSNumber { return n.doubleValue }
        return nil
    }
}

public enum Role: String, Sendable, Codable {
    case system, user, assistant, tool
}

/// A tool call plus what happened when the app ran it.
public struct ToolRun: Identifiable, Equatable, Sendable, Codable {
    public enum Status: Equatable, Sendable, Codable {
        case pending, waitingForApproval, running, done, denied
    }

    public var call: ToolCall
    public var status: Status = .pending
    public var result: String = ""
    /// PNG images the tool returned (view_image, screenshot).
    public var images: [Data] = []
    public var id: String { call.id }

    public init(call: ToolCall) {
        self.call = call
    }
}

/// Messages of the /decide flow in a chat.
public enum DecideKind: Equatable, Sendable, Codable {
    /// The app's questions and the user's answers while setting up a decision: never sent to the model.
    case prompt
    /// A decision result card (the id keys AppState.decisionCards). Its content, a text summary, is
    /// part of the conversation from then on.
    case card(UUID)
}

/// One entry in a conversation. Images are PNG bytes (sent as data URLs).
public struct ChatMessage: Identifiable, Equatable, Sendable, Codable {
    public var id = UUID()
    public var role: Role
    public var content: String
    public var reasoning: String = ""
    public var images: [Data] = []
    public var toolRuns: [ToolRun] = []
    /// For `role == .tool`: which call this answers.
    public var toolCallID: String? = nil
    public var toolName: String? = nil
    public var stats: GenStats? = nil
    public var error: String? = nil
    public var decide: DecideKind? = nil
    /// Tool routing result for this reply ("Tools: none (P(needed) 0.08)"), shown above it.
    public var route: String? = nil
    /// On an app notice: the summary that replaces every message before it in what the model sees
    /// (automatic when the context is nearly full, or /compact).
    public var compaction: String? = nil
    /// Reply review result ("Check: ✓ language 0.97 · …"), shown under the reply.
    public var check: String? = nil
    /// A draft replaced after review: shown collapsed, never sent to the model again.
    public var superseded: Bool? = nil
    /// The user's rating of this reply: "up" or "down" (training-data export).
    public var feedback: String? = nil
    /// The reply as the model wrote it, when the user edited it (training-data export: rejected vs chosen).
    public var editedFrom: String? = nil

    public init(role: Role, content: String, images: [Data] = []) {
        self.role = role
        self.content = content
        self.images = images
    }

    /// The OpenAI chat message JSON for this entry.
    public func openAIMessage() -> [String: Any] {
        var m: [String: Any] = ["role": role.rawValue]
        switch role {
        case .user where !images.isEmpty:
            var parts: [[String: Any]] = images.map { png in
                ["type": "image_url", "image_url": ["url": "data:image/png;base64," + png.base64EncodedString()]]
            }
            if !content.isEmpty { parts.append(["type": "text", "text": content]) }
            m["content"] = parts
        case .assistant:
            m["content"] = content
            if !reasoning.isEmpty { m["reasoning_content"] = reasoning }
            if !toolRuns.isEmpty {
                m["tool_calls"] = toolRuns.map { run -> [String: Any] in
                    ["id": run.call.id, "type": "function",
                     "function": ["name": run.call.name, "arguments": run.call.arguments]]
                }
            }
        case .tool where !images.isEmpty:
            var parts: [[String: Any]] = [["type": "text", "text": content]]
            parts += images.map { png -> [String: Any] in
                ["type": "image_url", "image_url": ["url": "data:image/png;base64," + png.base64EncodedString()]]
            }
            m["content"] = parts
            if let toolCallID { m["tool_call_id"] = toolCallID }
            if let toolName { m["name"] = toolName }
        case .tool:
            m["content"] = content
            if let toolCallID { m["tool_call_id"] = toolCallID }
            if let toolName { m["name"] = toolName }
        default:
            m["content"] = content
        }
        return m
    }
}

/// Sampling and decode options sent with every request.
public struct GenerationOptions: Equatable, Sendable {
    public var selfSpec: Bool = true
    public var thinking: Bool = true
    public var temperature: Double = 1.0
    public var topK: Int = 20
    public var topP: Double = 0.95
    public var maxTokens: Int = 4096

    public init() {}
}
