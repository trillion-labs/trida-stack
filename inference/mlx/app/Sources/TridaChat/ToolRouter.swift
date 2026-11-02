import Foundation
import TridaChatCore

/// Tool routing: before the model answers a message in agent mode, the decision model picks what the reply
/// needs, as one choice question: answer directly, or one kind of tool. It uses the direct decision only (no
/// reasoning pass: one before every message took too long); below 0.4 it leaves every tool available.
/// Tools that are not needed cannot be called: with none, the request goes out with tool_choice "none" (the
/// server keeps the tool schemas in the prompt, so the prompt cache still matches, but bans <tool_call>);
/// with some, calls to the others are refused.
enum ToolRouting: String, CaseIterable, Identifiable {
    case off, reason
    var id: String { rawValue }
    var label: String {
        switch self {
        case .off: return "Off (all tools every time)"
        case .reason: return "Decision model (direct, no reasoning)"
        }
    }
}

struct ToolRoute {
    /// nil: every offered tool; empty: no tool.
    var allowed: Set<String>?
    var note: String
}

@MainActor
enum ToolRouter {
    struct Group {
        let key: String
        let description: String
        let tools: [String]
    }

    static let groups: [Group] = [
        Group(key: "files", description: "the user's files: list a folder, find, read or search files, look at an image file, write a file",
              tools: ["list_dir", "read_file", "find_files", "search_files", "view_image", "write_file", "pick_file"]),
        Group(key: "screen", description: "look at what is on the user's screen now (screenshot)", tools: ["screenshot"]),
        Group(key: "time", description: "the current date or time", tools: ["now"]),
        Group(key: "math", description: "an exact arithmetic calculation", tools: ["calculator"]),
        Group(key: "shell", description: "run a shell command on the Mac", tools: ["run_shell"]),
        Group(key: "decide", description: "a calibrated pick among options or a yes/no judgement the user asked for",
              tools: ["decide_choice", "decide_yes_no"]),
        Group(key: "app", description: "operate an app on the Mac: click, type, press keys, open an app",
              tools: ["ui_elements", "ui_click", "ui_type", "ui_key", "open_app", "ui_check"]),
    ]

    static let question = "What does replying to the user's last message need?"
    static let answerOption = "answer directly from the conversation (chat, writing, translation, explaining or "
        + "summarizing text already in the conversation, describing an attached image)"
    /// Below this top probability the router does not restrict the tools.
    static let minConfidence = 0.4

    static func names(_ specs: [[String: Any]]) -> Set<String> {
        Set(specs.compactMap { ($0["function"] as? [String: Any])?["name"] as? String })
    }
}

extension AppState {
    /// Decides which tools the latest message may use. Errors and unsure answers fall back to every tool.
    func routeTools(chatID: Chat.ID, offered: Set<String>, mode: ToolRouting,
                    progress: @escaping (String) -> Void) async -> ToolRoute {
        let groups = ToolRouter.groups.filter { !offered.isDisjoint(with: $0.tools) }
        guard mode != .off, supportsDecision, !groups.isEmpty else { return ToolRoute(allowed: nil, note: "") }
        var lines = Array(stateLines(chatID).suffix(6)).map { $0.count > 600 ? String($0.prefix(600)) + "…" : $0 }
        if lines.isEmpty { lines = ["(empty)"] }
        let q = DecisionQuestion.choice("need", ToolRouter.question,
                                        [("answer", ToolRouter.answerOption)] + groups.map { ($0.key, $0.description) })
        var s = decisionSettings
        s.reasoning = "never"         // direct decision only: reasoning before every message took too long
        s.threshold = 0.7
        let body = Decision.body(state: .object([("conversation", .strings(lines))]), questions: [q], settings: s)
        progress("Choosing tools…")
        let r: DecisionResult
        do {
            r = try await runDecision(body, [q], progress: { t in progress("Choosing tools: " + t) })
        } catch {
            return ToolRoute(allowed: nil, note: "Tools: all (router failed: \(error.localizedDescription))")
        }
        guard let a = r.answers["need"] else { return ToolRoute(allowed: nil, note: "Tools: all") }
        let f = { (x: Double) in String(format: "%.2f", x) }
        let ranked = a.options.sorted { $0.p > $1.p }
        guard let top = ranked.first else { return ToolRoute(allowed: nil, note: "Tools: all") }
        var how = ""
        if r.escalated {
            let d = r.direct["need"]?.options.max { $0.p < $1.p }
            let before = d.map { " (direct: \($0.key) \(f($0.p)))" } ?? ""
            how = r.cut == nil ? ", after reasoning" + before : ", reasoning cut off"
        }
        if top.p < ToolRouter.minConfidence {
            return ToolRoute(allowed: nil, note: "Tools: all (unsure: \(top.key) \(f(top.p))\(how))")
        }
        if top.key == "answer" {
            return ToolRoute(allowed: [], note: "Tools: none (answer \(f(top.p))\(how))")
        }
        // the top kind, plus any other tool kind at 0.2 or more ("answer" adds nothing)
        let picked = ranked.enumerated().filter { $0.offset == 0 || ($0.element.p >= 0.15 && $0.element.key != "answer") }
            .map(\.element)
        var allowed = Set<String>()
        for o in picked { allowed.formUnion(groups.first { $0.key == o.key }?.tools ?? []) }
        // Without write_file, the terminal is how the model writes files: "files" includes run_shell then.
        if picked.contains(where: { $0.key == "files" }), !offered.contains("write_file"), offered.contains("run_shell") {
            allowed.insert("run_shell")
        }
        allowed.formIntersection(offered)
        let label = picked.map { "\($0.key) \(f($0.p))" }.joined(separator: ", ")
        return ToolRoute(allowed: allowed, note: "Tools: \(label)\(how)")
    }
}
