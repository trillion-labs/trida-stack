import Foundation
import TridaChatCore

/// Agent tools answered by a typed decision (POST /v1/decide) instead of generated text, like ui_click:
/// the options come from the call (or from the workspace / the screen), Zoro-1's pointer head scores them,
/// and when the top probability is below 0.7 the server reasons once (self-spec) and decides again.
/// Offered in agent mode when the served checkpoint has the decision head. None of them change anything.
@MainActor
enum DecisionTools {
    static let names: Set<String> = ["decide_choice", "decide_yes_no", "pick_file", "ui_check"]

    private static func fn(_ name: String, _ desc: String, _ props: [String: Any], _ required: [String]) -> [String: Any] {
        ["type": "function",
         "function": ["name": name, "description": desc,
                      "parameters": ["type": "object", "properties": props, "required": required]]]
    }

    static func specs(ui: Bool) -> [[String: Any]] {
        let str: [String: Any] = ["type": "string"]
        var s: [[String: Any]] = [
            fn("decide_choice", "Pick one of several options with the decision model, which returns a calibrated "
               + "probability for each (and reasons first when unsure). Use it for classify / which-one questions "
               + "whose options you can list, instead of guessing.",
               ["question": str,
                "options": ["type": "array", "items": str, "description": "2 to 20 short options."],
                "context": ["type": "string", "description": "The facts to decide on (text you read, the situation)."]],
               ["question", "options"]),
            fn("decide_yes_no", "Answer a yes/no question with the decision model: returns P(yes), reasoning first "
               + "when unsure. Use it for checks such as \"does this text say the refund was approved?\".",
               ["question": str, "context": str], ["question"]),
            fn("pick_file", "Pick the file in a workspace folder that best fits a goal, using the decision model over "
               + "the folder's file names. Returns the top candidates with probabilities.",
               ["goal": str, "path": ["type": "string", "description": "Folder relative to the workspace (default \".\")."]],
               ["goal"]),
        ]
        if ui {
            s.append(fn("ui_check", "Ask the decision model whether a goal is already done in the operated app, "
                        + "judging from its current elements. Returns P(done).", ["goal": str], ["goal"]))
        }
        return s
    }

    static let systemPrompt = "When the user asks you to classify something, choose between options or make a yes/no "
        + "judgement, you can call decide_choice or decide_yes_no (a calibrated decision model) and report its pick "
        + "and probability."
}

extension AppState {
    var decisionSettings: DecisionSettings {
        DecisionSettings(reasoning: "auto", threshold: 0.7, maxReasoningTokens: 8192, selfSpec: settings.selfSpec)
    }

    /// Streams one decision, showing progress (and the reasoning as it is written) through `progress`.
    func runDecision(_ body: OrderedJSON, _ questions: [DecisionQuestion],
                     progress: @escaping (String) -> Void) async throws -> DecisionResult {
        var reasoning = ""
        var why = ""
        for try await ev in client.decideStream(body, questions: questions) {
            switch ev {
            case let .direct(reason):
                if let reason { why = reason; progress("Unsure (\(reason)); reasoning…") }
            case let .reasoning(t):
                reasoning += t
                progress("Reasoning (\(why)):\n…" + String(reasoning.suffix(400)))
            case let .result(r):
                return r
            }
        }
        throw ChatClientError.http(200, "the decision stream ended without a result")
    }

    private func lastUserText(_ chatID: Chat.ID?) -> String {
        guard let chat = chats.first(where: { $0.id == chatID }) else { return "" }
        return chat.messages.last(where: { $0.role == .user })?.content ?? ""
    }

    private static func clip(_ s: String, _ n: Int) -> String { s.count > n ? String(s.prefix(n)) + "…" : s }

    private static func note(_ r: DecisionResult) -> String {
        var s = " — " + r.provenance
        if let t = r.reasoning?.trimmingCharacters(in: .whitespacesAndNewlines), !t.isEmpty, r.cut == nil {
            s += "\nReasoning: " + clip(t, 600)
        }
        return s
    }

    private static func ranked(_ a: DecisionAnswer, top n: Int) -> String {
        a.options.sorted { $0.p > $1.p }.prefix(n)
            .map { "\($0.label) " + String(format: "%.2f", $0.p) }.joined(separator: "; ")
    }

    /// Runs one decision tool (see DecisionTools).
    func runDecisionTool(_ call: ToolCall, chatID: Chat.ID, tools: AgentTools?,
                         progress: @escaping (String) -> Void) async -> ToolOutput {
        let a = call.argumentDict
        let request = Self.clip(lastUserText(chatID), 2000)
        do {
            switch call.name {
            case "decide_choice":
                guard let question = a["question"] as? String, !question.isEmpty else {
                    return ToolOutput(text: "error: missing argument 'question'")
                }
                var opts: [String] = []
                for o in (a["options"] as? [Any]) ?? [] {
                    let t = "\(o)".trimmingCharacters(in: .whitespacesAndNewlines)
                    if !t.isEmpty, !opts.contains(t) { opts.append(t) }
                }
                guard (2...20).contains(opts.count) else { return ToolOutput(text: "error: give 2 to 20 distinct options") }
                let q = DecisionQuestion.choice("answer", question, opts.map { ($0, "") })
                let state = OrderedJSON.object([("user request", .string(request)),
                                                ("context", .string((a["context"] as? String) ?? ""))])
                progress("Deciding among \(opts.count) options…")
                let r = try await runDecision(Decision.body(state: state, questions: [q], settings: decisionSettings), [q],
                                              progress: progress)
                guard let ans = r.answers["answer"] else { return ToolOutput(text: "error: no answer") }
                return ToolOutput(text: "Decision: \(ans.summary)\(Self.note(r))\nAll options: \(Self.ranked(ans, top: 20))")

            case "decide_yes_no":
                guard let question = a["question"] as? String, !question.isEmpty else {
                    return ToolOutput(text: "error: missing argument 'question'")
                }
                let q = DecisionQuestion.yesNo("answer", question)
                let state = OrderedJSON.object([("user request", .string(request)),
                                                ("context", .string((a["context"] as? String) ?? ""))])
                progress("Deciding yes / no…")
                let r = try await runDecision(Decision.body(state: state, questions: [q], settings: decisionSettings), [q],
                                              progress: progress)
                guard let ans = r.answers["answer"] else { return ToolOutput(text: "error: no answer") }
                return ToolOutput(text: "Decision: \(ans.pick) (P(yes) = \(String(format: "%.2f", ans.value)))\(Self.note(r))")

            case "pick_file":
                guard let goal = a["goal"] as? String, !goal.isEmpty else { return ToolOutput(text: "error: missing argument 'goal'") }
                guard let tools else { return ToolOutput(text: "error: agent mode is off") }
                let path = (a["path"] as? String) ?? "."
                let names = try tools.directoryEntries(path)
                guard !names.isEmpty else { return ToolOutput(text: "error: \(path) is empty") }
                if names.count == 1 { return ToolOutput(text: "Only one entry in \(path): \(names[0])") }
                let q = DecisionQuestion.choice("file", "Which of these files or folders best fits the goal?",
                                                names.map { ($0, "") })
                let state = OrderedJSON.object([("folder", .string(path)), ("goal", .string(goal)),
                                                ("user request", .string(request))])
                progress("Deciding among \(names.count) entries of \(path)…")
                let r = try await runDecision(Decision.body(state: state, questions: [q], settings: decisionSettings), [q],
                                              progress: progress)
                guard let ans = r.answers["file"] else { return ToolOutput(text: "error: no answer") }
                return ToolOutput(text: "Best match in \(path): \(ans.pick) (\(String(format: "%.2f", ans.top)))\(Self.note(r))\n"
                                  + "Top candidates: \(Self.ranked(ans, top: 5))")

            case "ui_check":
                guard let goal = a["goal"] as? String, !goal.isEmpty else { return ToolOutput(text: "error: missing argument 'goal'") }
                guard UIAgent.trusted else { return ToolOutput(text: "error: no Accessibility permission (Settings → Agent)") }
                guard let screen = UIAgent.shared.screen() else {
                    return ToolOutput(text: "error: no app to check. Ask the user to click into the app first, or call open_app.")
                }
                uiSession.lastScreen = screen
                let q = DecisionQuestion.yesNo("done", "Is the goal already achieved, as this screen shows?")
                let state = UITools.state(screen, goal: goal, history: uiSession.history, elements: screen.elements)
                progress("Checking \(screen.app)…")
                let r = try await runDecision(Decision.body(state: state, questions: [q], settings: decisionSettings), [q],
                                              progress: progress)
                guard let ans = r.answers["done"] else { return ToolOutput(text: "error: no answer") }
                return ToolOutput(text: "Goal done: \(ans.pick) (P(done) = \(String(format: "%.2f", ans.value)))\(Self.note(r))")

            default:
                return ToolOutput(text: "error: unknown tool \(call.name)")
            }
        } catch {
            return ToolOutput(text: "error: the decision failed: \(error.localizedDescription)")
        }
    }
}
