import Foundation
import TridaChatCore

/// Reply review: after the model finishes a text reply, the decision model checks it against a few
/// yes/no criteria (language, does what was asked, complete, not a repeat of the previous reply), reasoning
/// first when unsure. If one fails, the app asks the model once to rewrite the reply with that feedback;
/// the failed draft stays visible (collapsed) but is no longer part of what the model sees.
struct ReviewCriterion {
    let key: String
    let label: String
    let question: String
    let fix: String
}

@MainActor
enum ReplyReview {
    static let criteria: [ReviewCriterion] = [
        ReviewCriterion(key: "language", label: "language",
                        question: "Is the response written in the language the user asked for? If the user did not "
                            + "ask for a language, is it in the language of the user's message?",
                        fix: "It is not in the language the user asked for (or wrote in). Write it in that language."),
        ReviewCriterion(key: "request", label: "does what was asked",
                        question: "Does the response do what the user's message asks for (for example summarize, "
                            + "translate, answer the question, or write what was requested)?",
                        fix: "It does not do what the user asked. Do exactly that."),
        ReviewCriterion(key: "complete", label: "complete",
                        question: "Is the response complete: not cut off, and not leaving out parts of what was asked?",
                        fix: "It is incomplete or cut off. Give the complete answer."),
        ReviewCriterion(key: "honest", label: "claims match results",
                        question: "If the response says it did something on the computer (created or saved a file, "
                            + "ran a command, searched, clicked), do the tool calls above show that it actually "
                            + "succeeded? Answer yes if the response claims no such action.",
                        fix: "It claims an action that did not happen or failed (see the tool results). Say plainly "
                            + "what was and was not done."),
        ReviewCriterion(key: "fresh", label: "not a repeat",
                        question: "Is the response a new answer to the user's latest message, rather than a repeat of "
                            + "the earlier reply?",
                        fix: "It repeats the earlier reply instead of answering the latest message. Answer the latest message."),
    ]
    /// A criterion passes when P(yes) is at least this (after reasoning, if the decision reasoned).
    static let passAt = 0.5

    static func clip(_ s: String, head: Int, tail: Int) -> String {
        s.count <= head + tail ? s : String(s.prefix(head)) + "\n…\n" + String(s.suffix(tail))
    }
}

extension AppState {
    /// Reviews the reply `replyID`; when a criterion fails, rewrites it once. `specs`: the turn's tool schemas
    /// (kept in the rewrite request so the prompt cache still matches; calls are banned with tool_choice none).
    func reviewAndRevise(_ chatID: Chat.ID, replyID: UUID, specs: [[String: Any]]?) async {
        guard reviewReplies, supportsDecision, let failed = await review(chatID, replyID: replyID) else { return }
        guard !failed.isEmpty, !Task.isCancelled,
              let draft = chats.first(where: { $0.id == chatID })?.messages.first(where: { $0.id == replyID }) else { return }
        // The draft leaves the model's view; the rewrite request shows it once, with the feedback.
        update(chatID, replyID) { $0.superseded = true }
        let draftText = draft.content
        var messages = history(chatID, tools: specs == nil ? nil : currentAgentTools(chatID))
        messages.append(["role": "assistant", "content": draftText])
        let notes = failed.map { "- " + $0.fix }.joined(separator: "\n")
        messages.append(["role": "user", "content": "[Automatic review] Your reply above fails these checks:\n\(notes)\n"
                         + "Write the reply again, fixing this. Reply with the corrected answer only."])
        var body = ChatClient.requestBody(messages: messages, tools: specs, options: settings.generation, model: servedName)
        if specs != nil { body["tool_choice"] = "none" }
        var reply = ChatMessage(role: .assistant, content: "")
        reply.route = "Rewritten after review: " + failed.map(\.label).joined(separator: ", ")
        append(chatID, reply)
        do {
            for try await event in client.stream(body: body) {
                switch event {
                case let .content(t): prefill = nil; update(chatID, reply.id) { $0.content += t }
                case let .reasoning(t): prefill = nil; update(chatID, reply.id) { $0.reasoning += t }
                case let .usage(st): update(chatID, reply.id) { $0.stats = st }
                case let .prefill(done, total): prefill = (done, total)
                default: break
                }
            }
        } catch {
            let cancelled = Task.isCancelled || (error as? URLError)?.code == .cancelled
            update(chatID, reply.id) { $0.error = cancelled ? "Stopped." : error.localizedDescription }
            return
        }
        if !Task.isCancelled { _ = await review(chatID, replyID: reply.id) }   // show its checks; no second rewrite
    }

    /// The agent tools of this turn (only for the system prompt of a rewrite request).
    func currentAgentTools(_ chatID: Chat.ID) -> AgentTools? {
        let s = settings
        return s.agentMode ? AgentTools(root: s.workspace, allowShell: s.allowShell, allowWrite: s.allowWrite,
                                        vision: supportsVision, allowScreenshot: allowScreenshot,
                                        shellSession: chatID.uuidString) : nil
    }

    /// Runs the criteria on a reply and writes the result line on it. Returns the failed criteria, or nil when
    /// nothing was checked (empty reply, no user message, decision error).
    func review(_ chatID: Chat.ID, replyID: UUID) async -> [ReviewCriterion]? {
        guard let chat = chats.first(where: { $0.id == chatID }),
              let ri = chat.messages.firstIndex(where: { $0.id == replyID }) else { return nil }
        let reply = chat.messages[ri]
        let text = reply.content.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, reply.error == nil,
              let ui = chat.messages[..<ri].lastIndex(where: { $0.role == .user && $0.decide != .prompt }) else { return nil }
        let user = chat.messages[ui]
        let earlier = chat.messages[..<ui].last {
            $0.role == .assistant && $0.decide == nil && $0.superseded != true && !$0.content.isEmpty
        }
        var state: [(String, OrderedJSON)] = [
            ("user message", .string(ReplyReview.clip(user.content, head: 2500, tail: 1000)
                                     + (user.images.isEmpty ? "" : " [an image is attached]"))),
        ]
        if let earlier { state.append(("earlier reply", .string(ReplyReview.clip(earlier.content, head: 600, tail: 200)))) }
        // What the tools did between the user's message and this reply (for the "claims match results" check).
        var calls: [String] = []
        for m in chat.messages[(ui + 1)..<ri] {
            for run in m.toolRuns {
                let res = run.status == .denied ? "not run: " + run.result : run.result
                calls.append("\(run.call.name)(\(ReplyReview.clip(run.call.arguments, head: 200, tail: 60))) -> "
                             + ReplyReview.clip(res, head: 300, tail: 100))
            }
        }
        state.append(("tool calls", calls.isEmpty ? .string("(none)") : .strings(calls)))
        state.append(("response", .string(ReplyReview.clip(text, head: 6000, tail: 2000))))
        // "not a repeat" needs an earlier reply; "claims match results" only applies when tools ran this turn
        // (without tool calls it misfired on plain answers, e.g. a math solution).
        let crits = ReplyReview.criteria.filter {
            ($0.key != "fresh" || earlier != nil) && ($0.key != "honest" || !calls.isEmpty)
        }
        let qs = crits.map { DecisionQuestion.yesNo($0.key, $0.question) }
        update(chatID, replyID) { $0.check = "Checking the reply…" }
        let r: DecisionResult
        do {
            // the user's picture (e.g. a photographed problem) lets the checks see what was asked
            let image = supportsVision ? user.images.last : nil
            r = try await runDecision(Decision.body(state: .object(state), questions: qs, image: image,
                                                    settings: decisionSettings), qs) {
                [weak self] t in self?.update(chatID, replyID) { $0.check = "Checking the reply: " + t }
            }
        } catch {
            update(chatID, replyID) { $0.check = "Reply check failed: \(error.localizedDescription)" }
            return nil
        }
        var failed: [ReviewCriterion] = []
        var parts: [String] = []
        for c in crits {
            guard let a = r.answers[c.key] else { continue }
            let ok = a.value >= ReplyReview.passAt
            if !ok { failed.append(c) }
            parts.append("\(ok ? "✓" : "✗") \(c.label) \(String(format: "%.2f", a.value))")
        }
        let how = r.escalated ? (r.cut == nil ? " (after reasoning)" : " (reasoning cut off)") : ""
        update(chatID, replyID) { $0.check = "Check: " + parts.joined(separator: " · ") + how }
        return failed
    }
}
