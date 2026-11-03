import Foundation
import TridaChatCore

/// Reply review: after the model finishes a text reply, the decision model checks it against a few
/// yes/no criteria (language, does what was asked, complete, not a repeat of the previous reply), as one
/// direct decision (no reasoning pass: reasoning whenever a check was unsure took minutes on long replies).
/// Only a confident failure (P(yes) below `failBelow`) makes the app ask the model once to rewrite the reply
/// with that feedback; an unsure check is shown with "?" and changes nothing, since a rewrite costs far more
/// than the check. The failed draft stays visible (collapsed) but is no longer part of what the model sees.
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
        // Asked as two questions (see honestClaims / honestDone): one conditional question with an "answer yes
        // if it claims nothing" escape was read as "is everything in the reply backed by the tool results?" and
        // failed plain summaries of a file the model had read.
        ReviewCriterion(key: "honest", label: "claims match results",
                        question: "",
                        fix: "It claims an action that did not happen or failed (see the tool results). Say plainly "
                            + "what was and was not done."),
        ReviewCriterion(key: "fresh", label: "not a repeat",
                        question: "Is the response a new answer to the user's latest message, rather than a repeat of "
                            + "the earlier reply?",
                        fix: "It repeats the earlier reply instead of answering the latest message. Answer the latest message."),
    ]
    static let honestClaims = "Does the response say that it did something on the computer, such as creating, saving "
        + "or deleting a file, running a command, or clicking or typing in an app?"
    static let honestDone = "Do the tool calls above show that what the response says it did actually happened and "
        + "succeeded (not an error, not refused)?"
    /// Tools whose results a reply can misreport as done. "claims match results" runs only when one of these ran,
    /// or a call failed or was refused: reading or searching files cannot lead to a false "I saved it".
    static let actionTools: Set<String> = ["write_file", "run_shell", "screenshot", "ui_click", "ui_type", "ui_key",
                                           "open_app"]

    /// A criterion fails (and the reply is rewritten) only when P(yes) is below this; up to `passAt` it is unsure.
    static let failBelow = 0.2
    /// At or above this a criterion passes.
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
        var actions = false
        for m in chat.messages[(ui + 1)..<ri] {
            for run in m.toolRuns {
                if ReplyReview.actionTools.contains(run.call.name) || run.status == .denied
                    || run.result.hasPrefix("error") { actions = true }
                let res = run.status == .denied ? "not run: " + run.result : run.result
                calls.append("\(run.call.name)(\(ReplyReview.clip(run.call.arguments, head: 200, tail: 60))) -> "
                             + ReplyReview.clip(res, head: 300, tail: 100))
            }
        }
        state.append(("tool calls", calls.isEmpty ? .string("(none)") : .strings(calls)))
        state.append(("response", .string(ReplyReview.clip(text, head: 6000, tail: 2000))))
        // "not a repeat" needs an earlier reply; "claims match results" only applies after a tool that changes
        // something, or a failed / refused call (it misfired on plain answers and on summaries of read files).
        let crits = ReplyReview.criteria.filter {
            ($0.key != "fresh" || earlier != nil) && ($0.key != "honest" || actions)
        }
        let qs = crits.flatMap { c -> [DecisionQuestion] in
            c.key == "honest" ? [DecisionQuestion.yesNo("honest_claims", ReplyReview.honestClaims),
                                 DecisionQuestion.yesNo("honest_done", ReplyReview.honestDone)]
                : [DecisionQuestion.yesNo(c.key, c.question)]
        }
        update(chatID, replyID) { $0.check = "Checking the reply…" }
        let r: DecisionResult
        do {
            // the user's picture (e.g. a photographed problem) lets the checks see what was asked
            let image = supportsVision ? user.images.last : nil
            var s = decisionSettings
            s.reasoning = "never"
            r = try await runDecision(Decision.body(state: .object(state), questions: qs, image: image,
                                                    settings: s), qs,
                                          log: DecisionLog(chatID: chatID, messageID: replyID, kind: "Reply check")) {
                [weak self] t in self?.update(chatID, replyID) { $0.check = "Checking the reply: " + t }
            }
        } catch {
            update(chatID, replyID) { $0.check = "Reply check failed: \(error.localizedDescription)" }
            return nil
        }
        var failed: [ReviewCriterion] = []
        var parts: [String] = []
        for c in crits {
            let value: Double
            if c.key == "honest" {
                // fails only if it claims an action AND the tools don't show it: P(ok) = 1 - P(claims)(1 - P(done))
                guard let claims = r.answers["honest_claims"], let done = r.answers["honest_done"] else { continue }
                value = 1 - claims.value * (1 - done.value)
            } else {
                guard let a = r.answers[c.key] else { continue }
                value = a.value
            }
            let mark = value >= ReplyReview.passAt ? "✓" : value < ReplyReview.failBelow ? "✗" : "?"
            if mark == "✗" { failed.append(c) }
            parts.append("\(mark) \(c.label) \(String(format: "%.2f", value))")
        }
        update(chatID, replyID) { $0.check = "Check: " + parts.joined(separator: " · ") }
        return failed
    }
}
