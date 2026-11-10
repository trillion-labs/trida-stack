import Foundation
import TridaChatCore

/// The /decide command: the conversation so far becomes the decision state, then the app asks for each
/// question and its options in the chat, and finally shows the decision model's answers as a card.
struct DecideFlow {
    enum Step: Equatable {
        case situation    // the chat is empty: ask what the state is
        case question     // "what should be decided?"
        case options      // options for `pendingQuestion`
        case more         // add another question, or decide
    }

    var chatID: Chat.ID
    var step: Step
    var situation: String = ""
    var pendingQuestion = ""
    var questions: [DecisionQuestion] = []
}

/// A decision shown in the chat (keyed by the card message's DecideKind.card id).
struct DecisionCard {
    var questions: [DecisionQuestion]
    var running = true
    var phase = "Deciding…"
    var liveReasoning = ""
    var result: DecisionResult?
    var error: String?
}

extension AppState {
    static let decideHelp = "Answer in the box below. Type **cancel** to stop."

    /// Composer hint while the flow waits for input.
    var decidePlaceholder: String? {
        guard let f = decideFlow, f.chatID == selectedChatID else { return nil }
        switch f.step {
        case .situation: return "Describe the situation to decide about…"
        case .question: return "Question \(f.questions.count + 1), e.g. Which team should handle this?"
        case .options: return "Options, comma-separated (key: description), or yes/no, or score: low, medium, high"
        case .more: return "Next question, or \"decide\" to get the answers"
        }
    }

    /// `/decide [question]` typed in the composer.
    func startDecide(_ rest: String, chatID: Chat.ID, echo: String = "/decide") {
        append(chatID, prompt(echo, role: .user))   // the command itself, shown like a message
        guard supportsDecision else {
            append(chatID, prompt("Decision mode needs a model with the decision head (jev_head.pt). "
                                  + "Settings → Agent → Decision head → Download.", role: .assistant))
            return
        }
        let n = stateLines(chatID).count
        var f = DecideFlow(chatID: chatID, step: n == 0 ? .situation : .question)
        let image = stateImage(chatID) != nil ? " and its latest image" : ""
        var intro = n == 0
            ? "**Decision mode.** There is no conversation yet to decide about. What is the situation?"
            : "**Decision mode.** The conversation so far (\(n) message\(n == 1 ? "" : "s")\(image)) is the state."
        let q = rest.trimmingCharacters(in: .whitespacesAndNewlines)
        if n > 0 {
            if q.isEmpty {
                intro += "\n\nQuestion 1: what should be decided?"
            } else {
                f.pendingQuestion = q
                f.step = .options
                intro += "\n\nQuestion 1: \(q)\n\n" + Self.optionsAsk
            }
        }
        decideFlow = f
        append(chatID, prompt(intro + "\n\n" + Self.decideHelp, role: .assistant))
    }

    static let optionsAsk = "What are the options? Separate them with commas or new lines (`key: description` "
        + "is fine), or type `yes/no`, or `score: low, medium, high` for an ordered scale."

    /// One reply typed while the flow is active.
    func handleDecideInput(_ text: String, chatID: Chat.ID) {
        guard var f = decideFlow, f.chatID == chatID else { return }
        append(chatID, prompt(text, role: .user))
        let cmd = text.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)
        if ["cancel", "/cancel", "stop", "quit"].contains(cmd) {
            decideFlow = nil
            append(chatID, prompt("Decision cancelled.", role: .assistant))
            return
        }
        switch f.step {
        case .situation:
            f.situation = text
            f.step = .question
            decideFlow = f
            append(chatID, prompt("Question 1: what should be decided?", role: .assistant))

        case .question:
            if ["decide", "done", "/decide"].contains(cmd) {
                append(chatID, prompt("Add at least one question first. What should be decided?", role: .assistant))
                return
            }
            f.pendingQuestion = text
            f.step = .options
            decideFlow = f
            append(chatID, prompt(Self.optionsAsk, role: .assistant))

        case .options:
            let key = "q\(f.questions.count + 1)"
            let q: DecisionQuestion
            switch Self.parseOptionsReply(text) {
            case .yesNo:
                q = .yesNo(key, f.pendingQuestion)
            case let .score(levels):
                guard levels.count >= 2 else {
                    append(chatID, prompt("A score needs at least two levels, e.g. `score: low, medium, high`.", role: .assistant))
                    return
                }
                q = .score(key, f.pendingQuestion, levels: levels)
            case let .choice(opts):
                guard opts.count >= 2 else {
                    append(chatID, prompt("Give at least two options, separated by commas.", role: .assistant))
                    return
                }
                guard Set(opts.map(\.0)).count == opts.count else {
                    append(chatID, prompt("Two options have the same name; make them distinct.", role: .assistant))
                    return
                }
                q = .choice(key, f.pendingQuestion, opts)
            }
            f.questions.append(q)
            f.pendingQuestion = ""
            f.step = .more
            decideFlow = f
            append(chatID, prompt("Added question \(f.questions.count): \(Self.describe(q))\n\n"
                                  + "Add another question? Type it, or type **decide** to get the answers. If any answer is "
                                  + "unsure (top probability below 0.7), the model reasons first and decides again.", role: .assistant))

        case .more:
            if ["decide", "done", "go", "run", "/decide", "no"].contains(cmd) {
                runDecideFlow()
                return
            }
            f.pendingQuestion = text
            f.step = .options
            decideFlow = f
            append(chatID, prompt("Question \(f.questions.count + 1): \(text)\n\n" + Self.optionsAsk, role: .assistant))
        }
    }

    /// The buttons on the flow's last message.
    func decideAddQuestion() {
        guard var f = decideFlow else { return }
        let chatID = f.chatID
        f.step = .question
        decideFlow = f
        append(chatID, prompt("Question \(f.questions.count + 1): what should be decided?", role: .assistant))
    }

    func decideYesNo() {
        guard let f = decideFlow, f.step == .options else { return }
        handleDecideInput("yes/no", chatID: f.chatID)
    }

    func decideCancel() {
        guard let f = decideFlow else { return }
        handleDecideInput("cancel", chatID: f.chatID)
    }

    /// Sends the questions with the conversation as the state; the answers stream into a card message.
    func runDecideFlow() {
        guard let f = decideFlow, !f.questions.isEmpty else { return }
        decideFlow = nil
        let chatID = f.chatID
        let lines = stateLines(chatID)
        var state: [(String, OrderedJSON)] = []
        if !lines.isEmpty { state.append(("conversation", .strings(lines))) }
        if !f.situation.isEmpty { state.append(("situation", .string(f.situation))) }
        let image = supportsVision ? stateImage(chatID) : nil
        let qs = f.questions
        let body = Decision.body(state: .object(state), questions: qs, image: image, settings: decisionSettings, stream: true)
        let cardID = UUID()
        decisionCards[cardID] = DecisionCard(questions: qs)
        var m = ChatMessage(role: .assistant, content: "")
        m.decide = .card(cardID)
        append(chatID, m)
        let msgID = m.id
        let client = self.client
        isGenerating = true
        decideTask = Task {
            defer { isGenerating = false }
            do {
                for try await ev in client.decideStream(body, questions: qs) {
                    switch ev {
                    case let .direct(reason):
                        decisionCards[cardID]?.phase = reason.map { "Unsure (\($0)). Reasoning…" } ?? "Decided directly."
                    case let .reasoning(t):
                        decisionCards[cardID]?.liveReasoning += t
                    case let .result(r):
                        decisionCards[cardID]?.result = r
                        decisionCards[cardID]?.phase = r.escalated
                            ? (r.cut == nil ? "Decided after reasoning." : "Reasoning cut off; the direct decision is shown.")
                            : "Decided directly."
                        update(chatID, msgID) { $0.content = Self.summary(qs, r) }
                    }
                }
            } catch {
                let stopped = Task.isCancelled || (error as? URLError)?.code == .cancelled
                decisionCards[cardID]?.error = stopped ? "Stopped." : error.localizedDescription
                decisionCards[cardID]?.phase = stopped ? "Stopped." : "Failed."
            }
            decisionCards[cardID]?.running = false
        }
    }

    // MARK: helpers

    private func prompt(_ text: String, role: Role) -> ChatMessage {
        var m = ChatMessage(role: role, content: text)
        m.decide = .prompt
        return m
    }

    /// The conversation as decision state: one line per message, oldest first, without the /decide set-up
    /// messages and the reasoning; long messages are clipped and only the most recent ~16k characters kept
    /// (the decision state holds at most 7552 tokens).
    func stateLines(_ chatID: Chat.ID) -> [String] {
        guard let chat = chats.first(where: { $0.id == chatID }) else { return [] }
        var lines: [String] = []
        for m in chat.messages where m.decide != .prompt {
            let text = m.content.trimmingCharacters(in: .whitespacesAndNewlines)
            var line: String
            switch m.role {
            case .user: line = "user: " + text + (m.images.isEmpty ? "" : " [image]")
            case .assistant:
                let calls = m.toolRuns.map { "\($0.call.name)(\($0.call.arguments))" }.joined(separator: ", ")
                if text.isEmpty && calls.isEmpty { continue }
                line = "assistant: " + text + (calls.isEmpty ? "" : (text.isEmpty ? "" : " ") + "[called \(calls)]")
            case .tool: line = "tool result (\(m.toolName ?? "tool")): " + text
            case .system: continue
            }
            if line.count > 2000 { line = String(line.prefix(2000)) + "…" }
            lines.append(line)
        }
        var total = 0
        var kept: [String] = []
        for l in lines.reversed() {
            total += l.count
            if total > 16_000 && !kept.isEmpty { break }
            kept.append(l)
        }
        return kept.reversed()
    }

    /// The most recent image the user sent in this chat (the decision state carries one image).
    func stateImage(_ chatID: Chat.ID) -> Data? {
        guard let chat = chats.first(where: { $0.id == chatID }) else { return nil }
        let m = chat.messages.last(where: { $0.role == .user && !$0.images.isEmpty && $0.decide != .prompt })
        return m?.images.last
    }

    enum OptionsReply {
        case yesNo
        case score([String])
        case choice([(String, String)])
    }

    static func parseOptionsReply(_ text: String) -> OptionsReply {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let low = t.lowercased()
        if ["yes/no", "yes / no", "y/n", "yes no", "yesno", "yes or no", "noul"].contains(low) { return .yesNo }
        func split(_ s: String) -> [String] {
            let parts = s.contains("\n") ? s.split(separator: "\n") : s.split(separator: ",")
            return parts.map { $0.trimmingCharacters(in: .whitespaces) }.filter { !$0.isEmpty }
        }
        if low.hasPrefix("score:") { return .score(split(String(t.dropFirst(6)))) }
        let items = split(t)
        return .choice(DecisionQuestion.parseOptions(items.joined(separator: "\n"), type: .choice))
    }

    static func describe(_ q: DecisionQuestion) -> String {
        switch q.type {
        case .noul: return "\(q.instructions) — yes / no"
        case .score: return "\(q.instructions) — score: " + q.options.map(\.0).joined(separator: " < ")
        case .choice: return "\(q.instructions) — " + q.options.map(\.0).joined(separator: ", ")
        }
    }

    /// Text form of a decision, kept in the conversation for the model.
    static func summary(_ qs: [DecisionQuestion], _ r: DecisionResult) -> String {
        var s = "Decision (" + r.provenance + "):"
        for q in qs {
            guard let a = r.answers[q.key] else { continue }
            s += "\n- \(q.instructions) → \(a.summary)"
        }
        return s
    }
}
