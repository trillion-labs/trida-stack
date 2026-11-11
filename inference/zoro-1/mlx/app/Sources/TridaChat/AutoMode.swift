import Foundation
import TridaChatCore

/// The write the code viewer shows: a write_file tool run on a reply of the current chat.
struct ViewerTarget: Equatable {
    var messageID: UUID
    var runID: String
}

extension AppState {
    /// Auto mode's call on an action: `ask` with the reason, or run (the note says why it was let through).
    /// Risky patterns always ask; otherwise the decision model judges it, and anything unsure, or a failed
    /// check, asks.
    func autoVerdict(_ call: ToolCall, tools: AgentTools, chatID: Chat.ID, messageID: UUID) async -> (ask: Bool, note: String) {
        if let r = AutoApproval.risk(call, tools: tools) { return (true, r) }
        guard status.isUsable else { return (true, "the decision model is not available") }
        let a = call.argumentDict
        let user = chats.first { $0.id == chatID }?.messages.last { $0.role == .user && $0.decide != .prompt }?.content ?? ""
        var state: [(String, OrderedJSON)] = [
            ("user request", .string(ReplyReview.clip(user, head: 1500, tail: 500))),
            ("workspace", .string(tools.root.path)),
            ("tool", .string(call.name)),
        ]
        switch call.name {
        case "run_shell":
            state.append(("command", .string(ReplyReview.clip(a["command"] as? String ?? "", head: 1500, tail: 300))))
        case "write_file":
            let p = a["path"] as? String ?? ""
            let url = try? tools.path(p)
            state.append(("file", .string(url?.path ?? p)))
            state.append(("file exists", .bool(url.map { FileManager.default.fileExists(atPath: $0.path) } ?? false)))
            state.append(("new content", .string(ReplyReview.clip(a["content"] as? String ?? "", head: 1200, tail: 300))))
        default:
            state.append(("arguments", .string(ReplyReview.clip(call.arguments, head: 1200, tail: 200))))
        }
        let q = [DecisionQuestion.yesNo("safe", AutoApproval.question)]
        var s = decisionSettings
        s.reasoning = "never"
        do {
            let r = try await runDecision(Decision.body(state: .object(state), questions: q, settings: s), q,
                                          log: DecisionLog(chatID: chatID, messageID: messageID, kind: "Auto approval")) { _ in }
            guard let p = r.answers["safe"]?.value else { return (true, "the safety check gave no answer") }
            let note = "Zoro-1 judged it safe \(String(format: "%.2f", p))"
            return p >= AutoApproval.threshold ? (false, note) : (true, "Zoro-1 is not sure it is safe (\(String(format: "%.2f", p)))")
        } catch {
            return (true, "the safety check failed (\(error.localizedDescription))")
        }
    }

    /// A write finished: the viewer shows it (and opens, unless turned off).
    func showWrite(messageID: UUID, runID: String) {
        viewerTarget = ViewerTarget(messageID: messageID, runID: runID)
        viewerShowsDiff = true
        if openViewerOnWrite { showViewer = true }
    }

    /// The current chat's file writes, oldest first.
    var viewerWrites: [(messageID: UUID, run: ToolRun)] {
        guard let ci = currentChatIndex else { return [] }
        return chats[ci].messages.flatMap { m in
            m.toolRuns.filter { $0.call.name == "write_file" && $0.status == .done && !$0.result.hasPrefix("error") }
                .map { (messageID: m.id, run: $0) }
        }
    }
}
