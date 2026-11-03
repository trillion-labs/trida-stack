import AppKit
import SwiftUI
import TridaChatCore

struct RenameRequest: Identifiable {
    let id: Chat.ID
}

extension AppState {
    // ---- decision labels ---------------------------------------------------------------------------------

    /// Marks `key` as the right answer to one question of a decision kept on a reply (again: clears it).
    func labelDecision(chatID: Chat.ID, messageID: UUID, recordID: UUID, question: String, key: String) {
        update(chatID, messageID) { m in
            guard var ds = m.decisions, let i = ds.firstIndex(where: { $0.id == recordID }) else { return }
            var l = ds[i].labels ?? [:]
            l[question] = l[question] == key ? nil : key
            ds[i].labels = l.isEmpty ? nil : l
            m.decisions = ds
        }
    }

    /// Questions the user has answered, over all chats (Export Training Data writes them as kev records).
    var labeledDecisionCount: Int {
        chats.flatMap(\.messages).flatMap { $0.decisions ?? [] }.reduce(0) { $0 + ($1.labels?.count ?? 0) }
    }

    // ---- undo ----------------------------------------------------------------------------------------------

    /// Puts back the file a write_file call changed, and tells the model so in that call's result.
    func undoWrite(chatID: Chat.ID, messageID: UUID, runID: String) {
        guard let chat = chats.first(where: { $0.id == chatID }),
              let run = chat.messages.first(where: { $0.id == messageID })?.toolRuns.first(where: { $0.id == runID }),
              let u = run.undo, run.undone != true else { return }
        let note: String
        do {
            try FileUndo.restore(u)
            note = u.backup == nil ? "\n(Undone by the user: the file was removed, as it did not exist before.)"
                                   : "\n(Undone by the user: the file is back as it was before this write.)"
            setRun(chatID, messageID, runID) { $0.undone = true; $0.result += note }
        } catch {
            setRun(chatID, messageID, runID) { $0.result += "\n(Undo failed: \(error.localizedDescription))" }
            return
        }
        if let i = chats.firstIndex(where: { $0.id == chatID }),
           let t = chats[i].messages.firstIndex(where: { $0.role == .tool && $0.toolCallID == runID }) {
            chats[i].messages[t].content += note
        }
    }

    // ---- chats ---------------------------------------------------------------------------------------------

    /// The sidebar's chats: all of them, or those whose title or messages contain the search text.
    var visibleChats: [Chat] {
        let q = chatSearch.trimmingCharacters(in: .whitespaces)
        guard !q.isEmpty else { return chats }
        return chats.filter { c in
            c.title.localizedCaseInsensitiveContains(q) || c.messages.contains { $0.content.localizedCaseInsensitiveContains(q) }
        }
    }

    func beginRename(_ id: Chat.ID) {
        renameText = chats.first { $0.id == id }?.title ?? ""
        renaming = RenameRequest(id: id)
    }

    func finishRename(save: Bool) {
        let t = renameText.trimmingCharacters(in: .whitespacesAndNewlines)
        if save, let r = renaming, !t.isEmpty, let i = chats.firstIndex(where: { $0.id == r.id }) { chats[i].title = t }
        renaming = nil
    }
}

/// The typed decisions made for a reply, collapsed under it: each question's options with the model's
/// probabilities. Clicking an option marks it as the right answer (again: clears it); marked questions are
/// exported as decision-head training data.
struct DecisionsView: View {
    @EnvironmentObject var state: AppState
    let messageID: UUID
    let records: [DecisionRecord]

    private var marked: Int { records.reduce(0) { $0 + ($1.labels?.count ?? 0) } }

    var body: some View {
        DisclosureGroup {
            VStack(alignment: .leading, spacing: 12) {
                Text("Click the right answer when the model got it wrong (or to confirm it). Marked answers go into "
                     + "Export Training Data as decision-head training records.")
                    .font(.caption).foregroundStyle(Theme.muted)
                ForEach(records) { rec in record(rec) }
            }
            .padding(.top, 6)
        } label: {
            Label(marked > 0 ? "\(records.count) decisions, \(marked) marked" : "\(records.count) decisions",
                  systemImage: "slider.horizontal.3")
                .font(.caption)
                .foregroundStyle(Theme.muted)
        }
    }

    private func record(_ rec: DecisionRecord) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(rec.kind).font(.caption.weight(.semibold)).foregroundStyle(Theme.text)
            ForEach(rec.rows) { q in
                VStack(alignment: .leading, spacing: 3) {
                    Text(q.instructions).font(.caption).foregroundStyle(Theme.muted).lineLimit(2)
                    let top = q.options.map(\.p).max() ?? 0
                    ForEach(q.options) { o in
                        Button {
                            if let chat = state.selectedChatID {
                                state.labelDecision(chatID: chat, messageID: messageID, recordID: rec.id,
                                                    question: q.id, key: o.key)
                            }
                        } label: {
                            HStack(spacing: 8) {
                                Image(systemName: q.label == o.key ? "checkmark.circle.fill" : "circle")
                                    .foregroundStyle(q.label == o.key ? Theme.accent : Theme.muted)
                                Text(o.label)
                                    .lineLimit(1)
                                    .fontWeight(o.p == top ? .semibold : .regular)
                                    .foregroundStyle(Theme.text)
                                Spacer(minLength: 8)
                                ZStack(alignment: .leading) {
                                    RoundedRectangle(cornerRadius: 2).fill(Theme.accentSoft).frame(width: 80, height: 6)
                                    RoundedRectangle(cornerRadius: 2).fill(Theme.accent).frame(width: 80 * o.p, height: 6)
                                }
                                Text(String(format: "%.2f", o.p)).monospacedDigit().foregroundStyle(Theme.muted)
                            }
                            .font(.caption)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                        .help("Mark as the right answer")
                    }
                }
            }
        }
        .padding(10)
        .card(radius: 8)
    }
}

/// How full the context window is, next to the composer; the chat is summarized at 80%.
struct ContextMeter: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        if let id = state.selectedChatID, state.chats.contains(where: { $0.id == id && !$0.messages.isEmpty }) {
            let used = state.contextUsed(id)
            let total = max(state.settings.contextLength, 1)
            let f = min(Double(used) / Double(total), 1)
            let color = f >= AppState.compactAt ? Theme.danger : f >= 0.6 ? Theme.warning : Theme.accent
            HStack(spacing: 5) {
                ZStack {
                    Circle().stroke(Theme.accentSoft, lineWidth: 2.5)
                    Circle().trim(from: 0, to: f).stroke(color, style: StrokeStyle(lineWidth: 2.5, lineCap: .round))
                        .rotationEffect(.degrees(-90))
                }
                .frame(width: 11, height: 11)
                Text("\(Self.k(used)) of \(Self.k(total)) tokens")
                    .font(.caption2).monospacedDigit().foregroundStyle(Theme.muted)
            }
            .help("Context used by this chat. At \(Int(AppState.compactAt * 100))% the earlier conversation is "
                  + "summarized (or run /compact).")
        }
    }

    static func k(_ n: Int) -> String { n >= 1000 ? String(format: "%.0fK", Double(n) / 1000) : "\(n)" }
}

/// Renaming a chat (sidebar: right-click → Rename).
struct RenameSheet: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Rename chat").font(.headline)
            TextField("Name", text: $state.renameText)
                .textFieldStyle(.roundedBorder)
                .onSubmit { state.finishRename(save: true) }
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { state.finishRename(save: false) }.keyboardShortcut(.cancelAction)
                Button("Rename") { state.finishRename(save: true) }.keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(width: 380)
    }
}
