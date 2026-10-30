import AppKit
import SwiftUI
import TridaChatCore

/// Composer commands, picked from a menu that opens when the message starts with "/" (as in Claude Code).
struct SlashCommand: Identifiable, Equatable {
    let name: String
    let hint: String
    let summary: String
    var id: String { name }
}

extension AppState {
    static let slashCommands: [SlashCommand] = [
        SlashCommand(name: "/decide", hint: "[question]",
                     summary: "Decide about this conversation: question → options → answers with probabilities"),
        SlashCommand(name: "/new", hint: "", summary: "Start a new chat"),
        SlashCommand(name: "/compact", hint: "", summary: "Summarize the conversation so far to free up context"),
        SlashCommand(name: "/agent", hint: "", summary: "Turn agent mode (tools) on or off"),
        SlashCommand(name: "/mode", hint: "", summary: "Switch between self-spec and AR decoding"),
    ]

    /// Commands matching the draft while it is a bare "/word" (no arguments yet); empty otherwise.
    var slashMatches: [SlashCommand] {
        let d = draft.trimmingCharacters(in: .whitespaces).lowercased()
        guard d.hasPrefix("/"), !d.contains(" "), !d.contains("\n") else { return [] }
        if decideFlow?.chatID == selectedChatID { return [] }   // typing an answer in the /decide flow
        return Self.slashCommands.filter { $0.name.hasPrefix(d) }
    }

    /// Runs `/name args`. Returns false when `text` is not a known command (it is then sent as a message).
    @discardableResult
    func runSlash(_ text: String) -> Bool {
        let t = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let name = String(t.split(separator: " ", maxSplits: 1).first ?? "").lowercased()
        let args = t.count > name.count ? String(t.dropFirst(name.count)).trimmingCharacters(in: .whitespaces) : ""
        guard Self.slashCommands.contains(where: { $0.name == name }) else { return false }
        draft = ""
        slashSelection = 0
        switch name {
        case "/decide":
            guard !isGenerating, let ci = currentChatIndex else { return true }
            guard status.isUsable else {
                appendNotice("The server is not ready yet; try /decide again in a moment.")
                return true
            }
            let chatID = chats[ci].id
            if chats[ci].messages.isEmpty { chats[ci].title = "Decision" }
            startDecide(args, chatID: chatID, echo: t)
        case "/new":
            newChat()
        case "/compact":
            guard status.isUsable else {
                appendNotice("The server is not ready yet; try /compact again in a moment.")
                return true
            }
            compactNow()
        case "/agent":
            settings.agentMode.toggle()
            appendNotice("Agent mode is \(settings.agentMode ? "on" : "off").")
        case "/mode":
            settings.selfSpec.toggle()
            appendNotice("Decoding: \(settings.selfSpec ? "self-spec" : "AR").")
        default:
            return false
        }
        return true
    }

    /// A one-line app message in the current chat (never sent to the model).
    func appendNotice(_ text: String) {
        guard let ci = currentChatIndex else { return }
        var m = ChatMessage(role: .assistant, content: text)
        m.decide = .prompt
        chats[ci].messages.append(m)
    }
}

/// The menu above the composer while the draft is "/…".
struct SlashMenu: View {
    @EnvironmentObject var state: AppState
    let matches: [SlashCommand]

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ForEach(Array(matches.enumerated()), id: \.element.id) { i, c in
                let selected = i == min(state.slashSelection, matches.count - 1)
                Button {
                    state.runSlash(c.name)
                } label: {
                    HStack(spacing: 10) {
                        Text(c.name).font(.system(.body, design: .monospaced).weight(.semibold))
                        if !c.hint.isEmpty {
                            Text(c.hint).font(.system(.callout, design: .monospaced)).foregroundStyle(.secondary)
                        }
                        Text(c.summary).foregroundStyle(.secondary).lineLimit(1)
                        Spacer()
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .contentShape(Rectangle())
                    .background(RoundedRectangle(cornerRadius: 6).fill(selected ? Color.accentColor.opacity(0.18) : Color.clear))
                }
                .buttonStyle(.plain)
            }
            Text("↑↓ to choose · Return to run · Tab to complete · Esc to close")
                .font(.caption2).foregroundStyle(.secondary)
                .padding(.horizontal, 10).padding(.top, 4)
        }
        .padding(6)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .controlBackgroundColor)))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.secondary.opacity(0.25)))
    }
}
