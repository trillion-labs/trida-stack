import AppKit
import SwiftUI
import TridaChatCore

/// The reply being edited (the edit sheet's item).
struct EditTarget: Identifiable {
    let id = UUID()
    let chatID: Chat.ID
    let messageID: UUID
}

/// Rating and editing replies, and exporting everything as candidate training data (trida_mlx.datamine).
extension AppState {
    /// 👍 / 👎 on a reply of the current chat; the same button again clears it.
    func rate(_ messageID: UUID, _ rating: String) {
        guard let ci = currentChatIndex else { return }
        update(chats[ci].id, messageID) { $0.feedback = $0.feedback == rating ? nil : rating }
    }

    func beginEdit(_ m: ChatMessage) {
        guard let ci = currentChatIndex else { return }
        editText = m.content
        editTarget = EditTarget(chatID: chats[ci].id, messageID: m.id)
    }

    /// Replaces the reply with the user's version; the model's original is kept in `editedFrom` (first edit only).
    func saveEdit() {
        guard let t = editTarget else { return }
        let text = editText
        update(t.chatID, t.messageID) { m in
            guard m.content != text else { return }
            if m.editedFrom == nil { m.editedFrom = m.content }
            m.content = text
            if m.editedFrom == text { m.editedFrom = nil }   // edited back to the original
        }
        editTarget = nil
    }

    static let chatsDir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("Zoro-1 Chat/chats", isDirectory: true)

    /// Asks for consent and a folder, then writes the chats and request logs there as JSONL training data.
    func exportTrainingData() {
        let alert = NSAlert()
        alert.messageText = "Export training data?"
        alert.informativeText = """
            This reads your saved chats and request logs and writes candidate training data (JSONL) to a folder \
            you choose: preference pairs (drafts rewritten after a reply check, replies you edited), replies you \
            rated, tool-use turns, the decision model's verdicts, failed generations and full trajectories.

            Account names in paths, email addresses and access tokens are replaced, and images are left out. \
            Messages, file names and file contents that tools read stay in: review the files before sharing them. \
            Nothing is uploaded.
            """
        alert.addButton(withTitle: "Choose Folder…")
        alert.addButton(withTitle: "Cancel")
        guard alert.runModal() == .alertFirstButtonReturn else { return }
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.prompt = "Export Here"
        guard panel.runModal() == .OK, let dir = panel.url else { return }
        chatStore.save(chats, cards: decisionCards)          // include the latest changes
        let out = dir.appendingPathComponent("zoro-training-data-" + Self.stamp())
        let rt = runtime
        let (exe, args) = rt.datamine(["--out", out.path, "--chats", Self.chatsDir.path,
                                       "--logs", Self.requestLogDir.path])
        let runner = CommandRunner()
        Task {
            var log = ""
            let box = LogBox()
            let code = (try? await runner.run(exe, args, cwd: rt.workingDirectory, extraEnv: rt.environment,
                                              onOutput: { t in box.append(t) })) ?? -1
            log = box.text
            let done = NSAlert()
            if code == 0 {
                done.messageText = "Training data exported"
                let summary = log.split(separator: "\n").last(where: { $0.hasPrefix("{") }).map(String.init) ?? ""
                done.informativeText = Self.describeManifest(summary) + "\n\n" + out.path
                NSWorkspace.shared.activateFileViewerSelecting([out])
            } else {
                done.messageText = "Export failed (exit \(code))"
                done.informativeText = String(log.suffix(800))
            }
            done.runModal()
        }
    }

    static func stamp() -> String {
        let f = DateFormatter()
        f.dateFormat = "yyyyMMdd-HHmmss"
        return f.string(from: Date())
    }

    /// "preference_pairs 3 · feedback 5 · …" from datamine's manifest line.
    static func describeManifest(_ line: String) -> String {
        guard let d = line.data(using: .utf8),
              let o = try? JSONSerialization.jsonObject(with: d) as? [String: Any],
              let files = o["files"] as? [String: Any] else { return "" }
        return files.keys.sorted().map { "\($0): \(files[$0] ?? 0)" }.joined(separator: "\n")
    }
}

/// Collects a command's output from the runner's background thread.
final class LogBox: @unchecked Sendable {
    private let lock = NSLock()
    private var buf = ""
    func append(_ s: String) { lock.lock(); buf += s; lock.unlock() }
    var text: String { lock.lock(); defer { lock.unlock() }; return buf }
}

struct EditReplySheet: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Edit reply").font(.headline)
            Text("Rewrite the reply into what it should have been. The model sees your version from now on, and the export keeps both (the model's as rejected, yours as chosen).")
                .font(.caption).foregroundStyle(.secondary)
            TextEditor(text: $state.editText)
                .font(.body)
                .frame(minHeight: 240)
                .overlay(RoundedRectangle(cornerRadius: 4).stroke(Color.secondary.opacity(0.3)))
            HStack {
                Spacer()
                Button("Cancel", role: .cancel) { state.editTarget = nil }
                    .keyboardShortcut(.cancelAction)
                Button("Save") { state.saveEdit() }
                    .keyboardShortcut(.return, modifiers: .command)
            }
        }
        .padding(20)
        .frame(width: 620)
    }
}
