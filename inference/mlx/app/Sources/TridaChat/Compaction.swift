import Foundation
import TridaChatCore

/// Conversation compaction, as in Claude Code: when a chat nearly fills the context window, the model
/// summarizes it, and from then on it sees the summary (in the system message) instead of the messages
/// before it. The earlier messages stay visible in the chat; only what is sent to the model changes.
/// Runs automatically before a reply when the chat is over `compactAt` of the context length, when the
/// server rejects a prompt as too long, and on /compact.
extension AppState {
    /// Fraction of the context length at which a chat is compacted before the next reply.
    static let compactAt = 0.8
    /// Reply budget for the summary itself.
    static let summaryMaxTokens = 8192

    static let summaryInstruction = """
        Summarize our conversation so far so that it can continue without the earlier messages. Include the \
        user's goals and requests; what was done, found and decided; important facts, names and numbers; file \
        paths, commands and code exactly where they matter; what the tool calls returned; and anything still \
        open or promised. Be complete but concise. Write it in the language of the conversation. Output only \
        the summary.
        """

    /// Index of the latest compaction notice: the model sees only the messages after it.
    func compactionIndex(_ chat: Chat) -> Int? {
        chat.messages.lastIndex { $0.compaction != nil }
    }

    /// Tokens the chat would take now: the last reply's prompt + output as the server counted them, plus a
    /// rough count (3 characters a token) for messages added since.
    func contextUsed(_ chatID: Chat.ID) -> Int {
        guard let chat = chats.first(where: { $0.id == chatID }) else { return 0 }
        let start = (compactionIndex(chat) ?? -1) + 1
        let live = Array(chat.messages[start...])
        let rough = { (ms: ArraySlice<ChatMessage>) in
            ms.reduce(0) { $0 + ($1.content.count + $1.reasoning.count) / 3 + $1.images.count * 1000 }
        }
        if let i = live.lastIndex(where: { ($0.stats?.promptTokens ?? 0) > 0 }), let st = live[i].stats {
            return st.promptTokens + st.newTokens + rough(live[(i + 1)...])
        }
        return 2000 + rough(live[...])   // the system prompt and tool schemas, roughly
    }

    func needsCompaction(_ chatID: Chat.ID) -> Bool {
        Double(contextUsed(chatID)) > Self.compactAt * Double(settings.contextLength)
    }

    static func isContextOverflow(_ error: Error) -> Bool {
        let s = error.localizedDescription.lowercased()
        return s.contains("context_length_exceeded") || s.contains("maximum context length")
    }

    /// Summarizes the chat (everything before its latest user message when `keepLastUser`) and inserts a
    /// compaction notice holding the summary. Returns false when there was nothing to summarize or it failed.
    @discardableResult
    func compact(_ chatID: Chat.ID, keepLastUser: Bool, reason: String) async -> Bool {
        guard let ci = chats.firstIndex(where: { $0.id == chatID }) else { return false }
        let msgs = chats[ci].messages
        let lastUser = keepLastUser ? msgs.lastIndex { $0.role == .user && $0.decide != .prompt } : nil
        let cutoff = lastUser ?? msgs.count            // summarize messages before this index
        let start = (compactionIndex(chats[ci]) ?? -1) + 1
        guard start <= cutoff, msgs[start..<cutoff].contains(where: { $0.decide != .prompt && !$0.content.isEmpty }) else { return false }

        var notice = ChatMessage(role: .assistant, content: "Summarizing the earlier conversation to free up context (\(reason))…")
        notice.decide = .prompt
        chats[ci].messages.insert(notice, at: cutoff)
        let before = contextUsed(chatID)

        // What the model sees now, without the kept user message, plus the request to summarize.
        var messages = history(chatID, tools: nil)
        if lastUser != nil, let last = messages.last, last["role"] as? String == "user" { messages.removeLast() }
        messages.append(["role": "user", "content": Self.summaryInstruction])
        var options = settings.generation
        options.thinking = false                        // a summary needs no reasoning pass
        options.maxTokens = Self.summaryMaxTokens
        let body = ChatClient.requestBody(messages: messages, tools: nil, options: options, model: servedName)
        var summary = ""
        do {
            for try await event in client.stream(body: body) {
                switch event {
                case let .content(t):
                    summary += t
                    let n = summary.count
                    update(chatID, notice.id) { $0.content = "Summarizing the earlier conversation to free up context (\(reason))… \(n) characters" }
                case let .prefill(done, total):
                    prefill = (done, total)
                default:
                    break
                }
            }
        } catch {
            update(chatID, notice.id) { $0.content = "Could not summarize the conversation: \(error.localizedDescription)" }
            prefill = nil
            return false
        }
        prefill = nil
        summary = summary.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !summary.isEmpty else {
            update(chatID, notice.id) { $0.content = "Could not summarize the conversation (empty summary)." }
            return false
        }
        update(chatID, notice.id) {
            $0.content = "**Earlier conversation summarized** (\(reason)): about \(before.formatted()) tokens → "
                + "a \((summary.count / 3).formatted())-token summary. The model now sees the summary instead of the messages above."
            $0.compaction = summary
        }
        return true
    }

    /// /compact: summarize the whole chat now.
    func compactNow() {
        guard !isGenerating, let ci = currentChatIndex else { return }
        let chatID = chats[ci].id
        isGenerating = true
        generationTask = Task {
            defer { isGenerating = false }
            if !(await compact(chatID, keepLastUser: false, reason: "/compact")) {
                appendNotice("Nothing new to summarize.")
            }
        }
    }
}
