// The knowledge graph wired into the app (see Knowledge/INTEGRATION.md): Zoro-1 as its learning backend, learning
// after each final reply, and the remembered facts added to the latest user message.
import SwiftUI
import TridaChatCore
import TridaKnowledge
import TridaKnowledgeUI

/// Zoro-1's decisions and completions, for the knowledge graph's learning pipeline.
final class AppKnowledgeBackend: KnowledgeBackend, @unchecked Sendable {
    unowned let app: AppState

    init(app: AppState) { self.app = app }

    private static func state(_ kv: [(String, String)]) -> OrderedJSON { .object(kv.map { ($0.0, .string($0.1)) }) }

    @MainActor private var settings: DecisionSettings {
        var s = app.decisionSettings
        s.reasoning = "never"                     // direct decisions: these run after every reply
        return s
    }

    @MainActor func yesNo(state: [(String, String)], question: String) async throws -> Double {
        let q = [DecisionQuestion.yesNo("q", question)]
        let r = try await app.runDecision(Decision.body(state: Self.state(state), questions: q, settings: settings), q,
                                          log: nil) { _ in }
        return r.answers["q"]?.value ?? 0
    }

    @MainActor func choose(state: [(String, String)], question: String, options: [(key: String, label: String)])
        async throws -> [String: Double] {
        let q = [DecisionQuestion.choice("q", question, options.map { ($0.key, $0.label) })]
        let r = try await app.runDecision(Decision.body(state: Self.state(state), questions: q, settings: settings), q,
                                          log: nil) { _ in }
        return Dictionary((r.answers["q"]?.options ?? []).map { ($0.key, $0.p) }, uniquingKeysWith: max)
    }

    @MainActor func complete(system: String, user: String, maxTokens: Int) async throws -> String {
        var options = app.settings.generation
        options.thinking = false
        options.maxTokens = maxTokens
        options.temperature = 0
        let body = ChatClient.requestBody(messages: [["role": "system", "content": system],
                                                     ["role": "user", "content": user]],
                                          tools: nil, options: options, model: app.servedName)
        var out = ""
        for try await event in app.client.stream(body: body) {
            if case let .content(t) = event { out += t }
        }
        return out
    }
}

extension AppState {
    /// After a turn's text reply (and its review, which may have rewritten it): learn from the final reply.
    func learnFromLatestReply(_ chatID: Chat.ID) {
        guard let reply = chats.first(where: { $0.id == chatID })?.messages.last(where: {
            $0.role == .assistant && $0.decide == nil && $0.superseded != true && !$0.content.isEmpty
        }) else { return }
        learnFromReply(chatID: chatID, replyID: reply.id)
    }

    /// Learns from one reply in the background.
    func learnFromReply(chatID: Chat.ID, replyID: UUID) {
        guard knowledge.enabled, supportsDecision, let chat = chats.first(where: { $0.id == chatID }),
              let ri = chat.messages.firstIndex(where: { $0.id == replyID }),
              let ui = chat.messages[..<ri].lastIndex(where: { $0.role == .user && $0.decide == nil }) else { return }
        let reply = chat.messages[ri]
        guard reply.superseded != true, !reply.content.isEmpty else { return }
        // what the web tools returned in this turn, and from where
        var evidence: [String] = []
        var sources: [String] = []
        var files: [FileText] = []
        for m in chat.messages[(ui + 1)...ri] {
            for run in m.toolRuns where AgentTools.webTools.contains(run.call.name) && run.status == .done
                && !run.result.hasPrefix("error") {
                evidence.append(run.result)
                // fetch_url / open_link outlines put the page address on their second line
                let lines = run.result.split(separator: "\n", maxSplits: 2)
                if lines.count > 1, lines[1].hasPrefix("http") { sources.append(String(lines[1])) }
            }
            // documents read for the user: facts quoted from them can be learned too
            for run in m.toolRuns where ["read_file", "read_pdf"].contains(run.call.name) && run.status == .done
                && !run.result.hasPrefix("error") {
                guard let p = run.call.argumentDict["path"] as? String else { continue }
                let path = p.hasPrefix("/") ? p : settings.workspace.appendingPathComponent(p).standardizedFileURL.path
                files.append(FileText(path: path, text: run.result))
            }
        }
        var x = Exchange(chatID: chatID.uuidString, messageID: replyID.uuidString,
                         userText: chat.messages[ui].content, replyText: reply.content,
                         webEvidence: evidence.isEmpty ? nil : evidence.joined(separator: "\n\n"),
                         webSources: sources,
                         webGrounded: reply.check?.contains("✓ backed by the pages") == true,
                         files: files)
        x.chatTitle = chat.title
        x.model = servedName
        let backend = knowledgeBackend
        Task { @MainActor [weak self] in
            guard let self else { return }
            if await self.knowledge.learn(x, backend: backend) != nil { self.react(.notes) }   // the mascot: "Noted"
        }
    }

    /// /update-kg: learns from the whole chat, at the user's request. Every turn's messages, and every page and
    /// document read in it, go to the learner at once (no gate); web and document facts still need a quote that is
    /// really in what was read. The facts are recorded on the latest reply, so its chip shows them (and can undo
    /// them). `focus`: what the user wants kept, or facts they add ("/update-kg the blog posts").
    func updateKnowledge(chatID: Chat.ID, focus: String) {
        guard let chat = chats.first(where: { $0.id == chatID }) else { return }
        guard knowledge.enabled else {
            notice(chatID, "Learning is off. Turn on \"Learn from chats automatically\" in Settings to use /update-kg.")
            return
        }
        let msgs = chat.messages.filter { $0.decide == nil && $0.superseded != true }
        let users = msgs.filter { $0.role == .user }.map(\.content).filter { !$0.isEmpty }
        let replies = msgs.filter { $0.role == .assistant && !$0.content.isEmpty }
        guard let last = replies.last, !users.isEmpty else {
            notice(chatID, "Nothing in this chat to learn from yet.")
            return
        }
        var evidence: [String] = []
        var sources: [String] = []
        var files: [FileText] = []
        for m in msgs {
            for run in m.toolRuns where run.status == .done && !run.result.hasPrefix("error") {
                if AgentTools.webTools.contains(run.call.name) {
                    evidence.append(run.result)
                    let lines = run.result.split(separator: "\n", maxSplits: 2)
                    if lines.count > 1, lines[1].hasPrefix("http"), !sources.contains(String(lines[1])) {
                        sources.append(String(lines[1]))
                    }
                } else if ["read_file", "read_pdf"].contains(run.call.name),
                          let p = run.call.argumentDict["path"] as? String {
                    let path = p.hasPrefix("/") ? p : settings.workspace.appendingPathComponent(p).standardizedFileURL.path
                    files.append(FileText(path: path, text: run.result))
                }
            }
        }
        // the user's words first (the focus, then the messages); the latest replies first, so a long chat keeps them
        let userText = (focus.isEmpty ? [] : ["(/update-kg) " + focus]) + users
        let replyText = replies.reversed().map(\.content).joined(separator: "\n\n---\n\n")
        var x = Exchange(chatID: chatID.uuidString, messageID: last.id.uuidString,
                         userText: userText.joined(separator: "\n\n"), replyText: replyText,
                         webEvidence: evidence.isEmpty ? nil : evidence.joined(separator: "\n\n"),
                         webSources: sources, webGrounded: !evidence.isEmpty, files: files, explicit: true)
        x.wholeChat = true
        x.chatTitle = chat.title
        x.model = servedName
        let read = [evidence.isEmpty ? nil : "\(sources.count) page(s)", files.isEmpty ? nil : "\(files.count) document(s)"]
            .compactMap { $0 }
        notice(chatID, "Updating the knowledge graph from this chat" + (read.isEmpty ? "" : " and the "
                     + read.joined(separator: " and ") + " read in it") + "…")
        let backend = knowledgeBackend
        Task { @MainActor [weak self] in
            guard let self else { return }
            let had = Set(self.knowledge.record(x.messageID)?.edges ?? [])
            self.knowledge.lastError = nil
            let rec = await self.knowledge.learn(x, backend: backend)
            if rec == nil, let err = self.knowledge.lastError {
                self.notice(chatID, "Couldn't update the knowledge graph: \(err)")
                return
            }
            let added = (rec?.edges ?? []).filter { !had.contains($0) }
                .compactMap { id in self.knowledge.graph.edges.first { $0.id == id } }
            guard !added.isEmpty else {
                self.notice(chatID, "Nothing new to add: the facts in this chat are already in the knowledge graph, "
                                  + "or none could be checked against what was said or read.")
                return
            }
            let new = added.count
            self.react(.notes)
            let shown = added.prefix(12).map { "• " + self.knowledge.graph.describe($0) }
            self.notice(chatID, "Added \(new) fact\(new == 1 ? "" : "s") to the knowledge graph (⇧⌘K to see them, "
                              + "Undo on the reply's chip):\n" + shown.joined(separator: "\n")
                              + (new > shown.count ? "\n…" : ""))
        }
    }

    /// A one-line app message in that chat (never sent to the model), even if another chat is open by then.
    private func notice(_ chatID: Chat.ID, _ text: String) {
        guard let ci = chats.firstIndex(where: { $0.id == chatID }) else { return }
        var m = ChatMessage(role: .assistant, content: text)
        m.decide = .prompt
        chats[ci].messages.append(m)
    }

    /// Fixes the remembered facts for the latest user message, once (a retry or a tool step of the same turn keeps
    /// them). Computed when the turn starts, from what is known then.
    func stampKnowledge(_ chatID: Chat.ID) {
        guard let ci = chats.firstIndex(where: { $0.id == chatID }),
              let mi = chats[ci].messages.lastIndex(where: { $0.role == .user && $0.decide == nil }),
              chats[ci].messages[mi].knowledge == nil,
              let block = knowledge.contextBlock(for: chats[ci].messages[mi].content) else { return }
        chats[ci].messages[mi].knowledge = block
    }

    /// A user message as sent, with the facts it was stamped with appended (to its text, or as one more text part
    /// when it has images).
    static func withKnowledge(_ message: [String: Any], _ block: String?) -> [String: Any] {
        guard let block, message["role"] as? String == "user" else { return message }
        var m = message
        if let text = m["content"] as? String {
            m["content"] = text + "\n\n" + block
        } else if var parts = m["content"] as? [[String: Any]] {
            parts.append(["type": "text", "text": block])
            m["content"] = parts
        }
        return m
    }
}

/// View → Knowledge (⇧⌘K).
struct OpenKnowledgeButton: View {
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Button("Knowledge") { openWindow(id: "knowledge") }
    }
}

/// Tools that let Zoro-1 use the knowledge graph on purpose: look things up, remember what the user asks it to,
/// and forget facts the user says are wrong or private. (The facts relevant to each message are also added to it
/// automatically; see `stampKnowledge`.)
enum KnowledgeTools {
    static let names: Set<String> = ["recall", "remember", "forget"]

    static func fn(_ name: String, _ desc: String, _ props: [String: Any], _ required: [String] = []) -> [String: Any] {
        ["type": "function",
         "function": ["name": name, "description": desc,
                      "parameters": ["type": "object", "properties": props, "required": required]]]
    }

    static var specs: [[String: Any]] {
        let str: [String: Any] = ["type": "string"]
        return [
            fn("recall", "Look up what you remember from earlier chats with this user (their knowledge graph): facts "
               + "about them, their work, and people, places and things in their life, with when and where each was "
               + "learned. Without a query, lists what you know about the user. Use it before saying you don't know "
               + "something the user may have told you, or when they ask what you remember.", ["query": str]),
            fn("remember", "Save a fact the user explicitly asks you to remember (\"remember that…\", \"기억해줘\"). "
               + "Write the fact as one plain sentence in the user's words.", ["fact": str], ["fact"]),
            fn("forget", "Stop using remembered facts the user says are wrong, outdated or private (\"forget…\", "
               + "\"잊어줘\"). query names what to forget; the matching facts are marked as no longer true and can "
               + "be restored from the Knowledge window.", ["query": str], ["query"]),
        ]
    }

    static let systemPrompt = "You remember facts from earlier chats: the ones relevant to a message are added to it "
        + "as \"Facts remembered from earlier chats\". Use them when they help (say that the user told you, and how "
        + "long ago for facts that change), and call recall to look up more. When the user asks you to remember or "
        + "forget something, call remember or forget."
}

extension AppState {
    /// recall / remember / forget. `messageID` is the reply: what remember learns shows as that reply's chip.
    func runKnowledgeTool(_ call: ToolCall, chatID: Chat.ID, messageID: UUID) async -> String {
        let a = call.argumentDict
        switch call.name {
        case "recall":
            let q = (a["query"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            let g = knowledge.graph
            if q.isEmpty { return g.overview() }
            var out = g.search(q)
            if out.hasPrefix("No remembered facts") {
                // the query may name an entity whose facts do not contain the words: its current facts
                let ids = Set(g.mentioned(in: q).map(\.id))
                let facts = g.currentEdges(touching: ids)
                if !facts.isEmpty {
                    out = facts.prefix(20).map { "- \(g.describe($0)) (\(g.provenance($0)))" }.joined(separator: "\n")
                }
            }
            return out
        case "remember":
            guard knowledge.enabled else { return "error: learning from chats is turned off in Settings" }
            guard let fact = a["fact"] as? String, !fact.isEmpty else { return "error: missing argument 'fact'" }
            let userText = chats.first { $0.id == chatID }?.messages.last { $0.role == .user && $0.decide == nil }?.content ?? fact
            var x = Exchange(chatID: chatID.uuidString, messageID: messageID.uuidString, userText: userText,
                             replyText: fact, explicit: true)
            x.chatTitle = chats.first { $0.id == chatID }?.title
            x.model = servedName
            guard let rec = await knowledge.learn(x, backend: knowledgeBackend), !rec.edges.isEmpty else {
                return "Nothing new was saved: the fact is already remembered, or it could not be turned into a fact "
                    + "that names what the user said."
            }
            react(.notes)
            let g = knowledge.graph
            return "Remembered:\n" + rec.edges.compactMap { id in g.edges.first { $0.id == id }.map { "- " + g.describe($0) } }
                .joined(separator: "\n")
        case "forget":
            guard let q = a["query"] as? String, !q.isEmpty else { return "error: missing argument 'query'" }
            let hits = knowledge.graph.matching(q)
            guard !hits.isEmpty else { return "No remembered fact matches \"\(q)\"; nothing was changed." }
            let lines = hits.map { "- " + knowledge.graph.describe($0) }
            knowledge.mutate { g in
                for e in hits {
                    g.update(edge: e.id) {
                        $0.validTo = Date()
                        $0.log("forgotten", by: "Zoro-1 (you asked: \(q))")
                    }
                }
            }
            return "Marked as no longer true (the user can restore them in the Knowledge window):\n"
                + lines.joined(separator: "\n")
        default:
            return "unknown tool \(call.name)"
        }
    }
}
