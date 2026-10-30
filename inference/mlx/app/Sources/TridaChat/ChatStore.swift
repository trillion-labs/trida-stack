import Foundation
import TridaChatCore

/// A decision card as saved with its chat: the questions and the server's JSON answer.
struct StoredCard: Codable {
    var questions: [DecisionQuestion]
    var result: Data?
    var phase: String
    var liveReasoning: String
    var error: String?

    init(_ c: DecisionCard) {
        questions = c.questions
        result = c.result?.raw
        phase = c.phase
        liveReasoning = c.liveReasoning
        error = c.error
    }

    var card: DecisionCard {
        var c = DecisionCard(questions: questions)
        c.running = false
        c.result = result.flatMap { DecisionResult(raw: $0, questions: questions) }
        c.phase = phase
        c.liveReasoning = liveReasoning
        c.error = error ?? (c.result == nil ? "Interrupted." : nil)
        return c
    }
}

/// One chat on disk.
struct StoredChat: Codable {
    var version = 1
    var chat: Chat
    var cards: [String: StoredCard]
}

/// Saves chats as one JSON file each in ~/Library/Application Support/Zoro-1 Chat/chats, so conversations
/// (with their images, tool results and decision cards) survive quitting the app.
@MainActor
final class ChatStore {
    static let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("Zoro-1 Chat/chats", isDirectory: true)

    /// What was last written for each chat, so only changed chats are rewritten.
    private var saved: [Chat.ID: Chat] = [:]
    private var pending: Task<Void, Never>?

    private func url(_ id: Chat.ID) -> URL { Self.dir.appendingPathComponent("\(id.uuidString).json") }

    /// All saved chats, most recently updated first, with their decision cards.
    func load() -> (chats: [Chat], cards: [UUID: DecisionCard]) {
        let fm = FileManager.default
        guard let files = try? fm.contentsOfDirectory(at: Self.dir, includingPropertiesForKeys: nil) else { return ([], [:]) }
        var chats: [Chat] = []
        var cards: [UUID: DecisionCard] = [:]
        let dec = JSONDecoder()
        for f in files where f.pathExtension == "json" {
            guard let data = try? Data(contentsOf: f), let s = try? dec.decode(StoredChat.self, from: data) else { continue }
            var chat = s.chat
            // A reply or tool that was running when the app quit.
            for i in chat.messages.indices {
                for j in chat.messages[i].toolRuns.indices where chat.messages[i].toolRuns[j].status != .done
                    && chat.messages[i].toolRuns[j].status != .denied {
                    chat.messages[i].toolRuns[j].status = .denied
                    if chat.messages[i].toolRuns[j].result.isEmpty { chat.messages[i].toolRuns[j].result = "interrupted" }
                }
            }
            for (k, v) in s.cards { if let id = UUID(uuidString: k) { cards[id] = v.card } }
            saved[chat.id] = chat
            chats.append(chat)
        }
        chats.sort { $0.updated > $1.updated }
        return (chats, cards)
    }

    /// Writes the chats that changed, a moment after the last change (replies stream token by token).
    func scheduleSave(_ chats: [Chat], cards: [UUID: DecisionCard], immediately: Bool = false) {
        pending?.cancel()
        pending = Task { [weak self] in
            if !immediately { try? await Task.sleep(nanoseconds: 1_000_000_000) }
            if Task.isCancelled { return }
            self?.save(chats, cards: cards)
        }
    }

    func save(_ chats: [Chat], cards: [UUID: DecisionCard]) {
        try? FileManager.default.createDirectory(at: Self.dir, withIntermediateDirectories: true)
        let enc = JSONEncoder()
        for chat in chats where !chat.messages.isEmpty && saved[chat.id] != chat {
            var mine: [String: StoredCard] = [:]
            for m in chat.messages {
                if case let .card(id)? = m.decide, let c = cards[id] { mine[id.uuidString] = StoredCard(c) }
            }
            var stamped = chat
            stamped.updated = Date()
            guard let data = try? enc.encode(StoredChat(chat: stamped, cards: mine)) else { continue }
            if (try? data.write(to: url(chat.id), options: .atomic)) != nil { saved[chat.id] = chat }
        }
    }

    func delete(_ id: Chat.ID) {
        saved[id] = nil
        try? FileManager.default.removeItem(at: url(id))
    }
}
