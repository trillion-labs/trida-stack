import Foundation

/// What the learning pipeline needs from the app: Zoro-1's typed decisions and a short completion.
public protocol KnowledgeBackend: Sendable {
    /// P(yes) for a yes/no question about `state` (a direct decision, no reasoning).
    func yesNo(state: [(String, String)], question: String) async throws -> Double
    /// P(option) for each option key.
    func choose(state: [(String, String)], question: String, options: [(key: String, label: String)]) async throws
        -> [String: Double]
    /// A completion without thinking, for the JSON of proposed facts.
    func complete(system: String, user: String, maxTokens: Int) async throws -> String
}

/// One finished exchange: the user's message and the reply, and what web tools returned for it.
public struct Exchange: Sendable {
    public var chatID: String
    public var messageID: String
    public var userText: String
    public var replyText: String
    /// The text the web tools returned in this turn (page outlines, find_in_page snippets, sections), or nil.
    public var webEvidence: String?
    /// The pages it came from.
    public var webSources: [String]
    /// The reply passed the "backed by the pages" check (web facts are only learned then).
    public var webGrounded: Bool
    public var date: Date

    public init(chatID: String, messageID: String, userText: String, replyText: String, webEvidence: String? = nil,
                webSources: [String] = [], webGrounded: Bool = false, date: Date = Date()) {
        self.chatID = chatID
        self.messageID = messageID
        self.userText = userText
        self.replyText = replyText
        self.webEvidence = webEvidence
        self.webSources = webSources
        self.webGrounded = webGrounded
        self.date = date
    }

    var webAllowed: Bool { webGrounded && !(webEvidence ?? "").isEmpty }
}

/// A fact the model proposed, checked, with its two ends linked to nodes (or to be created).
public struct ProposedFact: Equatable, Sendable {
    public enum End: Equatable, Sendable {
        case me
        case existing(String)
        case new(name: String, type: NodeType)
    }

    public var subject: End
    public var relation: String
    public var object: End
    public var note: String?
    public var fromWeb: Bool
    public var quote: String?

    public init(subject: End, relation: String, object: End, note: String? = nil, fromWeb: Bool = false,
                quote: String? = nil) {
        self.subject = subject
        self.relation = relation
        self.object = object
        self.note = note
        self.fromWeb = fromWeb
        self.quote = quote
    }
}

/// A proposal as parsed from the model's JSON, before checks and linking.
public struct RawFact: Equatable, Sendable {
    public var subject: String
    public var subjectType: NodeType
    public var relation: String
    public var object: String
    public var objectType: NodeType
    public var source: String
    public var quote: String?
    public var note: String?

    public init(subject: String, subjectType: NodeType, relation: String, object: String, objectType: NodeType,
                source: String, quote: String? = nil, note: String? = nil) {
        self.subject = subject
        self.subjectType = subjectType
        self.relation = relation
        self.object = object
        self.objectType = objectType
        self.source = source
        self.quote = quote
        self.note = note
    }
}

/// Learning from a finished exchange: a strict yes/no gate, the model's proposed facts as JSON, checks against
/// their source, and entity linking as a choice. The choices go to Zoro-1's typed decisions; only the proposal is
/// free text.
public enum Learner {
    /// The gate passes at P(yes) ≥ this: remembering too much clutters the graph and feels intrusive.
    public static let gateThreshold = 0.8
    /// Linking to an existing node needs P ≥ this; otherwise a new node is made.
    public static let linkThreshold = 0.6
    public static let maxFacts = 3

    public static let gateQuestion = "Should something from this exchange be remembered for future chats: a lasting "
        + "fact about the user, their work, or the people, places and things in their life; a preference; a "
        + "correction of a fact; or a fact the reply looked up on the web to answer the user's question? Answer no "
        + "for small talk, one-off requests, and questions without a factual answer."

    static func clip(_ s: String, _ n: Int) -> String { s.count > n ? String(s.prefix(n)) + "…" : s }

    // MARK: - the pipeline

    /// Gate → proposal → checks → entity linking. Returns the facts to apply (possibly none). Throws only when the
    /// backend fails.
    public static func propose(_ x: Exchange, graph: KnowledgeGraph, backend: KnowledgeBackend) async throws
        -> [ProposedFact] {
        guard !x.userText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return [] }
        var state: [(String, String)] = [("user message", clip(x.userText, 2000)), ("reply", clip(x.replyText, 1500))]
        if x.webAllowed {
            state.append(("web lookup", "the reply is backed by: " + x.webSources.joined(separator: ", ")))
        }
        let p = try await backend.yesNo(state: state, question: gateQuestion)
        guard p >= gateThreshold else { return [] }
        let text = try await backend.complete(system: proposalSystem, user: proposalUser(x), maxTokens: 500)
        var out: [ProposedFact] = []
        var taken = graph           // later facts can link to the nodes earlier ones will create
        for r in parse(text).prefix(maxFacts) {
            guard let c = check(r, x) else { continue }
            // a node that only exists in `taken` (made for an earlier fact here) is still "new" for the real graph
            func real(_ e: ProposedFact.End) -> ProposedFact.End {
                if case let .existing(id) = e, graph.node(id) == nil, let n = taken.node(id) {
                    return .new(name: n.name, type: n.type)
                }
                return e
            }
            let s = try await real(link(c.subject, type: c.subjectType, x, graph: taken, backend: backend))
            let o = try await real(link(c.object, type: c.objectType, x, graph: taken, backend: backend))
            guard s != o else { continue }
            let f = ProposedFact(subject: s, relation: c.relation, object: o, note: c.note,
                                 fromWeb: c.source == "web", quote: c.quote)
            taken.apply([f], x)
            out.append(f)
        }
        return out
    }

    // MARK: - proposal

    public static var proposalSystem: String {
        let rels = Relations.all.map { "  - \($0.key): \($0.hint)" }.joined(separator: "\n")
        let types = NodeType.allCases.filter { $0 != .me }.map(\.rawValue).joined(separator: ", ")
        return "You extract facts worth remembering from a conversation, as a JSON array of at most \(maxFacts) "
            + "items ([] if there is nothing). Each item:\n"
            + #"{"subject": "...", "subject_type": "...", "relation": "...", "object": "...", "object_type": "...", "#
            + #""source": "user" or "web", "quote": "...", "note": "..."}"# + "\n"
            + "- Use \"me\" as the subject or object for the user.\n"
            + "- relation is one of:\n\(rels)\n"
            + "- types: \(types) (value is a literal such as a date, a number or a version).\n"
            + "- source \"user\": the user said it in their message. source \"web\": the reply found it on a web "
            + "page; then quote is the exact line from the web results that says it, copied, not paraphrased.\n"
            + "- Write names as they are usually written (for example \"Lee Jae-myung\", \"South Korea\"); keep the "
            + "user's language for names that only exist in it. Never include passwords, keys or account numbers.\n"
            + "Reply with the JSON array only."
    }

    static func proposalUser(_ x: Exchange) -> String {
        var s = "User's message:\n\(clip(x.userText, 3000))\n\nReply:\n\(clip(x.replyText, 2000))"
        if x.webAllowed, let w = x.webEvidence {
            s += "\n\nWeb results the reply used (from \(x.webSources.joined(separator: ", "))):\n\(clip(w, 5000))"
        }
        return s
    }

    /// The facts in the model's output: the first JSON array in it (text or a code fence around it is fine).
    public static func parse(_ text: String) -> [RawFact] {
        guard let lo = text.firstIndex(of: "["), let hi = text.lastIndex(of: "]"), lo < hi,
              let data = String(text[lo...hi]).data(using: .utf8),
              let arr = (try? JSONSerialization.jsonObject(with: data)) as? [[String: Any]] else { return [] }
        return arr.compactMap { d -> RawFact? in
            func str(_ k: String) -> String? {
                guard let v = d[k] as? String else { return nil }
                let t = v.trimmingCharacters(in: .whitespacesAndNewlines)
                return t.isEmpty ? nil : t
            }
            guard let s = str("subject"), let r = str("relation"), let o = str("object") else { return nil }
            return RawFact(subject: s, subjectType: NodeType(rawValue: str("subject_type") ?? "") ?? .topic,
                           relation: r.lowercased(), object: o,
                           objectType: NodeType(rawValue: str("object_type") ?? "") ?? .topic,
                           source: str("source")?.lowercased() == "web" ? "web" : "user",
                           quote: str("quote"), note: str("note"))
        }
    }

    // MARK: - checks

    static let meWords: Set<String> = ["me", "user", "the user", "i", "myself", "나", "저", "사용자"]

    static func isMe(_ s: String, _ t: NodeType) -> Bool { t == .me || meWords.contains(KnowledgeGraph.normalize(s)) }

    /// Keys, tokens, passwords, card, account and resident registration numbers: never stored.
    static let secretPatterns = [
        #"sk-[A-Za-z0-9_\-]{16,}"#, #"gh[pousr]_[A-Za-z0-9]{20,}"#, #"AKIA[0-9A-Z]{16}"#, #"xox[abp]-[A-Za-z0-9-]{10,}"#,
        #"-----BEGIN"#, #"(?i)\b(password|passwd|passcode|api[ _-]?key|secret|token)\b"#,
        #"비밀번호|암호|계좌번호|카드번호|주민등록번호"#,
        #"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b"#, #"\b\d{6}-\d{7}\b"#,
    ]

    public static func looksSecret(_ s: String) -> Bool {
        secretPatterns.contains { s.range(of: $0, options: .regularExpression) != nil }
    }

    /// A proposal is kept only when it is well formed and backed by its source: the user's message for "user",
    /// a quote that really is in the web results for "web" (and a page never says anything about the user).
    public static func check(_ raw: RawFact, _ x: Exchange) -> RawFact? {
        var r = raw
        guard Relations.named(r.relation) != nil else { return nil }
        let texts = [r.subject, r.object, r.quote ?? "", r.note ?? ""]
        guard texts.allSatisfy({ !$0.contains("\n") && $0.count <= 240 }), r.subject.count <= 80, r.object.count <= 80,
              !texts.contains(where: looksSecret) else { return nil }
        let subjMe = isMe(r.subject, r.subjectType), objMe = isMe(r.object, r.objectType)
        if subjMe { r.subject = KnowledgeGraph.meName; r.subjectType = .me }
        if objMe { r.object = KnowledgeGraph.meName; r.objectType = .me }
        guard KnowledgeGraph.normalize(r.subject) != KnowledgeGraph.normalize(r.object) else { return nil }
        if r.source == "web" {
            guard x.webAllowed, !subjMe, !objMe, let q = r.quote, q.count >= 8,
                  let evidence = x.webEvidence.map(KnowledgeGraph.normalize),
                  evidence.contains(KnowledgeGraph.normalize(q)) else { return nil }
            return r
        }
        // from the user: about the user, or naming something the user actually wrote
        let said = KnowledgeGraph.normalize(x.userText)
        let named = [r.subject, r.object].contains { name in
            let n = KnowledgeGraph.normalize(name)
            return n.count >= 2 && n != KnowledgeGraph.normalize(KnowledgeGraph.meName) && said.contains(n)
        }
        return subjMe || objMe || named ? r : nil
    }

    // MARK: - entity linking

    /// An end of a fact as a node: the user, an exact name match, a candidate Zoro-1 picks, or a new node.
    static func link(_ name: String, type: NodeType, _ x: Exchange, graph: KnowledgeGraph, backend: KnowledgeBackend)
        async throws -> ProposedFact.End {
        if type == .me { return .me }
        if let exact = graph.node(named: name, type: type), exact.type != .me { return .existing(exact.id) }
        if type == .value { return .new(name: name, type: type) }
        let cands = graph.similar(to: name)
        guard !cands.isEmpty else { return .new(name: name, type: type) }
        // short keys ("1", "2", …) for the decision; node ids are mapped back
        var options: [(key: String, label: String)] = cands.enumerated().map { i, n in
            let fact = graph.currentEdges(touching: [n.id]).first.map { "; e.g. " + graph.describe($0) } ?? ""
            let also = n.aliases.isEmpty ? "" : "; also called " + n.aliases.prefix(3).joined(separator: ", ")
            return (key: String(i + 1), label: "\(n.name) (\(n.type.rawValue)\(also)\(fact))")
        }
        options.append((key: "new", label: "none of these: \(name) is something else"))
        let ps = try await backend.choose(
            state: [("name in the conversation", name), ("its type", type.rawValue),
                    ("conversation", clip(x.userText, 800))],
            question: "Which entry already in the user's knowledge graph is the same thing as \"\(name)\"?",
            options: options)
        if let best = ps.filter({ $0.key != "new" }).max(by: { $0.value < $1.value }), best.value >= linkThreshold,
           let i = Int(best.key), cands.indices.contains(i - 1) {
            return .existing(cands[i - 1].id)
        }
        return .new(name: name, type: type)
    }
}

extension KnowledgeGraph {
    /// Applies proposed facts from one exchange and records what changed, for the chip's Undo. Returns nil when
    /// nothing was added (all duplicates, or web facts that a fact the user stated outranks).
    @discardableResult
    public mutating func apply(_ facts: [ProposedFact], _ x: Exchange) -> LearnedRecord? {
        var rec = records[x.messageID] ?? LearnedRecord()
        func resolve(_ end: ProposedFact.End) -> String? {
            switch end {
            case .me:
                let had = me != nil
                let id = ensureMe()
                if !had { rec.nodes.append(id) }
                return id
            case let .existing(id):
                return node(id) != nil ? id : nil          // deleted meanwhile: skip the fact
            case let .new(name, type):
                if let n = node(named: name, type: type), n.type == type { return n.id }
                let id = addNode(name: name, type: type, at: x.date)
                rec.nodes.append(id)
                return id
            }
        }
        for f in facts {
            guard let s = resolve(f.subject), let o = resolve(f.object), s != o else { continue }
            var e = KGEdge(subject: s, relation: f.relation, object: o, origin: f.fromWeb ? .web : .user,
                           learnedAt: x.date)
            e.note = f.note
            e.chatID = x.chatID
            e.messageID = x.messageID
            e.validFrom = x.date
            if f.fromWeb {
                e.sourceURL = x.webSources.first
                e.asOf = x.date
                e.quote = f.quote
            }
            guard let added = addFact(e, at: x.date) else { continue }
            rec.edges.append(added.id)
            rec.ended += added.ended
        }
        // nodes made for facts that were then skipped
        for id in rec.nodes where edges(touching: id).isEmpty && node(id)?.type != .me {
            nodes.removeAll { $0.id == id }
        }
        rec.nodes = rec.nodes.filter { node($0) != nil }
        guard !rec.edges.isEmpty else { return nil }
        records[x.messageID] = rec
        return rec
    }
}
