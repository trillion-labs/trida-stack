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
    /// The documents the reply read for the user in this turn (read_file / read_pdf results), with their paths.
    public var files: [FileText]
    /// The user asked to remember this (the remember tool): no gate.
    public var explicit: Bool
    /// The whole chat at the user's request (/update-kg): every turn, every page and document read in it; no gate,
    /// more facts. Web and document facts still need a quote that is really in what was read.
    public var wholeChat = false
    /// Recorded on each fact: the chat's title and the model doing the learning.
    public var chatTitle: String?
    public var model: String?
    public var date: Date

    public init(chatID: String, messageID: String, userText: String, replyText: String, webEvidence: String? = nil,
                webSources: [String] = [], webGrounded: Bool = false, files: [FileText] = [], explicit: Bool = false,
                date: Date = Date()) {
        self.files = files
        self.explicit = explicit
        self.chatTitle = nil
        self.model = nil
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
    var filesAllowed: Bool { files.contains { !$0.text.isEmpty } }

    /// The file whose text contains `quote` (normalized), if any.
    func file(quoting quote: String) -> FileText? {
        let q = KnowledgeGraph.normalize(quote)
        return files.first { KnowledgeGraph.normalize($0.text).contains(q) }
    }
}

/// A document the reply read: its path and the text the tool returned.
public struct FileText: Sendable, Equatable {
    public var path: String
    public var text: String

    public init(path: String, text: String) {
        self.path = path
        self.text = text
    }
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
    /// From a document the reply read (`quote` is in it).
    public var fromFile = false
    public var quote: String?
    /// A relation this fact introduces (added to the graph with it).
    public var newRelation: Relation?
    /// The gate's P(yes) for its exchange (1 when the user asked or no gate ran).
    public var confidence = 1.0

    public init(subject: End, relation: String, object: End, note: String? = nil, fromWeb: Bool = false,
                quote: String? = nil, newRelation: Relation? = nil) {
        self.newRelation = newRelation
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
    /// The gate passes at P(yes) ≥ this. 0.8 turned out too strict with the calibrated (flattened) text
    /// temperature: "my name is …, I work at …" scored 0.80, "I mostly use Python" 0.60. The checks after the
    /// proposal keep out what the gate lets through by mistake.
    public static let gateThreshold = 0.5
    /// Linking to an existing node needs P ≥ this; otherwise a new node is made.
    public static let linkThreshold = 0.6
    public static let maxFacts = 3
    /// More facts per exchange when the reply read documents or pages (a summary can hold many).
    public static let maxFactsWithSources = 8
    /// For /update-kg (the whole chat).
    public static let maxFactsWholeChat = 15

    public static let fileGateQuestion = "Did the reply read documents for the user (such as blog posts, reports, "
        + "papers or notes, not source code) that state lasting facts worth remembering, for example what they "
        + "are about, who made something, or what was decided?"

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
        var gate = 1.0
        var state: [(String, String)] = [("user message", clip(x.userText, 2000)), ("reply", clip(x.replyText, 1500))]
        if x.webAllowed {
            state.append(("web lookup", "the reply is backed by: " + x.webSources.joined(separator: ", ")))
        }
        if x.filesAllowed {
            state.append(("documents read", x.files.map(\.path).joined(separator: ", ")))
        }
        // A grounded web answer is by definition a fact the user wanted: no gate (it scored 0.35-0.38 there,
        // since the question is mostly about the user's own life).
        // Documents get their own question first: the general one is about the user's life, and scored 0.26 on
        // summaries of six blog posts.
        if !x.webAllowed && !x.explicit {
            var p = 0.0
            if x.filesAllowed { p = try await backend.yesNo(state: state, question: fileGateQuestion) }
            if p < gateThreshold { p = try await backend.yesNo(state: state, question: gateQuestion) }
            guard p >= gateThreshold else { return [] }
            gate = p
        }
        let sources = x.webAllowed || x.filesAllowed
        let limit = x.wholeChat ? maxFactsWholeChat : sources ? maxFactsWithSources : maxFacts
        let text = try await backend.complete(system: proposalSystem(graph, limit: limit), user: proposalUser(x),
                                              maxTokens: x.wholeChat ? 2400 : sources ? 1400 : 500)
        var out: [ProposedFact] = []
        var taken = graph           // later facts can link to the nodes earlier ones will create
        for r in parse(text).prefix(limit * 3) {
            if out.count >= limit { break }                  // the limit counts facts that pass, not proposals
            guard let c = check(r, x) else { continue }
            // a node that only exists in `taken` (made for an earlier fact here) is still "new" for the real graph
            func real(_ e: ProposedFact.End) -> ProposedFact.End {
                if case let .existing(id) = e, graph.node(id) == nil, let n = taken.node(id) {
                    return .new(name: n.name, type: n.type)
                }
                return e
            }
            let fact = "\(c.subject) —\(c.relation.replacingOccurrences(of: "_", with: " "))→ \(c.object)"
            var s = try await real(link(c.subject, type: c.subjectType, x, fact: fact, graph: taken, backend: backend))
            var o = try await real(link(c.object, type: c.objectType, x, fact: fact, graph: taken, backend: backend))
            guard s != o else { continue }
            let resolved = try await resolveRelation(c, x, graph: taken, backend: backend)
            let rel = resolved.key
            // the types the fact connects, and its direction
            func type(_ e: ProposedFact.End) -> NodeType {
                switch e {
                case .me: return .me
                case let .existing(id): return taken.node(id)?.type ?? .topic
                case let .new(_, t): return t
                }
            }
            let forward = Relations.fits(rel, type(s), type(o)), backward = Relations.fits(rel, type(o), type(s))
            if !forward && !backward { continue }                   // e.g. a company "member of" a job title
            var turn = !forward
            // both ways fit (two organizations, two projects): Zoro-1 says which way the fact goes
            if forward && backward && Relations.signatures[rel] != nil && type(s) != .me && type(o) != .me {
                turn = try await !rightWayRound(c, rel, graph: taken, backend: backend)
            }
            if turn { swap(&s, &o) }
            var newRel = resolved.new
            // a relation made for an earlier fact here is still new for the real graph
            if newRel == nil, graph.relation(rel) == nil { newRel = taken.relation(rel) }
            var note = c.note
            if rel == "related_to", c.relation != "related_to" {           // too many new relations already
                let label = c.relation.replacingOccurrences(of: "_", with: " ")
                note = note.map { label + "; " + $0 } ?? label
            }
            var f = ProposedFact(subject: s, relation: rel, object: o, note: note,
                                 fromWeb: c.source == "web", quote: c.quote, newRelation: newRel)
            f.fromFile = c.source == "file"
            f.confidence = gate
            taken.apply([f], x)
            out.append(f)
        }
        return out
    }

    // MARK: - proposal

    public static func proposalSystem(_ graph: KnowledgeGraph = KnowledgeGraph(), limit: Int = maxFacts) -> String {
        let rels = graph.relations.prefix(80).map { "  - \($0.key): \($0.hint)" }.joined(separator: "\n")
        let types = NodeType.allCases.filter { $0 != .me }.map(\.rawValue).joined(separator: ", ")
        return "You extract facts worth remembering from a conversation, as a JSON array of at most \(limit) "
            + "items ([] if there is nothing). Each item:\n"
            + #"{"subject": "...", "subject_type": "...", "relation": "...", "object": "...", "object_type": "...", "#
            + #""source": "user", "web" or "file", "quote": "...", "note": "..."}"# + "\n"
            + "- Use \"me\" as the subject or object for the user.\n"
            + "- relation is one of these:\n\(rels)\n"
            + "  Only when none of them fits, write a new relation in 1–4 English words in snake_case (for example "
            + "reports_to, birthday, studies_at).\n"
            + "- types: \(types) (value is a literal such as a date, a number or a version; role is a job title "
            + "or position such as researcher or intern, never a team).\n"
            + "- The subject is what the fact is about, and the arrow goes from it: \"A member_of B\" means A "
            + "belongs to B; \"A created_by B\" means B made A.\n"
            + "- source \"user\": the user said it in their message. source \"web\": the reply found it on a web "
            + "page; then quote is the exact line from the web results that says it, copied, not paraphrased. "
            + "source \"file\": the reply read it in a document; then quote is the exact line from that document.\n"
            + "- A quote is copied from the web results or the documents, never from the replies.\n"
            + "- Write names as they are usually written (for example \"Lee Jae-myung\", \"South Korea\"); keep the "
            + "user's language for names that only exist in it. Never include passwords, keys or account numbers.\n"
            + "- Only facts the user states (about themselves, their work, or the world) or facts the web results "
            + "show. Never facts about the assistant taken from its own reply. Job openings, menus, navigation, "
            + "buttons and page headings are not facts.\n"
            + "Example. User: \"나는 서울에 살고 Trillion Labs에서 연구자로 일해. Python을 주로 써.\" ->\n"
            + #"[{"subject": "me", "subject_type": "me", "relation": "lives_in", "object": "Seoul", "object_type": "place", "source": "user"}, "#
            + #"{"subject": "me", "subject_type": "me", "relation": "works_at", "object": "Trillion Labs", "object_type": "org", "source": "user"}, "#
            + #"{"subject": "me", "subject_type": "me", "relation": "role", "object": "researcher", "object_type": "role", "source": "user"}]"# + "\n"
            + "Reply with the JSON array only."
    }

    public static func proposalUser(_ x: Exchange) -> String {
        let whole = x.wholeChat
        var s = whole
            ? "The user's messages in this chat:\n\(clip(x.userText, 5000))\n\nThe replies:\n\(clip(x.replyText, 6000))"
            : "User's message:\n\(clip(x.userText, 3000))\n\nReply:\n\(clip(x.replyText, 2000))"
        if x.webAllowed, let w = x.webEvidence {
            s += "\n\nWeb results the reply used (from \(x.webSources.joined(separator: ", "))):\n\(clip(w, whole ? 10000 : 5000))"
        }
        if x.filesAllowed {
            // share the room between the documents, so each one is seen
            let room = max(800, (whole ? 12000 : 9000) / max(1, x.files.count))
            s += "\n\nDocuments the reply read:" + x.files.map { "\n\n[\($0.path)]\n" + clip($0.text, room) }.joined()
        }
        return s
    }

    /// The facts in the model's output: the first JSON array in it (text or a code fence around it is fine).
    /// The facts in the model's output: a JSON array, or one or more bare objects (Zoro-1 often writes a single
    /// object instead of an array), with any text or code fence around them.
    public static func parse(_ text: String) -> [RawFact] {
        let text = repairJSON(text)
        var dicts: [[String: Any]] = []
        if let lo = text.firstIndex(of: "["), let hi = text.lastIndex(of: "]"), lo < hi,
           let data = String(text[lo...hi]).data(using: .utf8),
           let arr = (try? JSONSerialization.jsonObject(with: data)) as? [[String: Any]] {
            dicts = arr
        } else {
            dicts = objects(in: text)
        }
        return dicts.compactMap { d -> RawFact? in
            func str(_ k: String) -> String? {
                guard let v = d[k] as? String else { return nil }
                let t = v.trimmingCharacters(in: .whitespacesAndNewlines)
                return t.isEmpty ? nil : t
            }
            guard let s = str("subject"), let r = str("relation"), let o = str("object") else { return nil }
            return RawFact(subject: s, subjectType: nodeType(str("subject_type")),
                           relation: r.lowercased(), object: o,
                           objectType: nodeType(str("object_type")),
                           source: ["web", "file"].contains(str("source")?.lowercased() ?? "") ? str("source")!.lowercased() : "user",
                           quote: str("quote"), note: str("note"))
        }
    }

    /// Zoro-1's JSON with the two mistakes it makes in quotes fixed: a `"` inside a string (`"quote": "it is
    /// "the" one"`) and a raw line break. A `"` ends a string only when what follows it is `,` `}` `]` `:` or the end.
    public static func repairJSON(_ text: String) -> String {
        let chars = Array(text.unicodeScalars)
        var out = String.UnicodeScalarView()
        let space: Set<Unicode.Scalar> = [" ", "\t", "\r", "\n"], after: Set<Unicode.Scalar> = [",", "}", "]", ":"]
        var inString = false, escaped = false
        for (i, ch) in chars.enumerated() {
            if inString {
                if escaped { escaped = false } else if ch == "\\" { escaped = true } else if ch == "\"" {
                    var j = i + 1
                    while j < chars.count, space.contains(chars[j]) { j += 1 }
                    if j == chars.count || after.contains(chars[j]) {
                        inString = false
                    } else {
                        out.append(contentsOf: "\\\"".unicodeScalars)        // a quote mark inside the text
                        continue
                    }
                } else if ch == "\n" || ch == "\r" {
                    out.append(contentsOf: "\\n".unicodeScalars)
                    continue
                }
            } else if ch == "\"" {
                inString = true
            }
            out.append(ch)
        }
        return String(out)
    }

    /// Every balanced top-level {...} in `text` that parses as a JSON object.
    static func objects(in text: String) -> [[String: Any]] {
        var out: [[String: Any]] = []
        var depth = 0, start: String.Index?, inString = false, escaped = false
        for i in text.indices {
            let ch = text[i]
            if inString {
                if escaped { escaped = false } else if ch == "\\" { escaped = true } else if ch == "\"" { inString = false }
                continue
            }
            if ch == "\"" { inString = true } else if ch == "{" {
                if depth == 0 { start = i }
                depth += 1
            } else if ch == "}", depth > 0 {
                depth -= 1
                if depth == 0, let s = start,
                   let obj = (try? JSONSerialization.jsonObject(with: Data(text[s...i].utf8))) as? [String: Any] {
                    out.append(obj)
                }
            }
        }
        return out
    }

    /// A type name from the model, with common synonyms ("organization", "company", "location", "date").
    static func nodeType(_ s: String?) -> NodeType {
        let t = (s ?? "").lowercased().trimmingCharacters(in: .whitespaces)
        if let n = NodeType(rawValue: t) { return n }
        switch t {
        case "organization", "organisation", "company", "team", "institution", "group": return .org
        case "location", "city", "country": return .place
        case "user", "self": return .me
        case "date", "number", "version", "time", "literal", "string": return .value
        case "job", "title", "job title", "position", "occupation": return .role
        case "product", "tool", "language", "software", "model", "concept", "field", "skill": return .topic
        case "document", "folder", "path": return .file
        default: return .topic
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
        guard let key = Relation.key(for: r.relation) else { return nil }      // an existing or a new relation
        r.relation = key
        // a quote can span lines on the page (normalize ignores the line breaks when it is looked up)
        r.quote = r.quote.map { $0.split(whereSeparator: \.isNewline).joined(separator: " ") }
        r.note = r.note.map { $0.split(whereSeparator: \.isNewline).joined(separator: " ") }
        let texts = [r.subject, r.object, r.quote ?? "", r.note ?? ""]
        guard texts.allSatisfy({ !$0.contains("\n") && $0.count <= 240 }), r.subject.count <= 80, r.object.count <= 80,
              !texts.contains(where: looksSecret) else { return nil }
        let subjMe = isMe(r.subject, r.subjectType), objMe = isMe(r.object, r.objectType)
        if subjMe { r.subject = KnowledgeGraph.meName; r.subjectType = .me }
        if objMe { r.object = KnowledgeGraph.meName; r.objectType = .me }
        guard KnowledgeGraph.normalize(r.subject) != KnowledgeGraph.normalize(r.object) else { return nil }
        if r.source == "web" && !x.webAllowed { r.source = "user" }    // mislabeled: no web results this turn
        if r.source == "file" && !x.filesAllowed { r.source = "user" }
        // a "web" quote that is really in a document read in the chat (or the other way round) is that source
        if let q = r.quote, q.count >= 8, r.source != "user" {
            let inWeb = x.webAllowed && x.webEvidence.map { KnowledgeGraph.normalize($0).contains(KnowledgeGraph.normalize(q)) } == true
            let inFile = x.file(quoting: q) != nil
            if r.source == "web" && !inWeb && inFile { r.source = "file" }
            if r.source == "file" && !inFile && inWeb { r.source = "web" }
        }
        if r.source == "file" {
            // a document can never say anything about the user, and the quote must really be in one
            guard !subjMe, !objMe, let q = r.quote, q.count >= 8, x.file(quoting: q) != nil else { return nil }
            return r
        }
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

    // MARK: - relations

    /// At most this many new relations wait to be approved; after that, new ones become "related to" with a note.
    public static let maxProvisional = 40

    public static let functionalQuestion = "Can one subject have only one of these at a time? For example, a person "
        + "works at one company at a time, but can know many people."
    public static let timeQuestion = "Does this relation change on its own over time, the way who holds an office or "
        + "a product's latest version does?"

    /// The relation of a checked fact: an existing one (by key or name, or the one Zoro-1 picks as meaning the same),
    /// or a new provisional relation whose two flags Zoro-1 decides.
    static func resolveRelation(_ c: RawFact, _ x: Exchange, graph: KnowledgeGraph, backend: KnowledgeBackend)
        async throws -> (key: String, new: Relation?) {
        if graph.relation(c.relation) != nil { return (c.relation, nil) }
        if let r = graph.relations.first(where: { Relation.key(for: $0.label) == c.relation }) { return (r.key, nil) }
        let label = c.relation.replacingOccurrences(of: "_", with: " ")
        let example = "\(c.subject) —\(label)→ \(c.object)"
        let all = graph.relations
        var options: [(key: String, label: String)] = all.enumerated().map { i, r in
            (key: String(i + 1), label: "\(r.label): \(r.hint)")
        }
        options.append((key: "new", label: "none of these: \"\(label)\" is a different relation"))
        let ps = try await backend.choose(
            state: [("fact", example), ("relation", label)],
            question: "Which existing relation means the same as \"\(label)\" in this fact?", options: options)
        if let best = ps.filter({ $0.key != "new" }).max(by: { $0.value < $1.value }), best.value >= linkThreshold,
           let i = Int(best.key), all.indices.contains(i - 1) {
            return (all[i - 1].key, nil)
        }
        if graph.customRelations.filter(\.provisional).count >= maxProvisional { return ("related_to", nil) }
        let state = [("relation", label), ("example", example)]
        let one = try await backend.yesNo(state: state, question: functionalQuestion)
        let changes = try await backend.yesNo(state: state, question: timeQuestion)
        return (c.relation, Relation(key: c.relation, label: label, functional: one >= 0.5, timeSensitive: changes >= 0.5,
                                     hint: "\(c.subjectType.rawValue) → \(c.objectType.rawValue)", custom: true,
                                     provisional: true, created: x.date))
    }

    // MARK: - entity linking

    /// An end of a fact as a node: the user, an exact name match, a candidate Zoro-1 picks, or a new node.
    static func link(_ name: String, type: NodeType, _ x: Exchange, fact: String = "", graph: KnowledgeGraph,
                     backend: KnowledgeBackend) async throws -> ProposedFact.End {
        if type == .me { return .me }
        if let exact = graph.node(named: name, type: type), exact.type != .me { return .existing(exact.id) }
        if type == .value { return .new(name: name, type: type) }
        let cands = graph.similar(to: name)
        guard !cands.isEmpty else {
            return .new(name: name, type: try await typeOfNew(name, guess: type, fact: fact, backend: backend))
        }
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
        return .new(name: name, type: try await typeOfNew(name, guess: type, fact: fact, backend: backend))
    }

    /// The type of a new entry, as Zoro-1 decides it (the proposal's type is a guess: "Member of Technical Staff"
    /// came as a team). Kept as guessed when the decision is unsure.
    public static let typeThreshold = 0.5

    static let typeOptions: [(key: String, label: String)] = [
        ("person", "a person"),
        ("org", "an organization, company, team or group"),
        ("role", "a job title or position (researcher, engineer, intern, Member of Technical Staff)"),
        ("project", "a project, product or model someone makes"),
        ("place", "a place: a country, city or address"),
        ("file", "a file or folder"),
        ("topic", "a subject, field, tool, language or idea"),
        ("event", "an event"),
        ("value", "a literal: a date, a number or a version"),
    ]

    static func typeOfNew(_ name: String, guess: NodeType, fact: String, backend: KnowledgeBackend) async throws
        -> NodeType {
        let ps = try await backend.choose(state: [("name", name), ("fact", fact)],
                                          question: "What kind of thing is \"\(name)\" in this fact?",
                                          options: typeOptions)
        guard let best = ps.max(by: { $0.value < $1.value }), best.value >= typeThreshold,
              let t = NodeType(rawValue: best.key) else { return guess }
        return t
    }

    /// For a fact whose types fit both ways (two organizations "member of" each other, a project "part of" a
    /// project): whether the proposal has it the right way round.
    static func rightWayRound(_ c: RawFact, _ rel: String, graph: KnowledgeGraph, backend: KnowledgeBackend)
        async throws -> Bool {
        let r = graph.relation(rel)
        let label = r?.label ?? rel.replacingOccurrences(of: "_", with: " ")
        let p = try await backend.yesNo(
            state: [("fact", "\(c.subject) —\(label)→ \(c.object)"), ("relation", "\(label): \(r?.hint ?? "")"),
                    ("evidence", c.quote ?? c.note ?? "")],
            question: "Does this fact go the right way round: is \"\(c.subject)\" the \(label) side, as the "
                + "relation reads from left to right?")
        return p >= 0.5
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
            var madeRelation = false
            if let r = f.newRelation, relation(r.key) == nil {
                addRelation(r)
                madeRelation = true
            }
            var e = KGEdge(subject: s, relation: f.relation, object: o,
                           origin: f.fromWeb ? .web : f.fromFile ? .file : .user, learnedAt: x.date)
            e.note = f.note
            e.chatID = x.chatID
            e.messageID = x.messageID
            e.validFrom = x.date
            e.confidence = f.confidence
            e.chatTitle = x.chatTitle
            e.model = x.model
            e.via = x.wholeChat ? "update-kg" : x.explicit ? "remember" : "learned"
            e.userText = String(x.userText.prefix(300))
            if f.fromWeb {
                e.sourceURL = x.webSources.first
                e.asOf = x.date
                e.quote = f.quote
            }
            if f.fromFile {
                e.sourceURL = f.quote.flatMap { x.file(quoting: $0)?.path } ?? x.files.first?.path
                e.asOf = x.date
                e.quote = f.quote
            }
            guard let added = addFact(e, at: x.date) else {
                if madeRelation { customRelations.removeAll { $0.key == f.relation } }
                continue
            }
            rec.edges.append(added.id)
            rec.ended += added.ended
            if madeRelation { rec.relations.append(f.relation) }
            promoteIfUsed(f.relation)
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
