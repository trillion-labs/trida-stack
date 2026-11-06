// Checks for the knowledge graph and the learning pipeline, runnable with only the Command Line Tools
// (no XCTest): `swift run kg-selftest` in inference/mlx/app/Knowledge. Exits 1 on the first failed group.
import Foundation
import TridaKnowledge

var failures = 0
func check(_ ok: Bool, _ what: String, file: String = #file, line: Int = #line) {
    if !ok {
        failures += 1
        print("FAIL \(what) (line \(line))")
    }
}

/// A backend that answers from fixed tables, so the pipeline runs without a server.
struct FakeBackend: KnowledgeBackend {
    var gate = 0.95
    var proposal = "[]"
    var linkPick: String? = nil

    func yesNo(state: [(String, String)], question: String) async throws -> Double { gate }

    func choose(state: [(String, String)], question: String, options: [(key: String, label: String)]) async throws
        -> [String: Double] {
        var out = Dictionary(uniqueKeysWithValues: options.map { ($0.key, 0.05) })
        out[linkPick ?? "new"] = 0.9
        return out
    }

    func complete(system: String, user: String, maxTokens: Int) async throws -> String { proposal }
}

/// Like FakeBackend, but types new entries by name (the "What kind of thing is …" choice).
struct TypingBackend: KnowledgeBackend {
    var proposal: String
    var types: [String: String]

    func yesNo(state: [(String, String)], question: String) async throws -> Double { 0.95 }

    func choose(state: [(String, String)], question: String, options: [(key: String, label: String)]) async throws
        -> [String: Double] {
        var out = Dictionary(uniqueKeysWithValues: options.map { ($0.key, 0.05) })
        if let name = state.first(where: { $0.0 == "name" })?.1, let t = types[name] { out[t] = 0.9 } else { out["new"] = 0.9 }
        return out
    }

    func complete(system: String, user: String, maxTokens: Int) async throws -> String { proposal }
}

let day = 86400.0
let t0 = Date(timeIntervalSince1970: 1_790_000_000)

// MARK: graph basics

do {
    var g = KnowledgeGraph()
    let me = g.ensureMe()
    let korea = g.addNode(name: "South Korea", type: .place)
    g.addAlias("대한민국", to: korea)
    g.addAlias("Korea", to: korea)
    let yoon = g.addNode(name: "Yoon Suk Yeol", type: .person)
    let lee = g.addNode(name: "Lee Jae-myung", type: .person)

    var e1 = KGEdge(subject: korea, relation: "president", object: yoon, origin: .web, learnedAt: t0)
    e1.asOf = t0
    check(g.addFact(e1, at: t0) != nil, "first fact is added")
    check(g.addFact(KGEdge(subject: korea, relation: "president", object: yoon, origin: .web), at: t0) == nil,
          "the same current fact is not added twice")

    // a functional relation: the new president ends the old fact, which is kept
    let e2 = KGEdge(subject: korea, relation: "president", object: lee, origin: .user, learnedAt: t0 + day)
    let added = g.addFact(e2, at: t0 + day)
    check(added?.ended == [e1.id], "the newer president ends the older fact")
    check(g.edges.first { $0.id == e1.id }?.isCurrent(at: t0 + 2 * day) == false, "the old fact has an end date")
    check(g.edges.count == 2, "ended facts are kept")

    // a web fact cannot replace one the user stated
    let e3 = KGEdge(subject: korea, relation: "president", object: yoon, origin: .web, learnedAt: t0 + 2 * day)
    check(g.addFact(e3, at: t0 + 2 * day) == nil, "the user's fact outranks a web fact")

    // lookups: aliases, Hangul inside words, normalization
    check(g.node(named: "korea")?.id == korea, "alias lookup is case-insensitive")
    check(g.mentioned(in: "대한민국의 대통령은 누구야?").map(\.id) == [korea], "Hangul names match with particles attached")
    check(g.mentioned(in: "Koreatown is nice").isEmpty, "Latin names match whole words only")
    check(g.similar(to: "Lee Jae myung").first?.id == lee, "similar names are candidates")

    // prompt block: the current fact, with its provenance
    let block = g.contextBlock(for: "who is the president of Korea", now: t0 + 3 * day) ?? ""
    check(block.contains("Lee Jae-myung") && !block.contains("Yoon"), "the block has only current facts")
    check(block.contains("you said"), "the block says where the fact comes from")
    let later = g.contextBlock(for: "Korea", now: t0 + 60 * day) ?? ""
    check(later.contains("may have changed"), "old time-sensitive facts are flagged")

    // merge: a duplicate entity
    let dup = g.addNode(name: "Republic of Korea", type: .place)
    let seoul = g.addNode(name: "Seoul", type: .place)
    _ = g.addFact(KGEdge(subject: dup, relation: "capital", object: seoul, origin: .user), at: t0)
    g.merge(dup, into: korea)
    check(g.node(dup) == nil && g.node(named: "Republic of Korea")?.id == korea, "merge moves names")
    check(g.currentEdges(touching: [korea]).contains { $0.relation == "capital" }, "merge moves facts")

    // delete
    g.deleteNode(yoon)
    check(!g.edges.contains { $0.object == yoon }, "deleting an entry deletes its facts")
    g.deleteNode(me)
    check(g.me != nil, "the user's own entry cannot be deleted")
    check(g.search("Seoul").contains("capital"), "search finds facts by name")
}

// MARK: proposal parsing and checks

do {
    let raw = Learner.parse("""
        Here you go:
        ```json
        [{"subject": "me", "subject_type": "me", "relation": "works_at", "object": "Trillion Labs", "object_type": "org", "source": "user"},
         {"subject": "x", "relation": "flies_to", "object": "y"}]
        ```
        """)
    check(raw.count == 2 && raw[0].object == "Trillion Labs", "JSON inside a code fence is parsed")

    // what Zoro-1 actually wrote: one bare object, not an array, with "organization" as the type
    let bare = Learner.parse(#"{"subject": "user", "subject_type": "person", "relation": "works_at", "object": "Trillionlabs", "object_type": "organization", "source": "user"} {"subject": "me", "relation": "uses", "object": "Python"}"#)
    check(bare.count == 2 && bare[0].objectType == .org, "bare objects and type synonyms are parsed")
    let said = Exchange(chatID: "c", messageID: "b", userText: "내 이름은 홍길동이야. Trillionlabs에서 일하고 있어", replyText: "반가워요")
    check(Learner.check(bare[0], said)?.subject == KnowledgeGraph.meName, "\"user\" is the user's own entry")

    let x = Exchange(chatID: "c", messageID: "m", userText: "I work at Trillion Labs now", replyText: "Nice!")
    check(Learner.check(raw[0], x) != nil, "a fact about the user is kept")
    check(Learner.check(raw[1], x) == nil, "a fact naming nothing the user wrote is dropped")
    var odd = raw[0]
    odd.relation = "is the person who happens to work at"
    check(Learner.check(odd, x) == nil, "a relation longer than 4 words is dropped")

    var secret = raw[0]
    secret.object = "my password is hunter2"
    check(Learner.check(secret, x) == nil, "secrets are dropped")

    var unsaid = raw[0]
    unsaid.subject = "Elon Musk"
    unsaid.subjectType = .person
    unsaid.object = "SpaceX"
    check(Learner.check(unsaid, x) == nil, "a user fact must be about the user or name what the user wrote")

    // web facts: only with a grounded reply, only with a quote that is in the evidence, never about the user
    let web = RawFact(subject: "United States", subjectType: .place, relation: "president", object: "Donald Trump",
                      objectType: .person, source: "web", quote: "Incumbent Donald Trump since January 20, 2025")
    let evidence = "President of the United States\nIncumbent\nDonald Trump\nsince January 20, 2025\n"
    let wx = Exchange(chatID: "c", messageID: "w", userText: "지금 미국의 대통령은 누구야", replyText: "도널드 트럼프입니다.",
                      webEvidence: evidence, webSources: ["https://en.wikipedia.org/wiki/President"], webGrounded: true)
    check(Learner.check(web, wx) != nil, "a grounded web fact with a real quote is kept")
    var ungrounded = wx
    ungrounded.webGrounded = false
    check(Learner.check(web, ungrounded) == nil, "no web facts when the reply was not backed by the pages")
    var madeUp = web
    madeUp.quote = "Incumbent Joe Biden since January 20, 2021"
    check(Learner.check(madeUp, wx) == nil, "a quote that is not in the page is dropped")
    var aboutMe = web
    aboutMe.subject = "me"
    aboutMe.subjectType = .me
    check(Learner.check(aboutMe, wx) == nil, "a page cannot say anything about the user")
}

// MARK: the pipeline end to end, and undo

// top-level await (main.swift): no Task, so nothing waits on a blocked main thread
do {
    var g = KnowledgeGraph()
    let x = Exchange(chatID: "c1", messageID: "m1", userText: "참고로 나는 Trillion Labs에서 일하고 Zoro-1 만들고 있어",
                     replyText: "알겠습니다!", date: t0)
    let backend = FakeBackend(proposal: """
        [{"subject": "me", "subject_type": "me", "relation": "works_at", "object": "Trillion Labs", "object_type": "org", "source": "user"},
         {"subject": "Zoro-1", "subject_type": "project", "relation": "created_by", "object": "Trillion Labs", "object_type": "org", "source": "user"}]
        """)
    do {
        let facts = try await Learner.propose(x, graph: g, backend: backend)
        check(facts.count == 2, "two facts proposed")
        let rec = g.apply(facts, x)
        check(rec?.edges.count == 2, "two facts applied")
        check(g.nodes.filter { $0.name == "Trillion Labs" }.count == 1, "the second fact reuses the node the first made")

        // the gate keeps small talk out
        let quiet = FakeBackend(gate: 0.3, proposal: backend.proposal)
        let none = try await Learner.propose(Exchange(chatID: "c1", messageID: "m2", userText: "ㅋㅋ 고마워",
                                                      replyText: "천만에요"), graph: g, backend: quiet)
        check(none.isEmpty, "the gate keeps small talk out")

        // linking: "Trillion" is the same as the existing "Trillion Labs" (Zoro-1 picks it)
        let tl = g.node(named: "Trillion Labs")!.id
        let link = FakeBackend(proposal: """
            [{"subject": "Trillion", "subject_type": "org", "relation": "located_in", "object": "Seoul", "object_type": "place", "source": "user"}]
            """, linkPick: "1")
        let x3 = Exchange(chatID: "c1", messageID: "m3", userText: "Trillion 사무실은 Seoul에 있어", replyText: "네", date: t0)
        g.apply(try await Learner.propose(x3, graph: g, backend: link), x3)
        check(g.currentEdges(touching: [tl]).contains { $0.relation == "located_in" }, "entity linking picks the existing node")
        check(g.node(named: "Trillion")?.id == nil || g.node(named: "Trillion")?.id == tl, "no duplicate node is made")

        // undo takes back exactly what one exchange added
        let before = g.edges.count
        g.undo(record: "m3")
        check(g.edges.count == before - 1 && g.node(named: "Seoul") == nil, "undo removes the fact and its new node")
        check(g.node(named: "Trillion Labs") != nil, "undo keeps nodes other facts use")
    } catch {
        check(false, "pipeline threw \(error)")
    }
}

// MARK: facts from documents the reply read

do {
    let blog = FileText(path: "/Users/me/blog/posts/zoro-mlx.md",
                        text: "# Running Zoro-1 on a Mac\nZoro-1 runs on Apple Silicon with MLX, using self-speculative decoding.")
    let x = Exchange(chatID: "c", messageID: "d1", userText: "블로그 글들 요약해줘", replyText: "요약입니다…", files: [blog])
    let fact = RawFact(subject: "Zoro-1", subjectType: .project, relation: "uses", object: "MLX", objectType: .topic,
                       source: "file", quote: "Zoro-1 runs on Apple Silicon with MLX")
    check(Learner.check(fact, x) != nil, "a document fact with a real quote is kept")
    var made = fact
    made.quote = "Zoro-1 runs on CUDA only"
    check(Learner.check(made, x) == nil, "a quote that is not in the document is dropped")
    var aboutMe = fact
    aboutMe.subject = "me"
    aboutMe.subjectType = .me
    check(Learner.check(aboutMe, x) == nil, "a document cannot say anything about the user")
    var g = KnowledgeGraph()
    let f = ProposedFact(subject: .new(name: "Zoro-1", type: .project), relation: "uses", object: .new(name: "MLX", type: .topic),
                         quote: fact.quote)
    var ff = f
    ff.fromFile = true
    g.apply([ff], x)
    check(g.edges.first?.origin == .file && g.edges.first?.sourceURL == blog.path, "the fact records its document")
    check(g.provenance(g.edges[0]).hasPrefix("from zoro-mlx.md"), "provenance names the document")
}

// MARK: provenance and history

do {
    var g = KnowledgeGraph()
    var x = Exchange(chatID: "chat-1", messageID: "msg-1", userText: "나는 서울에 살아", replyText: "네", date: t0)
    x.chatTitle = "Moving"
    x.model = "zoro-1-mlx-q8"
    var f = ProposedFact(subject: .me, relation: "lives_in", object: .new(name: "Seoul", type: .place))
    f.confidence = 0.83
    g.apply([f], x)
    let e = g.edges[0]
    check(e.learnedAt == t0 && e.chatID == "chat-1" && e.messageID == "msg-1" && e.chatTitle == "Moving"
          && e.model == "zoro-1-mlx-q8" && e.via == "learned" && e.userText == "나는 서울에 살아" && e.confidence == 0.83,
          "a fact records when, where, by which model, how, from what and how sure")
    // a newer fact ends it, and both sides of the change are in its history
    var x2 = x
    x2.messageID = "msg-2"
    x2.userText = "이제 부산에 살아"
    x2.date = t0 + day
    g.apply([ProposedFact(subject: .me, relation: "lives_in", object: .new(name: "Busan", type: .place))], x2)
    let old = g.edges.first { $0.id == e.id }!
    check(old.history?.last?.what == "replaced" && old.history?.last?.from == "Seoul" && old.history?.last?.to == "Busan",
          "the replaced fact's history says what replaced it")
    g.setEnded(old.id, at: nil)
    check(g.edges.first { $0.id == e.id }?.history?.last?.what == "reopened", "edits are logged")
}

// MARK: relations added while learning

do {
    // an old file without customRelations still loads
    let old = #"{"nodes": [], "edges": [], "records": {"m": {"edges": ["e"], "nodes": [], "ended": []}}}"#
    let dec = JSONDecoder()
    dec.dateDecodingStrategy = .iso8601
    let loaded = try? dec.decode(KnowledgeGraph.self, from: Data(old.utf8))
    check(loaded?.customRelations.isEmpty == true && loaded?.records["m"]?.relations.isEmpty == true,
          "a file from before custom relations still loads")
    check(Relation.key(for: "Reports To") == "reports_to" && Relation.key(for: "a b c d e") == nil,
          "relation names become snake_case keys of 1-4 words")

    // a new relation: Zoro-1 finds no existing one that means the same, then decides its two flags
    var g = KnowledgeGraph()
    let x = Exchange(chatID: "c", messageID: "r1", userText: "나는 김팀장한테 보고해", replyText: "알겠습니다", date: t0)
    let backend = FakeBackend(proposal: """
        [{"subject": "me", "subject_type": "me", "relation": "reports_to", "object": "김팀장", "object_type": "person", "source": "user"}]
        """)
    let facts = try await Learner.propose(x, graph: g, backend: backend)
    check(facts.first?.newRelation?.key == "reports_to" && facts.first?.newRelation?.provisional == true,
          "an unlisted relation is proposed as a new provisional one")
    g.apply(facts, x)
    check(g.relation("reports_to")?.custom == true && g.edges.count == 1, "the new relation and its fact are added")
    check(g.records["r1"]?.relations == ["reports_to"], "the record remembers the relation it made")
    check(Learner.proposalSystem(g).contains("reports_to"), "later proposals can reuse it")

    // a synonym maps onto the existing relation when Zoro-1 picks it
    let synonym = FakeBackend(proposal: """
        [{"subject": "me", "subject_type": "me", "relation": "employed_by", "object": "Trillion Labs", "object_type": "org", "source": "user"}]
        """, linkPick: String((g.relations.firstIndex { $0.key == "works_at" } ?? 0) + 1))
    let x2 = Exchange(chatID: "c", messageID: "r2", userText: "Trillion Labs에 다녀", replyText: "네", date: t0)
    let f2 = try await Learner.propose(x2, graph: g, backend: synonym)
    check(f2.first?.relation == "works_at" && f2.first?.newRelation == nil, "a synonym is mapped to the existing relation")

    // promotion after enough facts use it; merge; delete keeps the facts as "related to"
    for i in 0..<3 {
        let p = g.addNode(name: "P\(i)", type: .person)
        _ = g.addFact(KGEdge(subject: p, relation: "reports_to", object: g.me!.id, origin: .user), at: t0)
        g.promoteIfUsedForTests("reports_to")
    }
    check(g.relation("reports_to")?.provisional == false, "a relation used enough is no longer provisional")
    g.addRelation(Relation(key: "manager", label: "manager", functional: true, timeSensitive: false, hint: "",
                           provisional: true))
    let boss = g.addNode(name: "Boss", type: .person)
    _ = g.addFact(KGEdge(subject: g.me!.id, relation: "manager", object: boss, origin: .user), at: t0)
    g.mergeRelation("manager", into: "reports_to")
    check(g.relation("manager") == nil && g.edges.contains { $0.relation == "reports_to" && $0.object == boss },
          "merging moves the facts")
    let n = g.edges.filter { $0.relation == "reports_to" }.count
    g.deleteRelation("reports_to")
    check(g.relation("reports_to") == nil && g.edges.filter { $0.relation == "related_to" && $0.note == "reports to" }.count == n,
          "deleting keeps the facts as related to, with a note")
    check(g.relation("works_at") != nil, "built-in relations stay")
} catch {
    check(false, "relations threw \(error)")
}

// MARK: relation types and direction

do {
    check(!Relations.fits("member_of", .org, .role) && !Relations.fits("member_of", .role, .org),
          "a job title is not a member, and nothing is a member of one")
    check(!Relations.fits("created_by", .org, .value), "created by a date does not fit")
    check(Relations.fits("works_at", .me, .org) && Relations.fits("related_to", .role, .value), "fits checks types")

    // the hiring section of a company page: "Member of Technical Staff" is a role (Zoro-1 types new entries)
    let x = Exchange(chatID: "c", messageID: "jobs", userText: "Trillion Labs에 대해 알려줘", replyText: "…",
                     webEvidence: "Join our team\nMember of Technical Staff\nTrillion Labs is based in Seoul, Korea",
                     webSources: ["https://trillionlabs.co"], webGrounded: true)
    let proposal = """
        [{"subject": "Trillion Labs", "subject_type": "org", "relation": "member_of", "object": "Technical Staff", "object_type": "team", "source": "web", "quote": "Member of Technical Staff"},
         {"subject": "Seoul", "subject_type": "place", "relation": "located_in", "object": "Trillion Labs", "object_type": "org", "source": "web", "quote": "Trillion Labs is based in Seoul, Korea"}]
        """
    var g = KnowledgeGraph()
    let types = ["Technical Staff": "role", "Trillion Labs": "org", "Seoul": "place"]
    let facts = try await Learner.propose(x, graph: g, backend: TypingBackend(proposal: proposal, types: types))
    g.apply(facts, x)
    check(!g.edges.contains { $0.relation == "member_of" }, "a company is not a member of a job title")
    let loc = g.edges.first { $0.relation == "located_in" }
    check(loc.map { g.node($0.subject)?.name == "Trillion Labs" && g.node($0.object)?.name == "Seoul" } == true,
          "a fact that only fits the other way round is turned around")
}

// MARK: /update-kg (the whole chat)

do {
    // an earlier turn read the page; the last turn only listed facts (the gate would have said no)
    var x = Exchange(chatID: "c", messageID: "last", userText: "Trillion Labs에 대해 알려줘\n\ncan you list up triples?",
                     replyText: "Trillion Labs → located in → Seoul", webEvidence: "Trillion Labs is based in Seoul, Korea",
                     webSources: ["https://trillionlabs.co"], webGrounded: true, explicit: true)
    x.wholeChat = true
    let proposal = """
        [{"subject": "Trillion Labs", "subject_type": "org", "relation": "located_in", "object": "Seoul", "object_type": "place", "source": "web", "quote": "Trillion Labs is based in Seoul, Korea"},
         {"subject": "Trillion Labs", "subject_type": "org", "relation": "ceo", "object": "Someone", "object_type": "person", "source": "web", "quote": "a line that is not on the page"}]
        """
    var g = KnowledgeGraph()
    let facts = try await Learner.propose(x, graph: g, backend: FakeBackend(gate: 0.1, proposal: proposal))
    g.apply(facts, x)
    check(g.edges.count == 1 && g.edges.first?.relation == "located_in", "/update-kg skips the gate but still checks quotes")
    check(g.edges.first?.via == "update-kg", "facts from /update-kg say so")
    check(Learner.proposalUser(x).contains("The user's messages in this chat"), "the whole chat goes to the proposal")
}

// MARK: repairing Zoro-1's JSON

do {
    let broken = #"[{"subject": "A Co", "relation": "is_a", "object": "lab", "source": "web", "quote": "Our mission is "recursive" AI"}, {"subject": "B", "relation": "uses", "object": "C", "quote": "line one"# + "\n" + #"line two"}]"#
    let facts = Learner.parse(broken)
    check(facts.count == 2, "a quote mark or a line break inside a string does not lose the facts after it")
    check(facts.first?.quote == "Our mission is \"recursive\" AI", "the quote keeps its quote marks")
    let fine = #"[{"subject": "x", "relation": "uses", "object": "y \"z\"", "source": "user"}]"#
    check(Learner.parse(fine).first?.object == "y \"z\"", "valid JSON is unchanged")
}

// MARK: whole-graph layout

do {
    var g = KnowledgeGraph()
    let a = g.addNode(name: "A", type: .person), b = g.addNode(name: "B", type: .org)
    let c = g.addNode(name: "C", type: .place), d = g.addNode(name: "D", type: .topic)
    let lone = g.addNode(name: "Lone", type: .topic)
    _ = g.addFact(KGEdge(subject: a, relation: "works_at", object: b, origin: .user), at: t0)
    _ = g.addFact(KGEdge(subject: c, relation: "related_to", object: d, origin: .user), at: t0)
    let pos = KnowledgeLayout.layout(g)
    check(pos.count == g.nodes.count, "every entry gets a position, isolated ones too")
    func dist(_ x: String, _ y: String) -> Double {
        guard let p = pos[x], let q = pos[y] else { return -1 }
        return ((p.x - q.x) * (p.x - q.x) + (p.y - q.y) * (p.y - q.y)).squareRoot()
    }
    check(dist(a, b) > 20 && dist(c, d) > 20, "linked entries do not sit on top of each other")
    check([dist(a, c), dist(a, d), dist(b, c), dist(b, d), dist(a, lone), dist(c, lone)].allSatisfy { $0 > 40 },
          "separate groups are kept apart")
    check(KnowledgeLayout.layout(g) == pos, "the layout is deterministic")
}

print(failures == 0 ? "kg-selftest: all checks passed" : "kg-selftest: \(failures) check(s) failed")
exit(failures == 0 ? 0 : 1)
