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

    let x = Exchange(chatID: "c", messageID: "m", userText: "I work at Trillion Labs now", replyText: "Nice!")
    check(Learner.check(raw[0], x) != nil, "a fact about the user is kept")
    check(Learner.check(raw[1], x) == nil, "an unknown relation is dropped")

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

print(failures == 0 ? "kg-selftest: all checks passed" : "kg-selftest: \(failures) check(s) failed")
exit(failures == 0 ? 0 : 1)
