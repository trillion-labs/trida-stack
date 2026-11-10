import Combine
import Foundation
import TridaKnowledge

/// The knowledge graph as the app uses it: loaded from and saved to one JSON file, every change undoable, learning
/// in the background, and the Knowledge window's view state (kept here because SwiftUI's @State is not available
/// with only the Command Line Tools installed).
@MainActor
public final class KnowledgeModel: ObservableObject {
    @Published public private(set) var graph: KnowledgeGraph
    /// Learn from chats automatically (Settings).
    @Published public var enabled: Bool {
        didSet { UserDefaults.standard.set(enabled, forKey: "TridaKnowledge.enabled") }
    }
    /// The exchange currently being learned from (the chip shows "Learning…").
    @Published public private(set) var learning: String?
    @Published public var lastError: String?
    /// The last time learning added facts, and how many (the brain's pop and "+N").
    @Published public private(set) var lastLearned: LearnedPulse?

    public struct LearnedPulse: Equatable {
        public var count: Int
        public var at: Date
    }

    // the Knowledge window
    @Published public var search = ""
    @Published public var typeFilter: NodeType?
    @Published public var focus: String?
    @Published public var selectedNode: String?
    @Published public var selectedEdge: String?
    @Published public var showEnded = true
    @Published public var renameText = ""
    @Published public var aliasText = ""
    @Published public var newRelation = "related_to"
    @Published public var newObject = ""
    @Published public var newObjectType: NodeType = .topic
    @Published public var newObjectIsSubject = false
    // the whole-graph view: zoom and pan (and the layout, computed once per graph shape)
    @Published public var wholeGraph = false
    @Published public var zoom: CGFloat = 1
    @Published public var pan: CGSize = .zero
    var zoomBase: CGFloat?
    var panBase: CGSize?
    private var layoutKey = ""
    private var layoutCache: [String: KnowledgeLayout.Point] = [:]

    /// Node positions for the whole-graph view; recomputed only when entries or links change.
    public func layout() -> [String: KnowledgeLayout.Point] {
        let key = graph.nodes.map(\.id).joined(separator: ",") + "|"
            + graph.edges.filter { showEnded || $0.isCurrent() }.map { $0.subject + ">" + $0.object }.joined(separator: ",")
        if key != layoutKey {
            layoutCache = KnowledgeLayout.layout(graph, includeEnded: showEnded)
            layoutKey = key
        }
        return layoutCache
    }

    public func resetView() {
        zoom = 1
        pan = .zero
    }

    // the Relations tab
    @Published public var showRelations = false
    @Published public var selectedRelation: String?
    @Published public var relationText = ""

    /// Opens the chat message a fact came from (set by the app).
    public var openSource: ((_ chatID: String, _ messageID: String) -> Void)?

    public let file: URL
    private var undoStack: [KnowledgeGraph] = []
    private var saveTask: Task<Void, Never>?

    nonisolated public static var defaultFile: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Zoro-1 Chat/knowledge.json")
    }

    public init(file: URL = KnowledgeModel.defaultFile) {
        self.file = file
        enabled = UserDefaults.standard.object(forKey: "TridaKnowledge.enabled") as? Bool ?? true
        graph = Self.load(file) ?? KnowledgeGraph()
    }

    static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.dateDecodingStrategy = .iso8601
        return d
    }()

    static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.dateEncodingStrategy = .iso8601
        e.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        return e
    }()

    static func load(_ url: URL) -> KnowledgeGraph? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return try? decoder.decode(KnowledgeGraph.self, from: data)
    }

    // MARK: - changing the graph

    public var canUndo: Bool { !undoStack.isEmpty }

    /// Every change goes through here: it can be undone, and the file is written shortly after.
    public func mutate(_ f: (inout KnowledgeGraph) -> Void) {
        var g = graph
        f(&g)
        guard g != graph else { return }
        undoStack.append(graph)
        if undoStack.count > 100 { undoStack.removeFirst() }
        graph = g
        scheduleSave()
    }

    public func undo() {
        guard let g = undoStack.popLast() else { return }
        graph = g
        scheduleSave()
    }

    /// Writes the file a moment after the last change (a personal graph is small).
    private func scheduleSave() {
        saveTask?.cancel()
        saveTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 400_000_000)
            guard !Task.isCancelled, let self else { return }
            self.saveNow()
        }
    }

    public func saveNow() {
        guard let data = try? Self.encoder.encode(graph) else { return }
        try? FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
        try? data.write(to: file, options: .atomic)
    }

    // MARK: - learning and recall

    /// Learns from a finished exchange in the background. Returns what was learned (nil: nothing, or off).
    @discardableResult
    public func learn(_ x: Exchange, backend: KnowledgeBackend) async -> LearnedRecord? {
        guard enabled else { return nil }
        learning = x.messageID
        defer { if learning == x.messageID { learning = nil } }
        do {
            let facts = try await Learner.propose(x, graph: graph, backend: backend)
            guard !facts.isEmpty else { return nil }
            var rec: LearnedRecord?
            mutate { rec = $0.apply(facts, x) }
            lastError = nil
            if let rec { lastLearned = LearnedPulse(count: rec.edges.count, at: Date()) }
            return rec
        } catch {
            lastError = error.localizedDescription
            return nil
        }
    }

    /// What one reply learned (for its chip).
    public func record(_ messageID: String) -> LearnedRecord? { graph.records[messageID] }

    /// The chip's Undo: takes back what that exchange learned.
    public func undoLearned(_ messageID: String) { mutate { $0.undo(record: messageID) } }

    /// The facts to add to the user's message, or nil.
    public func contextBlock(for text: String) -> String? { enabled ? graph.contextBlock(for: text) : nil }

    // MARK: - the window

    /// Shows a fact (from the chip): focuses its subject and selects it.
    public func show(edge id: String) {
        guard let e = graph.edges.first(where: { $0.id == id }) else { return }
        focus = e.subject
        selectedEdge = id
        selectedNode = nil
    }

    public func select(relation key: String) {
        selectedRelation = key
        relationText = graph.relation(key)?.label ?? ""
    }

    public func select(node id: String) {
        selectedNode = id
        selectedEdge = nil
        renameText = graph.node(id)?.name ?? ""
        aliasText = ""
        newObject = ""
    }

    /// The node the graph view is centered on: the focus, else the user, else the most connected node.
    public var center: String? {
        if let focus, graph.node(focus) != nil { return focus }
        if let me = graph.me, !graph.edges(touching: me.id).isEmpty { return me.id }
        let counts = Dictionary(grouping: graph.edges.flatMap { [$0.subject, $0.object] }, by: { $0 }).mapValues(\.count)
        return counts.max { $0.value < $1.value }?.key ?? graph.nodes.first?.id
    }

    /// The sidebar list: nodes matching the search and type filter, most connected first.
    public var listedNodes: [KGNode] {
        let q = KnowledgeGraph.normalize(search)
        let counts = Dictionary(grouping: graph.edges.flatMap { [$0.subject, $0.object] }, by: { $0 }).mapValues(\.count)
        return graph.nodes.filter { n in
            (typeFilter == nil || n.type == typeFilter)
                && (q.isEmpty || n.names.contains { KnowledgeGraph.normalize($0).contains(q) })
        }.sorted { (counts[$0.id] ?? 0, $1.name) > (counts[$1.id] ?? 0, $0.name) }
    }

    /// Adds a fact from the inspector: the selected node and a named node (found or created).
    public func addFact(from node: String) {
        let name = newObject.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return }
        let type = newObjectType, flipped = newObjectIsSubject, relation = newRelation
        mutate { g in
            let other: String
            if let found = g.node(named: name) { other = found.id } else { other = g.addNode(name: name, type: type) }
            let (s, o) = flipped ? (other, node) : (node, other)
            guard s != o else { return }
            var e = KGEdge(subject: s, relation: relation, object: o, origin: .edited)
            e.validFrom = Date()
            e.via = "manual"
            _ = g.addFact(e)
        }
        newObject = ""
    }
}
