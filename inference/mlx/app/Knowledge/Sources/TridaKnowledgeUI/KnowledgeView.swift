import AppKit
import SwiftUI
import TridaKnowledge

/// Colors for node types: a fixed, calm palette that reads in light and dark mode.
public enum KnowledgeColors {
    public static func of(_ t: NodeType) -> Color {
        switch t {
        case .me: return Color(red: 0.12, green: 0.54, blue: 0.36)
        case .person: return Color(red: 0.23, green: 0.45, blue: 0.78)
        case .org: return Color(red: 0.49, green: 0.36, blue: 0.75)
        case .project: return Color(red: 0.85, green: 0.47, blue: 0.16)
        case .place: return Color(red: 0.13, green: 0.6, blue: 0.64)
        case .file: return Color(red: 0.45, green: 0.47, blue: 0.5)
        case .topic: return Color(red: 0.77, green: 0.33, blue: 0.5)
        case .event: return Color(red: 0.72, green: 0.58, blue: 0.1)
        case .value: return Color(red: 0.55, green: 0.55, blue: 0.55)
        }
    }
}

/// The Knowledge window: the list of entities, the graph around one of them, and an inspector to edit it.
public struct KnowledgeView: View {
    @ObservedObject var model: KnowledgeModel

    public init(model: KnowledgeModel) { self.model = model }

    public var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            HStack(spacing: 0) {
                KnowledgeSidebar(model: model).frame(width: 230)
                Divider()
                KnowledgeGraphView(model: model).frame(minWidth: 420, maxWidth: .infinity, maxHeight: .infinity)
                Divider()
                KnowledgeInspector(model: model).frame(width: 310)
            }
        }
        .frame(minWidth: 1000, minHeight: 560)
    }

    var header: some View {
        HStack(spacing: 14) {
            Text("Knowledge").font(.headline)
            Text("\(model.graph.nodes.count) entries · \(model.graph.edges.filter { $0.isCurrent() }.count) facts")
                .font(.caption).foregroundStyle(.secondary)
            Spacer()
            Toggle("Show facts that ended", isOn: $model.showEnded).toggleStyle(.checkbox)
            Toggle("Learn from chats", isOn: $model.enabled).toggleStyle(.switch).controlSize(.small)
            Button { model.undo() } label: { Label("Undo", systemImage: "arrow.uturn.backward") }
                .keyboardShortcut("z", modifiers: .command)
                .disabled(!model.canUndo)
        }
        .padding(.horizontal, 14)
        .padding(.vertical, 8)
    }
}

// MARK: - sidebar

struct KnowledgeSidebar: View {
    @ObservedObject var model: KnowledgeModel

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            TextField("Search", text: $model.search).textFieldStyle(.roundedBorder)
            Picker("Type", selection: $model.typeFilter) {
                Text("All types").tag(NodeType?.none)
                ForEach(NodeType.allCases, id: \.self) { t in Text(t.label).tag(NodeType?.some(t)) }
            }
            .labelsHidden()
            let nodes = model.listedNodes
            if nodes.isEmpty {
                Text(model.graph.nodes.isEmpty
                     ? "Nothing learned yet. Facts from your chats show up here."
                     : "No matches.")
                    .font(.callout).foregroundStyle(.secondary).padding(.top, 8)
                Spacer()
            } else {
                List(nodes) { n in
                    HStack(spacing: 8) {
                        Circle().fill(KnowledgeColors.of(n.type)).frame(width: 9, height: 9)
                        Text(n.name).lineLimit(1)
                        Spacer()
                        Text("\(model.graph.edges(touching: n.id).count)").font(.caption).foregroundStyle(.secondary)
                    }
                    .contentShape(Rectangle())
                    .onTapGesture {
                        model.focus = n.id
                        model.select(node: n.id)
                    }
                    .listRowBackground(n.id == model.center ? Color.accentColor.opacity(0.15) : Color.clear)
                }
                .listStyle(.plain)
            }
        }
        .padding(10)
    }
}

// MARK: - graph

/// The focus view: one entity in the middle, the entities it has facts with around it, a line per pair labeled
/// with the facts. Ended facts are dashed and faded. Click an entity to inspect it, double-click to move there;
/// click a label to inspect that fact.
struct KnowledgeGraphView: View {
    @ObservedObject var model: KnowledgeModel

    struct Spoke: Identifiable {
        var id: String { node.id }
        let node: KGNode
        let edges: [KGEdge]
        var pos: CGPoint
    }

    var body: some View {
        GeometryReader { geo in
            let size = geo.size
            if let cid = model.center, let center = model.graph.node(cid) {
                let mid = CGPoint(x: size.width / 2, y: size.height / 2)
                let spokes = layout(cid, in: size)
                ZStack {
                    Canvas { ctx, _ in
                        for s in spokes {
                            var p = Path()
                            p.move(to: mid)
                            p.addLine(to: s.pos)
                            let current = s.edges.contains { $0.isCurrent() }
                            ctx.stroke(p, with: .color(.secondary.opacity(current ? 0.55 : 0.25)),
                                       style: StrokeStyle(lineWidth: current ? 1.4 : 1, dash: current ? [] : [5, 4]))
                        }
                    }
                    ForEach(spokes) { s in
                        edgeLabels(s).position(x: (mid.x + s.pos.x) / 2, y: (mid.y + s.pos.y) / 2)
                        nodeBubble(s.node, big: false).position(s.pos)
                    }
                    nodeBubble(center, big: true).position(mid)
                    if spokes.isEmpty {
                        Text("No facts about \(center.name) yet.").font(.callout).foregroundStyle(.secondary)
                            .position(x: mid.x, y: mid.y + 60)
                    }
                }
            } else {
                VStack(spacing: 8) {
                    Image(systemName: "point.3.connected.trianglepath.dotted").font(.system(size: 40))
                        .foregroundStyle(.secondary)
                    Text("Your knowledge graph is empty.").font(.headline)
                    Text("As you chat, Zoro-1 remembers lasting facts: about you and your work, corrections you make, "
                         + "and answers it looked up on the web. Each one shows up under the reply with an Undo.")
                        .font(.callout).foregroundStyle(.secondary).multilineTextAlignment(.center)
                        .frame(maxWidth: 360)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .background(Color(nsColor: .textBackgroundColor).opacity(0.5))
    }

    /// Neighbors on one ring (two alternating rings when crowded), grouped by type around the circle.
    func layout(_ cid: String, in size: CGSize) -> [Spoke] {
        let g = model.graph
        var byNode: [String: [KGEdge]] = [:]
        for e in g.edges(touching: cid) where model.showEnded || e.isCurrent() {
            let other = e.subject == cid ? e.object : e.subject
            guard other != cid else { continue }
            byNode[other, default: []].append(e)
        }
        let nodes = byNode.keys.compactMap { g.node($0) }.sorted {
            ($0.type.rawValue, $0.name) < ($1.type.rawValue, $1.name)
        }.prefix(36)
        let r = max(90, min(size.width, size.height) / 2 - 70)
        let mid = CGPoint(x: size.width / 2, y: size.height / 2)
        return nodes.enumerated().map { i, n in
            let a = Double(i) / Double(max(nodes.count, 1)) * 2 * .pi - .pi / 2
            let rr = nodes.count > 14 && i % 2 == 1 ? r * 0.62 : r
            return Spoke(node: n, edges: byNode[n.id] ?? [],
                         pos: CGPoint(x: mid.x + rr * cos(a), y: mid.y + rr * sin(a)))
        }
    }

    func nodeBubble(_ n: KGNode, big: Bool) -> some View {
        let selected = model.selectedNode == n.id && model.selectedEdge == nil
        return VStack(spacing: 3) {
            Circle()
                .fill(KnowledgeColors.of(n.type))
                .frame(width: big ? 30 : 18, height: big ? 30 : 18)
                .overlay(Circle().stroke(Color.primary.opacity(selected ? 0.8 : 0), lineWidth: 2).padding(-3))
            Text(n.name)
                .font(big ? .callout.weight(.semibold) : .caption)
                .lineLimit(2)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 120)
        }
        .contentShape(Rectangle())
        .onTapGesture(count: 2) {
            model.focus = n.id
            model.select(node: n.id)
        }
        .onTapGesture { model.select(node: n.id) }
        .help(n.type.label + (n.aliases.isEmpty ? "" : " · also " + n.aliases.joined(separator: ", ")))
    }

    func edgeLabels(_ s: Spoke) -> some View {
        VStack(spacing: 2) {
            ForEach(s.edges.prefix(3)) { e in
                let out = e.subject == model.center
                Text((out ? "" : "← ") + Relations.label(e.relation) + (out ? " →" : ""))
                    .font(.caption2)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Capsule().fill(model.selectedEdge == e.id ? Color.accentColor.opacity(0.3)
                                                                         : Color(nsColor: .windowBackgroundColor)))
                    .overlay(Capsule().stroke(Color.secondary.opacity(0.3)))
                    .opacity(e.isCurrent() ? 1 : 0.5)
                    .onTapGesture { model.selectedEdge = e.id }
                    .help(model.graph.describe(e))
            }
            if s.edges.count > 3 {
                Text("+\(s.edges.count - 3)").font(.caption2).foregroundStyle(.secondary)
            }
        }
    }
}

// MARK: - inspector

struct KnowledgeInspector: View {
    @ObservedObject var model: KnowledgeModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                if let eid = model.selectedEdge, let e = model.graph.edges.first(where: { $0.id == eid }) {
                    edgeInspector(e)
                } else if let nid = model.selectedNode ?? model.center, let n = model.graph.node(nid) {
                    nodeInspector(n)
                } else {
                    Text("Select an entry or a fact.").foregroundStyle(.secondary)
                }
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    // a node: name, type, aliases, facts, add a fact, merge, delete

    func nodeInspector(_ n: KGNode) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Circle().fill(KnowledgeColors.of(n.type)).frame(width: 12, height: 12)
                Text(n.name).font(.title3.weight(.semibold))
            }
            if n.type != .me {
                TextField("Name", text: $model.renameText)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit {
                        let t = model.renameText.trimmingCharacters(in: .whitespacesAndNewlines)
                        if !t.isEmpty { model.mutate { $0.update(node: n.id) { $0.name = t } } }
                    }
                Picker("Type", selection: Binding(get: { n.type }, set: { t in
                    model.mutate { $0.update(node: n.id) { $0.type = t } }
                })) {
                    ForEach(NodeType.allCases.filter { $0 != .me }, id: \.self) { Text($0.label).tag($0) }
                }
            }
            section("Also called") {
                ForEach(n.aliases, id: \.self) { a in
                    HStack {
                        Text(a)
                        Spacer()
                        Button { model.mutate { $0.update(node: n.id) { $0.aliases.removeAll { $0 == a } } } } label: {
                            Image(systemName: "xmark.circle")
                        }
                        .buttonStyle(.borderless)
                    }
                }
                TextField("Add a name", text: $model.aliasText)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit {
                        let a = model.aliasText
                        model.mutate { $0.addAlias(a, to: n.id) }
                        model.aliasText = ""
                    }
            }
            section("Facts") {
                let es = model.graph.edges(touching: n.id).sorted {
                    ($0.isCurrent() ? 1 : 0, $0.learnedAt) > ($1.isCurrent() ? 1 : 0, $1.learnedAt)
                }
                if es.isEmpty { Text("None").foregroundStyle(.secondary) }
                ForEach(es) { e in
                    Button { model.selectedEdge = e.id } label: {
                        Text(model.graph.describe(e))
                            .font(.callout)
                            .foregroundStyle(e.isCurrent() ? Color.primary : Color.secondary)
                            .strikethrough(!e.isCurrent())
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .buttonStyle(.plain)
                }
            }
            section("Add a fact") {
                Picker("Relation", selection: $model.newRelation) {
                    ForEach(Relations.all, id: \.key) { Text($0.label).tag($0.key) }
                }
                Toggle("\(n.name) is the object", isOn: $model.newObjectIsSubject).toggleStyle(.checkbox)
                TextField("Other entry (found by name, or new)", text: $model.newObject)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { model.addFact(from: n.id) }
                Picker("New entry's type", selection: $model.newObjectType) {
                    ForEach(NodeType.allCases.filter { $0 != .me }, id: \.self) { Text($0.label).tag($0) }
                }
                Button("Add") { model.addFact(from: n.id) }
                    .disabled(model.newObject.trimmingCharacters(in: .whitespaces).isEmpty)
            }
            if n.type != .me {
                HStack {
                    let others = model.graph.similar(to: n.name, excluding: [n.id], limit: 8)
                        + model.graph.nodes.filter { $0.id != n.id && $0.type == n.type && $0.type != .me }.prefix(12)
                    Menu("Merge into…") {
                        ForEach(Array(Set(others.map(\.id))).compactMap { model.graph.node($0) }.sorted { $0.name < $1.name }) { o in
                            Button("\(o.name) (\(o.type.label))") {
                                model.mutate { $0.merge(n.id, into: o.id) }
                                model.focus = o.id
                                model.select(node: o.id)
                            }
                        }
                    }
                    .help("The same thing under two names: move this entry's names and facts onto another")
                    .disabled(others.isEmpty)
                    Spacer()
                    Button("Delete", role: .destructive) {
                        model.mutate { $0.deleteNode(n.id) }
                        model.selectedNode = nil
                        if model.focus == n.id { model.focus = nil }
                    }
                }
            }
        }
    }

    // a fact: what it says, where it came from, still true or not, delete

    func edgeInspector(_ e: KGEdge) -> some View {
        let g = model.graph
        return VStack(alignment: .leading, spacing: 12) {
            Button { model.selectedEdge = nil } label: { Label("Back", systemImage: "chevron.left") }
                .buttonStyle(.borderless)
            Text(g.describe(e)).font(.title3.weight(.semibold)).fixedSize(horizontal: false, vertical: true)
            HStack(spacing: 6) {
                Button(g.name(e.subject)) { model.select(node: e.subject) }
                Picker("", selection: Binding(get: { e.relation }, set: { r in
                    model.mutate { $0.update(edge: e.id) { $0.relation = r; $0.origin = .edited } }
                })) {
                    ForEach(Relations.all, id: \.key) { Text($0.label).tag($0.key) }
                }
                .labelsHidden()
                .frame(maxWidth: 150)
                Button(g.name(e.object)) { model.select(node: e.object) }
            }
            .buttonStyle(.link)
            section("Where it comes from") {
                Text(g.provenance(e)).font(.callout)
                if let q = e.quote {
                    Text("“\(q)”").font(.callout).italic().foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                if let u = e.sourceURL, let url = URL(string: u) { Link(u, destination: url).font(.caption).lineLimit(1) }
                if let c = e.chatID, let m = e.messageID, let open = model.openSource {
                    Button("Open the message it came from") { open(c, m) }.buttonStyle(.link)
                }
            }
            section("When") {
                Text("True since " + (e.validFrom.map(Self.day) ?? "—")
                     + (e.validTo.map { " · ended " + Self.day($0) } ?? " · still true"))
                    .font(.callout)
                if let by = e.supersededBy, let newer = g.edges.first(where: { $0.id == by }) {
                    Button("Replaced by: " + g.describe(newer)) { model.selectedEdge = newer.id }.buttonStyle(.link)
                }
            }
            HStack {
                if e.isCurrent() {
                    Button("No longer true") { model.mutate { $0.setEnded(e.id) } }
                } else {
                    Button("Still true") { model.mutate { $0.setEnded(e.id, at: nil) } }
                }
                Spacer()
                Button("Delete", role: .destructive) {
                    model.mutate { $0.deleteEdge(e.id) }
                    model.selectedEdge = nil
                }
            }
        }
    }

    static func day(_ d: Date) -> String { d.formatted(date: .abbreviated, time: .omitted) }

    func section<C: View>(_ title: String, @ViewBuilder _ content: () -> C) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title.uppercased()).font(.caption2.weight(.semibold)).foregroundStyle(.secondary)
            content()
        }
    }
}

// MARK: - chip under a reply

/// Under a reply: "Learning…" while it runs, then "Learned: A —relation→ B (+N)" with Undo. Clicking the text opens
/// the fact in the Knowledge window (`open` is given the edge id).
public struct LearnedChip: View {
    @ObservedObject var model: KnowledgeModel
    let messageID: String
    let open: (String) -> Void

    public init(model: KnowledgeModel, messageID: String, open: @escaping (String) -> Void) {
        self.model = model
        self.messageID = messageID
        self.open = open
    }

    public var body: some View {
        if model.learning == messageID {
            HStack(spacing: 6) {
                ProgressView().controlSize(.mini)
                Text("Remembering…").font(.caption).foregroundStyle(.secondary)
            }
        } else if let rec = model.record(messageID), let first = rec.edges.first,
                  let e = model.graph.edges.first(where: { $0.id == first }) {
            HStack(spacing: 8) {
                Image(systemName: e.origin == .web ? "globe" : "brain").foregroundStyle(.secondary)
                Button {
                    open(first)
                } label: {
                    Text((e.origin == .web ? "Learned from \(e.sourceURL.flatMap { URL(string: $0)?.host } ?? "the web"): "
                                           : "Learned: ") + model.graph.describe(e)
                         + (rec.edges.count > 1 ? "  +\(rec.edges.count - 1) more" : ""))
                        .font(.caption)
                        .lineLimit(1)
                }
                .buttonStyle(.plain)
                .help("Open in the Knowledge window")
                Button("Undo") { model.undoLearned(messageID) }
                    .buttonStyle(.borderless)
                    .font(.caption)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 3)
            .background(Capsule().fill(Color.secondary.opacity(0.1)))
        }
    }
}
