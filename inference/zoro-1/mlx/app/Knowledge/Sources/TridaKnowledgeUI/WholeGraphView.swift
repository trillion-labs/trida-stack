import AppKit
import SwiftUI
import TridaKnowledge

/// Every entry and fact at once: each connected group laid out by force, the groups (isolated entries too) packed
/// side by side. Facts are arrows from subject to object. Drag to pan, pinch to zoom; click an entry to inspect it,
/// double-click to open it in the focus view.
struct WholeGraphView: View {
    @ObservedObject var model: KnowledgeModel

    static let radius: CGFloat = 9

    /// The facts between two entries, drawn as one line (heads show which way they go).
    struct Pair {
        var a: String
        var b: String
        var ab = false
        var ba = false
        var current = false
        var labels: [String] = []
    }

    var body: some View {
        GeometryReader { geo in
            let g = model.graph
            let pos = model.layout()
            let size = geo.size
            let t = transform(pos, size)
            let screen: [String: CGPoint] = pos.mapValues { t($0) }
            let scale = fitScale(pos, size) * model.zoom
            let edges = g.edges.filter { model.showEnded || $0.isCurrent() }
            let sel = model.selectedNode
            ZStack {
                if g.nodes.isEmpty {
                    Text("Nothing learned yet.").foregroundStyle(.secondary)
                } else {
                    Canvas { ctx, _ in
                        // pairs with facts both ways get one line with two heads
                        var pairs: [String: Pair] = [:]
                        for e in edges {
                            let lo = min(e.subject, e.object), hi = max(e.subject, e.object)
                            let k = lo + "|" + hi
                            var p = pairs[k] ?? Pair(a: lo, b: hi)
                            if e.subject == p.a { p.ab = true } else { p.ba = true }
                            p.current = p.current || e.isCurrent()
                            p.labels.append(g.relationLabel(e.relation))
                            pairs[k] = p
                        }
                        let showLabels = scale >= 0.85
                        for (_, p) in pairs {
                            guard let a = screen[p.a], let b = screen[p.b] else { continue }
                            let hot = sel == p.a || sel == p.b
                            Arrow.draw(in: ctx, from: a, to: b, fromRadius: Self.radius + 2, toRadius: Self.radius + 2,
                                       headAtEnd: p.ab, headAtStart: p.ba,
                                       color: hot ? Color.accentColor.opacity(0.9)
                                                  : Color.secondary.opacity(p.current ? 0.55 : 0.25),
                                       width: hot ? 1.8 : 1.2, dashed: !p.current, head: 7)
                            if showLabels {
                                let mid = CGPoint(x: (a.x + b.x) / 2, y: (a.y + b.y) / 2)
                                let text = p.labels.prefix(2).joined(separator: " · ") + (p.labels.count > 2 ? " …" : "")
                                ctx.draw(Text(text).font(.system(size: 9)).foregroundColor(.secondary), at: mid)
                            }
                        }
                        let showNames = scale >= 0.55 || g.nodes.count <= 60
                        for n in g.nodes {
                            guard let c = screen[n.id] else { continue }
                            let selected = sel == n.id
                            let r = Self.radius * (n.type == .me ? 1.4 : 1)
                            let circle = Path(ellipseIn: CGRect(x: c.x - r, y: c.y - r, width: 2 * r, height: 2 * r))
                            ctx.fill(circle, with: .color(KnowledgeColors.of(n.type)))
                            if selected {
                                ctx.stroke(Path(ellipseIn: CGRect(x: c.x - r - 3, y: c.y - r - 3, width: 2 * r + 6,
                                                                  height: 2 * r + 6)),
                                           with: .color(.primary.opacity(0.8)), lineWidth: 2)
                            }
                            if showNames || selected {
                                ctx.draw(Text(n.name).font(.system(size: 11, weight: selected ? .semibold : .regular))
                                            .foregroundColor(.primary),
                                         at: CGPoint(x: c.x, y: c.y + r + 9))
                            }
                        }
                    }
                    .contentShape(Rectangle())
                    .gesture(SpatialTapGesture(count: 2).onEnded { v in
                        if let id = hit(v.location, screen) {
                            model.focus = id
                            model.select(node: id)
                            model.wholeGraph = false
                        }
                    }.exclusively(before: SpatialTapGesture().onEnded { v in
                        if let id = hit(v.location, screen) { model.select(node: id) } else { model.selectedNode = nil }
                    }))
                    .gesture(DragGesture(minimumDistance: 3).onChanged { v in
                        let base = model.panBase ?? model.pan
                        model.panBase = base
                        model.pan = CGSize(width: base.width + v.translation.width, height: base.height + v.translation.height)
                    }.onEnded { _ in model.panBase = nil })
                    .simultaneousGesture(MagnifyGesture().onChanged { v in
                        let base = model.zoomBase ?? model.zoom
                        model.zoomBase = base
                        model.zoom = min(6, max(0.2, base * v.magnification))
                    }.onEnded { _ in model.zoomBase = nil })
                }
                VStack {
                    Spacer()
                    HStack(spacing: 6) {
                        Text(summary(g)).font(.caption).foregroundStyle(.secondary)
                        Spacer()
                        Button { model.zoom = max(0.2, model.zoom / 1.25) } label: { Image(systemName: "minus.magnifyingglass") }
                        Button { model.zoom = min(6, model.zoom * 1.25) } label: { Image(systemName: "plus.magnifyingglass") }
                        Button("Fit") { model.resetView() }
                    }
                    .buttonStyle(.borderless)
                    .padding(8)
                    .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
                    .padding(10)
                }
            }
        }
        .background(Color(nsColor: .textBackgroundColor).opacity(0.5))
        .clipped()
    }

    func summary(_ g: KnowledgeGraph) -> String {
        let linked = Set(g.edges.flatMap { [$0.subject, $0.object] })
        let isolated = g.nodes.filter { !linked.contains($0.id) }.count
        return "\(g.nodes.count) entries · \(g.edges.filter { $0.isCurrent() }.count) facts"
            + (isolated > 0 ? " · \(isolated) without facts" : "") + " · drag to move, pinch to zoom"
    }

    /// Fits the layout into the view (with room for names), before the user's zoom.
    func fitScale(_ pos: [String: KnowledgeLayout.Point], _ size: CGSize) -> CGFloat {
        let xs = pos.values.map { CGFloat($0.x) }, ys = pos.values.map { CGFloat($0.y) }
        let w: CGFloat = max(1, (xs.max() ?? 0) - (xs.min() ?? 0))
        let h: CGFloat = max(1, (ys.max() ?? 0) - (ys.min() ?? 0))
        let fx: CGFloat = (size.width - 120) / w, fy: CGFloat = (size.height - 110) / h
        return max(0.05, min(1.6, min(fx, fy)))
    }

    func transform(_ pos: [String: KnowledgeLayout.Point], _ size: CGSize) -> (KnowledgeLayout.Point) -> CGPoint {
        let xs = pos.values.map { CGFloat($0.x) }, ys = pos.values.map { CGFloat($0.y) }
        let cx: CGFloat = ((xs.max() ?? 0) + (xs.min() ?? 0)) / 2
        let cy: CGFloat = ((ys.max() ?? 0) + (ys.min() ?? 0)) / 2
        let s: CGFloat = fitScale(pos, size) * model.zoom
        let ox: CGFloat = size.width / 2 + model.pan.width
        let oy: CGFloat = (size.height - 30) / 2 + model.pan.height
        return { p in CGPoint(x: ox + (CGFloat(p.x) - cx) * s, y: oy + (CGFloat(p.y) - cy) * s) }
    }

    func hit(_ p: CGPoint, _ screen: [String: CGPoint]) -> String? {
        let best = screen.min { a, b in
            hypot(a.value.x - p.x, a.value.y - p.y) < hypot(b.value.x - p.x, b.value.y - p.y)
        }
        guard let best, hypot(best.value.x - p.x, best.value.y - p.y) <= Self.radius + 8 else { return nil }
        return best.key
    }
}
