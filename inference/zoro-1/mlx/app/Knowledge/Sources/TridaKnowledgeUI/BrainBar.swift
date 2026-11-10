import AppKit
import SwiftUI
import TridaKnowledge

/// The pixel brain at the bottom of the sidebar, with a small graph drawn across it (the knowledge graph is
/// Zoro-1's memory): click it to open the Knowledge window. It shows what the graph is doing: a node glowing now and
/// then when idle, a spark running along the links while learning, a pop and "+N" when
/// something was learned, an orange dot when new relations wait for review, grey when learning is off.
public struct BrainBar: View {
    @ObservedObject var model: KnowledgeModel
    /// Opens the Knowledge window; `true` opens it on the Relations tab.
    let open: (_ relations: Bool) -> Void

    public init(model: KnowledgeModel, open: @escaping (_ relations: Bool) -> Void) {
        self.model = model
        self.open = open
    }

    static let pulseSeconds = 4.0

    public var body: some View {
        let busy = model.learning != nil || model.lastLearned.map { Date().timeIntervalSince($0.at) < 2 } == true
        TimelineView(.periodic(from: .now, by: busy ? 0.1 : 0.5)) { tl in
            let now = tl.date
            let still = NSWorkspace.shared.accessibilityDisplayShouldReduceMotion
            let since = model.lastLearned.map { now.timeIntervalSince($0.at) }
            let pending = model.graph.customRelations.filter(\.provisional).count
            VStack(spacing: 6) {
                BrainSprite(time: still ? 0 : now.timeIntervalSinceReferenceDate,
                            learning: model.learning != nil && !still, off: !model.enabled)
                    .frame(width: 110, height: 80)          // 22 x 16 cells at 5 pt each
                    .scaleEffect(still ? 1 : Self.pop(since))
                    .overlay(alignment: .topTrailing) {
                        if pending > 0 {
                            Circle().fill(Color.orange).frame(width: 13, height: 13)
                                .overlay(Circle().stroke(Color(nsColor: .windowBackgroundColor), lineWidth: 2))
                                .offset(x: 4, y: 2)
                        }
                    }
                    .overlay(alignment: .top) {
                        if let since, since < 1.6, let n = model.lastLearned?.count, !still {
                            Text("+\(n)")
                                .font(.title3.weight(.bold))
                                .foregroundStyle(Color(red: 0.12, green: 0.54, blue: 0.36))
                                .offset(y: -14 - since * 22)
                                .opacity(1 - since / 1.6)
                        }
                    }
                VStack(spacing: 2) {
                    Text("Knowledge").font(.callout.weight(.semibold))
                    Text(caption(since: since))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                    if pending > 0 {
                        Button("Review \(pending) new relation\(pending == 1 ? "" : "s")") { open(true) }
                            .buttonStyle(.link)
                            .font(.caption)
                            .help("Relations added while learning: approve, rename or merge them")
                    }
                }
            }
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 12)
            .padding(.top, 20)          // room for the "+N" floating up
            .padding(.bottom, 10)
            .contentShape(Rectangle())
            .onTapGesture { open(false) }
            .help("Open the Knowledge window (⇧⌘K): what Zoro-1 remembers from your chats")
            .accessibilityElement(children: .combine)
            .accessibilityLabel("Knowledge: " + caption(since: since))
            .accessibilityAddTraits(.isButton)
        }
    }

    func caption(since: TimeInterval?) -> String {
        let g = model.graph
        if !model.enabled { return "Learning is off" }
        if model.learning != nil { return "Remembering…" }
        if let since, since < Self.pulseSeconds, let n = model.lastLearned?.count {
            return "Learned \(n) fact\(n == 1 ? "" : "s")"
        }
        let facts = g.edges.filter { $0.isCurrent() }.count
        if facts == 0 { return "Nothing learned yet" }
        return "\(facts) fact\(facts == 1 ? "" : "s") · \(g.nodes.count) entr\(g.nodes.count == 1 ? "y" : "ies")"
    }

    /// 1 → 1.18 → 1 over the first 0.35 s after learning.
    static func pop(_ since: TimeInterval?) -> CGFloat {
        guard let t = since, t >= 0, t < 0.35 else { return 1 }
        return 1 + 0.18 * CGFloat(sin(Double.pi * t / 0.35))
    }
}

/// The 22 × 16 pixel brain with a small knowledge graph drawn across it (green nodes, white links that also cross
/// between the hemispheres), on a Canvas with whole-point cells.
struct BrainSprite: View {
    let time: TimeInterval
    let learning: Bool
    let off: Bool

    // 'o' outline (column 11 is the fissure between the hemispheres), 'f' fill, 'h' highlight, 'l' stem
    static let sprite: [[Character]] = [
        ".....ooooo..ooooo.....",
        "...oohhfffoofffffoo...",
        "..ohhffffffoffffffo...",
        ".ohffffffffofffffffo..",
        ".offfffffffoffffffffo.",
        "offffffffffofffffffffo",
        "offffffffffofffffffffo",
        "offffffffffofffffffffo",
        "offffffffffofffffffffo",
        "offffffffffofffffffffo",
        ".offffffffofffffffffo.",
        "..offffffffoffffffffo.",
        "...ooffffffffffffoo...",
        ".....oooooooolllo.....",
        "............olllo.....",
        ".............ooo......",
    ].map(Array.init)

    /// Graph nodes (top-left cell of a 2 × 2 node) and the links between them.
    static let nodes: [(Int, Int)] = [(3, 4), (6, 8), (9, 3), (3, 15), (8, 17), (10, 13), (6, 13)]
    static let links: [(Int, Int)] = [(0, 1), (1, 2), (1, 6), (6, 3), (6, 4), (4, 5), (5, 2), (3, 4)]

    /// The cells of a straight line between two cells (Bresenham), drawn only inside the brain.
    static func line(_ a: (Int, Int), _ b: (Int, Int)) -> [(Int, Int)] {
        var (r, c) = a
        let dr = abs(b.0 - a.0), dc = abs(b.1 - a.1), sr = b.0 > a.0 ? 1 : -1, sc = b.1 > a.1 ? 1 : -1
        var err = dc - dr, out: [(Int, Int)] = []
        while true {
            out.append((r, c))
            if r == b.0 && c == b.1 { break }
            let e2 = 2 * err
            if e2 > -dr { err -= dr; c += sc }
            if e2 < dc { err += dc; r += sr }
        }
        return out.filter { r, c in
            let ch = sprite[r][c]
            return ch == "f" || ch == "h" || (c == 11 && (1...11).contains(r))
        }
    }

    /// Link cells in order, link after link: the path the learning spark runs along.
    static let linkCells: [(Int, Int)] = links.flatMap { line(nodes[$0.0], nodes[$0.1]) }

    static let colors: [Character: Color] = [
        "o": Color(red: 0.54, green: 0.23, blue: 0.35),
        "f": Color(red: 0.95, green: 0.61, blue: 0.71),
        "h": Color(red: 0.98, green: 0.83, blue: 0.87),
        "l": Color(red: 0.79, green: 0.37, blue: 0.51),
    ]
    static let link = Color.white
    static let node = Color(red: 0.12, green: 0.54, blue: 0.36)
    static let nodeLit = Color(red: 0.37, green: 0.84, blue: 0.63)
    static let spark = Color(red: 1.0, green: 0.84, blue: 0.35)

    var body: some View {
        Canvas { ctx, size in
            let rows = Self.sprite.count, cols = Self.sprite[0].count
            let cell = max(1, floor(min(size.width / CGFloat(cols), size.height / CGFloat(rows))))
            let ox = (size.width - cell * CGFloat(cols)) / 2, oy = (size.height - cell * CGFloat(rows)) / 2
            func fill(_ r: Int, _ c: Int, _ color: Color) {
                ctx.fill(Path(CGRect(x: ox + CGFloat(c) * cell, y: oy + CGFloat(r) * cell, width: cell, height: cell)),
                         with: .color(off ? color.grey : color))
            }
            for (r, row) in Self.sprite.enumerated() {
                for (c, ch) in row.enumerated() {
                    if let color = Self.colors[ch] { fill(r, c, color) }
                }
            }
            for (r, c) in Self.linkCells { fill(r, c, Self.link) }
            // which node glows: idle, one every three seconds for half a second; learning, the one the spark reaches
            var lit: Int? = nil
            if !off, time > 0, !Self.linkCells.isEmpty {
                let n = Self.linkCells.count
                if learning {
                    let head = Int(time * 12) % n
                    for k in 0..<3 {
                        let (r, c) = Self.linkCells[(head - k + n) % n]
                        fill(r, c, Self.spark.opacity(1 - Double(k) * 0.3))
                    }
                    let (hr, hc) = Self.linkCells[head]
                    lit = Self.nodes.firstIndex { abs($0.0 - hr) <= 1 && abs($0.1 - hc) <= 1 }
                } else if time.truncatingRemainder(dividingBy: 3) < 0.5 {
                    lit = (Int(time / 3) &* 7919) % Self.nodes.count
                }
            }
            for (i, (r, c)) in Self.nodes.enumerated() {
                for dr in 0...1 { for dc in 0...1 { fill(r + dr, c + dc, i == lit ? Self.nodeLit : Self.node) } }
            }
        }
        .accessibilityHidden(true)
    }
}

extension Color {
    /// The same color without its hue (the brain when learning is off).
    var grey: Color {
        let c = NSColor(self).usingColorSpace(.sRGB) ?? .gray
        let y = 0.3 * c.redComponent + 0.59 * c.greenComponent + 0.11 * c.blueComponent
        return Color(white: y)
    }
}
