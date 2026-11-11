import Foundation

/// Positions for the whole-graph view: each connected group of entries is laid out with a force-directed layout
/// (Fruchterman–Reingold), and the groups, isolated entries included, are packed in rows, largest first.
/// Deterministic: the same graph always gets the same picture.
public enum KnowledgeLayout {
    public struct Point: Equatable, Sendable {
        public var x: Double
        public var y: Double
        public init(_ x: Double, _ y: Double) { self.x = x; self.y = y }
    }

    /// The distance between linked entries (the layout's unit).
    public static let unit = 90.0

    /// Node id → position, for every node (`includeEnded`: ended facts also pull their entries together).
    public static func layout(_ g: KnowledgeGraph, includeEnded: Bool = true) -> [String: Point] {
        let ids = g.nodes.map(\.id)
        let index = Dictionary(uniqueKeysWithValues: ids.enumerated().map { ($1, $0) })
        var adj = [Set<Int>](repeating: [], count: ids.count)
        for e in g.edges where includeEnded || e.isCurrent() {
            guard let a = index[e.subject], let b = index[e.object], a != b else { continue }
            adj[a].insert(b)
            adj[b].insert(a)
        }
        // connected groups
        var group = [Int](repeating: -1, count: ids.count), groups: [[Int]] = []
        for start in ids.indices where group[start] < 0 {
            var stack = [start], members: [Int] = []
            group[start] = groups.count
            while let v = stack.popLast() {
                members.append(v)
                for w in adj[v] where group[w] < 0 { group[w] = groups.count; stack.append(w) }
            }
            groups.append(members.sorted())
        }
        // each group on its own, then packed: largest first, in rows about as wide as the total is tall
        var placed: [(members: [Int], pos: [Point], w: Double, h: Double)] = groups.map { m in
            let p = forceLayout(m, adj)
            let xs = p.map(\.x), ys = p.map(\.y)
            let minX = xs.min() ?? 0, minY = ys.min() ?? 0
            let shifted = p.map { Point($0.x - minX, $0.y - minY) }
            return (m, shifted, (xs.max() ?? 0) - minX, (ys.max() ?? 0) - minY)
        }
        placed.sort { ($0.members.count, $1.members.first ?? 0) > ($1.members.count, $0.members.first ?? 0) }
        let gap = unit * 0.9
        let area = placed.reduce(0.0) { $0 + ($1.w + gap) * ($1.h + gap) }
        let rowWidth = max(unit * 4, sqrt(area) * 1.4)
        var out: [String: Point] = [:]
        var x = 0.0, y = 0.0, rowH = 0.0
        for p in placed {
            if x > 0 && x + p.w > rowWidth { x = 0; y += rowH + gap; rowH = 0 }
            for (k, v) in p.members.enumerated() { out[ids[v]] = Point(x + p.pos[k].x, y + p.pos[k].y) }
            x += p.w + gap
            rowH = max(rowH, p.h)
        }
        return out
    }

    /// Fruchterman–Reingold for one group (positions relative to the group).
    static func forceLayout(_ members: [Int], _ adj: [Set<Int>]) -> [Point] {
        let n = members.count
        if n == 1 { return [Point(0, 0)] }
        let local = Dictionary(uniqueKeysWithValues: members.enumerated().map { ($1, $0) })
        let k = unit
        // start on a circle (deterministic), sized for the group
        let r0 = k * sqrt(Double(n)) * 0.6
        var x = (0..<n).map { r0 * cos(2 * .pi * Double($0) / Double(n)) }
        var y = (0..<n).map { r0 * sin(2 * .pi * Double($0) / Double(n)) }
        var t = r0 / 2
        let iterations = n > 200 ? 120 : 250
        for _ in 0..<iterations {
            var dx = [Double](repeating: 0, count: n), dy = [Double](repeating: 0, count: n)
            for i in 0..<n {
                for j in (i + 1)..<n {
                    var ddx = x[i] - x[j], ddy = y[i] - y[j]
                    var d = (ddx * ddx + ddy * ddy).squareRoot()
                    if d < 0.01 { ddx = 0.01 * Double(i - j); ddy = 0.01; d = 0.015 }
                    let f = k * k / d                         // repulsion
                    dx[i] += ddx / d * f; dy[i] += ddy / d * f
                    dx[j] -= ddx / d * f; dy[j] -= ddy / d * f
                }
            }
            for (a, v) in members.enumerated() {
                for w in adj[v] {
                    guard let b = local[w], b > a else { continue }
                    let ddx = x[a] - x[b], ddy = y[a] - y[b]
                    let d = max(0.01, (ddx * ddx + ddy * ddy).squareRoot())
                    let f = d * d / k                         // attraction along links
                    dx[a] -= ddx / d * f; dy[a] -= ddy / d * f
                    dx[b] += ddx / d * f; dy[b] += ddy / d * f
                }
            }
            for i in 0..<n {
                let d = max(0.01, (dx[i] * dx[i] + dy[i] * dy[i]).squareRoot())
                x[i] += dx[i] / d * min(d, t)
                y[i] += dy[i] / d * min(d, t)
            }
            t = max(1, t * 0.96)                             // cooling
        }
        return (0..<n).map { Point(x[$0], y[$0]) }
    }
}
