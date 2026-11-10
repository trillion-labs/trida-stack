import Foundation

/// A line diff of two texts (longest common subsequence), for the code viewer.
public enum LineDiff {
    public enum Op: Equatable, Sendable { case same, added, removed }

    public struct Line: Equatable, Sendable {
        public var op: Op
        public var text: String
        /// 1-based line numbers in the old and new text (nil on the side the line is not in).
        public var old: Int?
        public var new: Int?

        public init(op: Op, text: String, old: Int? = nil, new: Int? = nil) {
            self.op = op
            self.text = text
            self.old = old
            self.new = new
        }
    }

    /// Above this many old × new lines (after the common start and end are taken off), the middle is shown as
    /// all removed, then all added, instead of computing the table.
    public static let maxCells = 4_000_000

    public static func lines(_ s: String) -> [String] {
        if s.isEmpty { return [] }
        var l = s.components(separatedBy: "\n")
        if l.last == "" { l.removeLast() }       // a trailing newline is not an extra empty line
        return l
    }

    public static func diff(_ before: String, _ after: String) -> [Line] {
        let a = lines(before), b = lines(after)
        var pre = 0
        while pre < a.count, pre < b.count, a[pre] == b[pre] { pre += 1 }
        var suf = 0
        while suf < a.count - pre, suf < b.count - pre, a[a.count - 1 - suf] == b[b.count - 1 - suf] { suf += 1 }
        var out: [Line] = (0..<pre).map { Line(op: .same, text: a[$0], old: $0 + 1, new: $0 + 1) }
        let ma = Array(a[pre..<(a.count - suf)]), mb = Array(b[pre..<(b.count - suf)])
        out += middle(ma, mb, oldStart: pre + 1, newStart: pre + 1)
        for k in 0..<suf {
            let i = a.count - suf + k, j = b.count - suf + k
            out.append(Line(op: .same, text: a[i], old: i + 1, new: j + 1))
        }
        return out
    }

    static func middle(_ a: [String], _ b: [String], oldStart: Int, newStart: Int) -> [Line] {
        let n = a.count, m = b.count
        if n == 0 || m == 0 || n * m > maxCells {
            return a.enumerated().map { Line(op: .removed, text: $1, old: oldStart + $0, new: nil) }
                + b.enumerated().map { Line(op: .added, text: $1, old: nil, new: newStart + $0) }
        }
        // lcs[i][j] = LCS length of a[i...] and b[j...], in one flat table
        var lcs = [Int32](repeating: 0, count: (n + 1) * (m + 1))
        let w = m + 1
        for i in stride(from: n - 1, through: 0, by: -1) {
            for j in stride(from: m - 1, through: 0, by: -1) {
                lcs[i * w + j] = a[i] == b[j] ? lcs[(i + 1) * w + j + 1] + 1 : max(lcs[(i + 1) * w + j], lcs[i * w + j + 1])
            }
        }
        var out: [Line] = []
        var i = 0, j = 0
        while i < n || j < m {
            if i < n, j < m, a[i] == b[j] {
                out.append(Line(op: .same, text: a[i], old: oldStart + i, new: newStart + j)); i += 1; j += 1
            } else if i < n, j == m || lcs[(i + 1) * w + j] >= lcs[i * w + j + 1] {   // removals before additions
                out.append(Line(op: .removed, text: a[i], old: oldStart + i, new: nil)); i += 1
            } else {
                out.append(Line(op: .added, text: b[j], old: nil, new: newStart + j)); j += 1
            }
        }
        return out
    }

    /// The changed lines with `context` unchanged lines around each change; nil entries mark skipped runs.
    public static func hunks(_ d: [Line], context: Int = 3) -> [Line?] {
        let changed = d.indices.filter { d[$0].op != .same }
        guard !changed.isEmpty else { return [] }
        var keep = [Bool](repeating: false, count: d.count)
        for c in changed { for k in max(0, c - context)...min(d.count - 1, c + context) { keep[k] = true } }
        var out: [Line?] = []
        for (k, line) in d.enumerated() {
            if keep[k] { out.append(line) } else if out.last.map({ $0 != nil }) ?? true { out.append(nil) }
        }
        if out.last == .some(nil) { out.removeLast() }
        return out
    }
}
