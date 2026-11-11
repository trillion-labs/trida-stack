import Foundation

/// The block structure of a Markdown reply, for the app's native renderer: headings, paragraphs, lists
/// (nested, ordered, task items), block quotes, fenced code, GitHub tables, rules and display math
/// (`$$ … $$`, `\[ … \]`). Inline syntax (bold, italics, code, links) stays in the text for
/// AttributedString's inline parser; `\( … \)` becomes a code span. Line-based and forgiving, because it
/// runs on every streamed token: an unclosed code fence or math block renders as one so far.
public enum MDBlock: Equatable, Sendable {
    case heading(level: Int, text: String)
    case paragraph(String)
    case list([MDListItem])
    case quote(String)
    case code(language: String, text: String)
    case table(header: [String], alignments: [MDAlignment], rows: [[String]])
    case rule
    case math(String)
}

public struct MDListItem: Equatable, Sendable {
    public var depth: Int            // 0 for top-level items, +1 per nesting level
    public var marker: String        // "•", "1.", "2)", ...
    public var checked: Bool?        // task items: false = "[ ]", true = "[x]"
    public var text: String
}

public enum MDAlignment: Equatable, Sendable { case leading, center, trailing }

public enum Markdown {
    public static func parse(_ source: String) -> [MDBlock] {
        var out: [MDBlock] = []
        let lines = source.replacingOccurrences(of: "\r\n", with: "\n").components(separatedBy: "\n")
        var i = 0
        var para: [String] = []
        var items: [MDListItem] = []
        var quote: [String] = []

        func flushPara() {
            let t = para.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
            if !t.isEmpty { out.append(.paragraph(inline(t))) }
            para = []
        }
        func flushList() {
            // depth holds raw indentation until here: number the distinct indents 0, 1, 2, ...
            let levels = Array(Set(items.map(\.depth))).sorted()
            for k in items.indices { items[k].depth = levels.firstIndex(of: items[k].depth) ?? 0 }
            if !items.isEmpty { out.append(.list(items)) }
            items = []
        }
        func flushQuote() {
            if !quote.isEmpty { out.append(.quote(inline(quote.joined(separator: "\n")))) }
            quote = []
        }
        func flushAll() { flushPara(); flushList(); flushQuote() }

        while i < lines.count {
            let line = lines[i]
            let t = line.trimmingCharacters(in: .whitespaces)

            // fenced code
            if let fence = ["```", "~~~"].first(where: { t.hasPrefix($0) }) {
                flushAll()
                let lang = String(t.dropFirst(fence.count)).trimmingCharacters(in: .whitespaces)
                var body: [String] = []
                i += 1
                while i < lines.count, !lines[i].trimmingCharacters(in: .whitespaces).hasPrefix(fence) {
                    body.append(lines[i]); i += 1
                }
                out.append(.code(language: lang, text: body.joined(separator: "\n")))
                i += 1
                continue
            }
            // display math: $$ … $$ or \[ … \] (on their own lines, or on one line)
            if t.hasPrefix("$$") || t.hasPrefix("\\[") {
                flushAll()
                let close = t.hasPrefix("$$") ? "$$" : "\\]"
                var rest = String(t.dropFirst(2))
                if rest.hasSuffix(close), !rest.isEmpty {
                    rest = String(rest.dropLast(2))
                    out.append(.math(rest.trimmingCharacters(in: .whitespaces)))
                    i += 1
                    continue
                }
                var body = rest.isEmpty ? [] : [rest]
                i += 1
                while i < lines.count {
                    let l = lines[i].trimmingCharacters(in: .whitespaces)
                    if l.hasSuffix(close) {
                        let b = String(l.dropLast(2)).trimmingCharacters(in: .whitespaces)
                        if !b.isEmpty { body.append(b) }
                        i += 1
                        break
                    }
                    body.append(l); i += 1
                }
                out.append(.math(body.joined(separator: "\n")))
                continue
            }
            if t.isEmpty {
                flushAll()
                i += 1
                continue
            }
            // heading
            if let h = heading(t) {
                flushAll()
                out.append(.heading(level: h.0, text: inline(h.1)))
                i += 1
                continue
            }
            // table: a row with pipes, then a separator row
            if t.contains("|"), i + 1 < lines.count, let al = separator(lines[i + 1]) {
                flushAll()
                let header = cells(t)
                var rows: [[String]] = []
                i += 2
                while i < lines.count {
                    let r = lines[i].trimmingCharacters(in: .whitespaces)
                    guard r.contains("|"), !r.isEmpty else { break }
                    rows.append(cells(r)); i += 1
                }
                let n = max(header.count, rows.map(\.count).max() ?? 0)
                func pad(_ r: [String]) -> [String] { (r + Array(repeating: "", count: max(0, n - r.count))).map(inline) }
                out.append(.table(header: pad(header), alignments: al + Array(repeating: .leading, count: max(0, n - al.count)),
                                  rows: rows.map(pad)))
                continue
            }
            // rule
            if isRule(t) {
                flushAll()
                out.append(.rule)
                i += 1
                continue
            }
            // block quote
            if t.hasPrefix(">") {
                flushPara(); flushList()
                var q = String(t.dropFirst())
                if q.hasPrefix(" ") { q.removeFirst() }
                quote.append(q)
                i += 1
                continue
            }
            // list item
            if let item = listItem(line) {
                flushPara(); flushQuote()
                items.append(item)
                i += 1
                continue
            }
            // continuation of a list item (indented, or a lazy line right after one)
            if !items.isEmpty, para.isEmpty {
                items[items.count - 1].text += " " + inline(t)
                i += 1
                continue
            }
            if !quote.isEmpty { quote.append(t); i += 1; continue }
            para.append(line)
            i += 1
        }
        flushAll()
        return out
    }

    static func heading(_ t: String) -> (Int, String)? {
        let hashes = t.prefix { $0 == "#" }.count
        guard (1...6).contains(hashes) else { return nil }
        let rest = t.dropFirst(hashes)
        guard rest.isEmpty || rest.first == " " else { return nil }
        var text = rest.trimmingCharacters(in: .whitespaces)
        let closing = text.reversed().prefix { $0 == "#" }.count       // "## Title ##" (but keep "C#")
        if closing > 0, closing == text.count || text.dropLast(closing).hasSuffix(" ") { text.removeLast(closing) }
        return (hashes, text.trimmingCharacters(in: .whitespaces))
    }

    static func isRule(_ t: String) -> Bool {
        let s = t.replacingOccurrences(of: " ", with: "")
        guard s.count >= 3, let c = s.first, "-*_".contains(c) else { return false }
        return s.allSatisfy { $0 == c }
    }

    /// The column alignments of a table separator row (`| :--- | :---: | ---: |`), or nil if it isn't one.
    static func separator(_ line: String) -> [MDAlignment]? {
        let t = line.trimmingCharacters(in: .whitespaces)
        guard t.contains("-") else { return nil }
        let cs = cells(t)
        guard !cs.isEmpty, t.contains("|") || cs.count > 1 else { return nil }
        var out: [MDAlignment] = []
        for c in cs {
            let s = c.trimmingCharacters(in: .whitespaces)
            let core = s.trimmingCharacters(in: CharacterSet(charactersIn: ":"))
            guard core.count >= 1, core.allSatisfy({ $0 == "-" }) else { return nil }
            out.append(s.hasPrefix(":") && s.hasSuffix(":") ? .center : s.hasSuffix(":") ? .trailing : .leading)
        }
        return out
    }

    /// Cells of a table row; escaped pipes (`\|`) and pipes inside `code` stay in the cell.
    static func cells(_ row: String) -> [String] {
        var t = row.trimmingCharacters(in: .whitespaces)
        if t.hasPrefix("|") { t.removeFirst() }
        if t.hasSuffix("|"), !t.hasSuffix("\\|") { t.removeLast() }
        var out: [String] = []
        var cur = ""
        var inCode = false
        var prev: Character = " "
        for ch in t {
            if ch == "`" { inCode.toggle() }
            if ch == "|", !inCode, prev != "\\" {
                out.append(cur.trimmingCharacters(in: .whitespaces)); cur = ""
            } else if ch == "|", prev == "\\" {
                cur.removeLast(); cur.append(ch)
            } else {
                cur.append(ch)
            }
            prev = ch
        }
        out.append(cur.trimmingCharacters(in: .whitespaces))
        return out
    }

    static func listItem(_ line: String) -> MDListItem? {
        let indent = line.prefix { $0 == " " || $0 == "\t" }.reduce(0) { $0 + ($1 == "\t" ? 4 : 1) }
        let t = line.trimmingCharacters(in: .whitespaces)
        var marker: String
        var rest: Substring
        if let c = t.first, "-*+".contains(c), t.dropFirst().first == " " {
            marker = "•"
            rest = t.dropFirst(2)
        } else {
            let digits = t.prefix { $0.isNumber }
            guard (1...9).contains(digits.count) else { return nil }
            let after = t.dropFirst(digits.count)
            guard let p = after.first, p == "." || p == ")", after.dropFirst().first == " " else { return nil }
            marker = String(digits) + String(p)
            rest = after.dropFirst(2)
        }
        var checked: Bool?
        if rest.hasPrefix("[ ] ") || rest.hasPrefix("[x] ") || rest.hasPrefix("[X] ") {
            checked = !rest.hasPrefix("[ ]")
            rest = rest.dropFirst(4)
        }
        if checked != nil, marker == "•" { marker = "" }
        return MDListItem(depth: indent, marker: marker, checked: checked, text: inline(String(rest)))
    }

    /// Inline rewrites the renderer's AttributedString parser doesn't do: `\( x \)` -> `x` as code.
    static func inline(_ s: String) -> String {
        guard s.contains("\\(") else { return s }
        var out = ""
        var rest = Substring(s)
        while let open = rest.range(of: "\\("), let close = rest[open.upperBound...].range(of: "\\)") {
            out += rest[..<open.lowerBound]
            out += "`" + rest[open.upperBound..<close.lowerBound].trimmingCharacters(in: .whitespaces) + "`"
            rest = rest[close.upperBound...]
        }
        return out + rest
    }
}
