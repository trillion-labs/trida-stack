import Foundation

/// A web page as the model reads it: its readable text (headings as "## …" lines) and its links.
public struct WebPage: Sendable {
    public struct Link: Sendable, Equatable {
        public var text: String
        public var url: String
    }

    public var url: String
    public var title: String
    public var text: String
    public var links: [Link]

    public init(url: String, title: String, text: String, links: [Link]) {
        self.url = url
        self.title = title
        self.text = text
        self.links = links
    }

    public struct Section: Sendable {
        public var level: Int
        public var title: String
        public var range: Range<String.Index>      // the heading line and everything up to the next heading at its level or above
    }

    /// The "## …" sections, in order.
    public var sections: [Section] {
        var heads: [(Int, String, String.Index)] = []
        var i = text.startIndex
        while i < text.endIndex {
            let end = text[i...].firstIndex(of: "\n") ?? text.endIndex
            let line = text[i..<end]
            if line.hasPrefix("#") {
                let level = line.prefix { $0 == "#" }.count
                let title = line.dropFirst(level).trimmingCharacters(in: .whitespaces)
                if (1...6).contains(level), !title.isEmpty { heads.append((level, title, i)) }
            }
            i = end < text.endIndex ? text.index(after: end) : end
        }
        return heads.enumerated().map { k, h in
            let next = heads[(k + 1)...].first { $0.0 <= h.0 }?.2 ?? text.endIndex
            return Section(level: h.0, title: h.1, range: h.2..<next)
        }
    }

    /// What fetch_url returns: title, address, the section list, the link count and the start of the text
    /// (on Wikipedia: the infobox and the lead), with how to read further.
    public func outline(preview: Int = 2500) -> String {
        let secs = sections.filter { $0.level >= 2 }.map(\.title)
        var out = "\(title)\n\(url)\n"
        if !secs.isEmpty { out += "Sections: " + secs.prefix(40).joined(separator: " · ") + (secs.count > 40 ? " …" : "") + "\n" }
        if !links.isEmpty { out += "Links: \(links.count) (open one with open_link)\n" }
        out += "\n" + String(text.prefix(preview))
        if text.count > preview {
            out += "\n\n[… \(text.count - preview) more characters. find_in_page(query) finds a fact on this page, "
                + "read_section(heading) reads a section, open_link(link) follows a link.]"
        }
        return out
    }

    /// The places `query` occurs (case- and accent-insensitive), with `window` characters around each and the
    /// section it is in; when the whole query does not occur, its longest words are tried one by one. Links whose
    /// text matches are listed too.
    public func find(_ query: String, maxHits: Int = 8, window: Int = 280) -> String {
        let q = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !q.isEmpty else { return "error: missing argument 'query'" }
        let opts: String.CompareOptions = [.caseInsensitive, .diacriticInsensitive]
        func ranges(_ needle: String) -> [Range<String.Index>] {
            var out: [Range<String.Index>] = []
            var from = text.startIndex
            while let r = text.range(of: needle, options: opts, range: from..<text.endIndex) {
                out.append(r)
                from = r.upperBound
            }
            return out
        }
        var used = q
        var hits = ranges(q)
        if hits.isEmpty {
            for w in q.split(separator: " ").map(String.init).filter({ $0.count >= 3 }).sorted(by: { $0.count > $1.count }) {
                let h = ranges(w)
                if !h.isEmpty { hits = h; used = w; break }
            }
        }
        let linkHits = links.enumerated().filter { $0.element.text.range(of: q, options: opts) != nil }.prefix(8)
        if hits.isEmpty && linkHits.isEmpty { return "No match for \"\(q)\" on \(title)." }
        let secs = sections
        var out = hits.isEmpty ? "" : "\(hits.count) match\(hits.count == 1 ? "" : "es") for \"\(used)\" on \(title)"
            + (used == q ? "" : " (\"\(q)\" itself does not occur)") + (hits.count > maxHits ? "; the first \(maxHits):" : ":")
        var lastEnd = text.startIndex
        for r in hits.prefix(maxHits) {
            let lo = text.index(r.lowerBound, offsetBy: -window, limitedBy: text.startIndex) ?? text.startIndex
            let hi = text.index(r.upperBound, offsetBy: window, limitedBy: text.endIndex) ?? text.endIndex
            let start = max(lo, lastEnd)
            guard start < hi else { continue }                  // already shown with the previous match
            lastEnd = hi
            let sec = secs.last { $0.range.contains(r.lowerBound) && $0.level >= 2 }?.title
            let snippet = text[start..<hi].replacingOccurrences(of: "\n", with: " ⏎ ")
            out += "\n\n" + (sec.map { "[\($0)] " } ?? "") + (start > text.startIndex ? "…" : "") + snippet
                + (hi < text.endIndex ? "…" : "")
        }
        if !linkHits.isEmpty {
            out += (out.isEmpty ? "" : "\n\n") + "Links: "
                + linkHits.map { "[\($0.offset + 1)] \($0.element.text)" }.joined(separator: ", ")
        }
        return out
    }

    /// A section by heading (exact, then the first containing the words), at most `maxChars`.
    public func section(_ heading: String, maxChars: Int = 12_000) -> String {
        let h = heading.trimmingCharacters(in: .whitespacesAndNewlines).trimmingCharacters(in: CharacterSet(charactersIn: "#"))
            .trimmingCharacters(in: .whitespaces)
        let secs = sections
        let opts: String.CompareOptions = [.caseInsensitive, .diacriticInsensitive]
        guard let s = secs.first(where: { $0.title.compare(h, options: opts) == .orderedSame })
                ?? secs.first(where: { $0.title.range(of: h, options: opts) != nil }) else {
            return "No section \"\(h)\" on \(title). Sections: "
                + secs.filter { $0.level >= 2 }.map(\.title).prefix(60).joined(separator: " · ")
        }
        let body = text[s.range]
        return body.count > maxChars
            ? String(body.prefix(maxChars)) + "\n\n[… the section goes on for \(body.count - maxChars) characters; "
                + "read_section one of its subsections, or find_in_page.]"
            : String(body)
    }

    /// A link by its number in `links` (1-based), its text (exact, then containing), or an address.
    public func link(_ ref: String) -> Link? {
        let r = ref.trimmingCharacters(in: .whitespacesAndNewlines).trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
        if let n = Int(r), links.indices.contains(n - 1) { return links[n - 1] }
        let opts: String.CompareOptions = [.caseInsensitive, .diacriticInsensitive]
        if let l = links.first(where: { $0.text.compare(r, options: opts) == .orderedSame }) { return l }
        if let l = links.first(where: { $0.text.range(of: r, options: opts) != nil }) { return l }
        if r.contains("/"), WebFetch.url(r) != nil { return Link(text: r, url: r) }
        return nil
    }
}

/// One chat's web browsing: the page find_in_page, read_section and open_link work on (the last one opened).
public final class WebSession: @unchecked Sendable {
    private static let lock = NSLock()
    private static var sessions: [String: WebSession] = [:]

    public static func session(_ id: String) -> WebSession {
        lock.lock()
        defer { lock.unlock() }
        if let s = sessions[id] { return s }
        let s = WebSession()
        sessions[id] = s
        return s
    }

    private let lock = NSLock()
    private var _current: WebPage?

    public var current: WebPage? {
        lock.lock()
        defer { lock.unlock() }
        return _current
    }

    public func open(_ page: WebPage) {
        lock.lock()
        _current = page
        lock.unlock()
    }
}
