import Foundation

/// fetch_url: a web page as readable text (scripts, styles and tags removed), or a JSON / plain-text response
/// as is. GET only, http(s) only, at most `maxBytes`.
public enum WebFetch {
    public static let maxBytes = 4_000_000
    public static let timeout: TimeInterval = 25

    enum FetchError: LocalizedError {
        case badURL(String), notText(String), http(Int)
        var errorDescription: String? {
            switch self {
            case let .badURL(s): return "not an http(s) URL: \(s)"
            case let .notText(t): return "the page is not text (\(t)); download it with run_shell instead"
            case let .http(c): return "the server answered HTTP \(c)"
            }
        }
    }

    /// The words a search or encyclopedia URL looks up (its `query` / `q` / `search` / `wd` parameter, or the
    /// title after /wiki/ or /w/), decoded and lowercased; nil for other pages.
    public static func searchWords(_ s: String) -> [String]? {
        guard let u = url(s), let c = URLComponents(url: u, resolvingAgainstBaseURL: false) else { return nil }
        var text: String?
        if let q = c.queryItems?.first(where: { ["query", "q", "search", "wd", "p"].contains($0.name.lowercased()) }) {
            text = q.value
        } else {
            let parts = c.path.split(separator: "/")
            if parts.count >= 2, ["wiki", "w"].contains(parts[parts.count - 2].lowercased()) {
                text = String(parts.last!).removingPercentEncoding ?? String(parts.last!)
            }
        }
        guard let t = text?.replacingOccurrences(of: "+", with: " ").replacingOccurrences(of: "_", with: " ") else {
            return nil
        }
        let words = t.lowercased().split { !$0.isLetter && !$0.isNumber }.map(String.init).filter { $0.count >= 2 }
        return words.isEmpty ? nil : words
    }

    /// Whether a search shares no word with what the user wrote (a name searched in another spelling: the user
    /// wrote a Korean school name and the search is for a romanized guess at it). A word counts when the user's
    /// text contains it, so Korean words with particles attached still match.
    /// Only searches (a query parameter): an encyclopedia title in English is a fine source for a Korean question.
    public static func searchMissesUserWords(_ s: String, userText: String) -> Bool {
        guard let u = url(s), let c = URLComponents(url: u, resolvingAgainstBaseURL: false),
              c.queryItems?.contains(where: { ["query", "q", "search", "wd", "p"].contains($0.name.lowercased()) }) == true,
              let words = searchWords(s) else { return false }
        let said = userText.lowercased()
        return !words.contains { said.contains($0) }
    }

    public static func url(_ s: String) -> URL? {
        let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let u = URL(string: t.contains("://") ? t : "https://" + t),
              let scheme = u.scheme?.lowercased(), scheme == "http" || scheme == "https", u.host != nil else { return nil }
        return u
    }

    /// localhost and private, link-local or loopback addresses (the user's own machine and network).
    public static func isPrivate(_ u: URL) -> Bool {
        guard let h = u.host?.lowercased() else { return false }
        if h == "localhost" || h.hasSuffix(".local") || h.hasSuffix(".localhost") || h == "::1" || h == "0.0.0.0" { return true }
        let p = h.split(separator: ".").compactMap { Int($0) }
        guard p.count == 4 else { return false }
        return p[0] == 10 || p[0] == 127 || (p[0] == 172 && (16...31).contains(p[1])) || (p[0] == 192 && p[1] == 168)
            || (p[0] == 169 && p[1] == 254)
    }

    /// Downloads `s` and reads it into a `WebPage` (see `WebBrowser` for how the model navigates it).
    public static func fetch(_ s: String) async throws -> WebPage {
        guard var u = url(s) else { throw FetchError.badURL(s) }
        // App Transport Security only allows plain http on the local network: ask for https elsewhere
        if u.scheme?.lowercased() == "http", !isPrivate(u), var c = URLComponents(url: u, resolvingAgainstBaseURL: false) {
            c.scheme = "https"
            u = c.url ?? u
        }
        var req = URLRequest(url: u, timeoutInterval: timeout)
        req.setValue("Zoro-1-Chat/0.4 (on-device assistant; macOS)", forHTTPHeaderField: "User-Agent")
        req.setValue("text/html,application/json,text/plain;q=0.9,*/*;q=0.5", forHTTPHeaderField: "Accept")
        let (data, resp) = try await URLSession.shared.data(for: req)
        let http = resp as? HTTPURLResponse
        if let c = http?.statusCode, !(200..<300).contains(c) { throw FetchError.http(c) }
        let type = (http?.value(forHTTPHeaderField: "Content-Type") ?? "").lowercased()
        let body = String(decoding: data.prefix(maxBytes), as: UTF8.self)
        let final = resp.url?.absoluteString ?? u.absoluteString
        let isHTML = type.contains("html") || (type.isEmpty && body.range(of: "<html", options: .caseInsensitive) != nil)
        let base = URL(string: final) ?? u
        if isHTML {
            let title = firstMatch(#"(?is)<title[^>]*>(.*?)</title>"#, in: body).map(decodeEntities)?
                .trimmingCharacters(in: .whitespacesAndNewlines) ?? final
            let main = mainContent(body)
            return WebPage(url: final, title: title, text: text(fromHTML: main, isMain: true),
                           links: links(in: main, base: base))
        }
        guard type.isEmpty || type.contains("json") || type.contains("text") || type.contains("xml") else {
            throw FetchError.notText(type)
        }
        return WebPage(url: final, title: final, text: body, links: [])
    }

    /// The links in a page's content, in order, without duplicates, page-internal anchors, scripts, mail
    /// addresses or (on Wikipedia) special pages such as File: or Help:.
    public static func links(in html: String, base: URL, limit: Int = 600) -> [WebPage.Link] {
        guard let re = try? NSRegularExpression(pattern: #"(?is)<a\b[^>]*?href="([^"]+)"[^>]*>(.*?)</a>"#) else { return [] }
        var out: [WebPage.Link] = []
        var seen = Set<String>()
        for m in re.matches(in: html, range: NSRange(html.startIndex..., in: html)) {
            guard let hr = Range(m.range(at: 1), in: html), let tr = Range(m.range(at: 2), in: html) else { continue }
            let href = decodeEntities(String(html[hr]))
            if href.hasPrefix("#") || href.hasPrefix("javascript:") || href.hasPrefix("mailto:") { continue }
            guard let u = URL(string: href, relativeTo: base)?.absoluteURL,
                  let scheme = u.scheme?.lowercased(), scheme == "http" || scheme == "https" else { continue }
            if u.host?.hasSuffix("wikipedia.org") == true,
               u.path.hasPrefix("/wiki/") ? u.path.dropFirst(6).contains(":") : true { continue }
            let text = decodeEntities(String(html[tr]).replacingOccurrences(of: #"(?s)<[^>]+>"#, with: "",
                                                                             options: .regularExpression))
                .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            var key = u.absoluteString
            if let hash = key.firstIndex(of: "#") { key = String(key[..<hash]) }
            guard !text.isEmpty, !text.hasPrefix("["), seen.insert(key).inserted else { continue }
            out.append(WebPage.Link(text: text, url: key))
            if out.count >= limit { break }
        }
        return out
    }

    static func firstMatch(_ pattern: String, in s: String) -> String? {
        guard let re = try? NSRegularExpression(pattern: pattern),
              let m = re.firstMatch(in: s, range: NSRange(s.startIndex..., in: s)), m.numberOfRanges > 1,
              let r = Range(m.range(at: 1), in: s) else { return nil }
        return String(s[r])
    }

    /// Readable text of an HTML page: the main content (see `mainContent`) without scripts, styles, menus,
    /// citation marks and tags; block elements become line breaks, list items "- ", table cells tabs, entities
    /// decoded, blank runs collapsed.
    public static func text(fromHTML html: String, isMain: Bool = false) -> String {
        var s = isMain ? html : mainContent(html)
        func sub(_ p: String, _ r: String) { s = s.replacingOccurrences(of: p, with: r, options: .regularExpression) }
        sub(#"(?is)<(script|style|noscript|svg|template|iframe|button|select|form)\b.*?</\1\s*>"#, " ")
        sub(#"(?is)<!--.*?-->"#, " ")
        sub(#"(?is)<(nav|footer|header|aside)\b.*?</\1\s*>"#, " ")
        sub(#"(?is)<sup\b[^>]*class="[^"]*reference[^"]*"[^>]*>.*?</sup>"#, "")     // Wikipedia's [1] marks
        // headings as "## Title" lines: the sections read_section finds
        for level in 1...4 {
            sub(#"(?is)<h\#(level)\b[^>]*>(.*?)</h\#(level)\s*>"#, "\n" + String(repeating: "#", count: level) + " $1\n")
        }
        sub(#"(?i)<li\b[^>]*>"#, "\n- ")
        sub(#"(?i)<br\s*/?>|</(p|div|h[1-6]|tr|ul|ol|table|section|article|blockquote|pre|dd|dt|caption)\s*>"#, "\n")
        sub(#"(?i)</t[dh]\s*>"#, "\t")
        sub(#"(?s)<[^>]+>"#, "")
        s = decodeEntities(s)
        sub(#"\[\s*edit\s*\]"#, "")                                      // Wikipedia's section edit links
        // ICU spells a code point \x{...}; whitespace-only lines become empty, then blank runs collapse
        sub(#"[ \t\x{00A0}\x{200B}]+"#, " ")
        sub(#" *\n[ \n]*\n *"#, "\n\n")
        sub(#" *\n *"#, "\n")
        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// The part of the page with the content: Wikipedia's article body (up to its reference list), else
    /// `<main>`, `<article>` or the `role="main"` element, else the body. Menus, the language list and the page
    /// chrome around the content are left out.
    static func mainContent(_ html: String) -> String {
        // markers can sit inside a tag (an attribute): start and cut at that tag's "<"
        func from(_ marker: String) -> Substring? {
            guard let r = html.range(of: marker, options: .caseInsensitive) else { return nil }
            if marker.hasPrefix("<") { return html[r.lowerBound...] }
            return html[(html[..<r.lowerBound].lastIndex(of: "<") ?? r.lowerBound)...]
        }
        func cut(_ s: Substring, at markers: [String]) -> Substring {
            let ends = markers.compactMap { m -> String.Index? in
                guard let r = s.range(of: m, options: .caseInsensitive) else { return nil }
                return m.hasPrefix("<") ? r.lowerBound : s[..<r.lowerBound].lastIndex(of: "<") ?? r.lowerBound
            }
            return ends.min().map { s[..<$0] } ?? s
        }
        if let w = from(#"id="mw-content-text""#) {
            return String(cut(w, at: [#"class="reflist"#, #"class="mw-references-wrap"#, #"class="printfooter""#,
                                      #"id="catlinks""#, "</main>"]))
        }
        if let m = from("<main") { return String(cut(m, at: ["</main>"])) }
        if let a = from("<article") { return String(cut(a, at: ["</article>"])) }
        if let r = from(#"role="main""#) { return String(r) }
        if let b = from("<body") { return String(b) }
        return html
    }

    static let named: [String: String] = ["amp": "&", "lt": "<", "gt": ">", "quot": "\"", "apos": "'", "#39": "'",
                                          "nbsp": " ", "ndash": "–", "mdash": "—", "hellip": "…", "middot": "·",
                                          "copy": "©", "reg": "®", "rsquo": "’", "lsquo": "‘", "rdquo": "”", "ldquo": "“"]

    public static func decodeEntities(_ s: String) -> String {
        guard s.contains("&") else { return s }
        var out = ""
        var i = s.startIndex
        while i < s.endIndex {
            if s[i] == "&", let semi = s[i...].prefix(12).firstIndex(of: ";") {
                let name = String(s[s.index(after: i)..<semi])
                var rep: String?
                if name.hasPrefix("#x") || name.hasPrefix("#X") {
                    rep = UInt32(name.dropFirst(2), radix: 16).flatMap { Unicode.Scalar($0) }.map { String(Character($0)) }
                } else if name.hasPrefix("#") {
                    rep = UInt32(name.dropFirst()).flatMap { Unicode.Scalar($0) }.map { String(Character($0)) }
                } else {
                    rep = named[name]
                }
                if let rep {
                    out += rep
                    i = s.index(after: semi)
                    continue
                }
            }
            out.append(s[i])
            i = s.index(after: i)
        }
        return out
    }
}
