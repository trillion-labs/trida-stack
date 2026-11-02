import Foundation

/// Finds the files in a folder most likely to hold what a question asks about, with the decision model
/// instead of an embedding index (DecisionRAG, arXiv:2610.35868). Every file is described by its path and
/// opening lines; descriptions go 40 to a request, listed *before* the question, so the server's decision
/// state cache reuses them across searches of the same folder (only the question is read again). Each
/// request is one multiple-choice question that keeps its top 5; the pooled files plus keyword search's
/// top 10 go to one final choice. This is `flatc+bm25` from inference/mlx/filesearch/eval.py, which put the
/// answer file in the top 5 for 94% of its 35 questions about this repository (keyword search alone: 63%).
public struct FileLocator: Sendable {
    public struct Entry: Sendable, Equatable {
        public var path: String       // relative to the folder searched
        public var summary: String
        public var text: String       // the file's start, for keyword search
    }

    public struct Hit: Sendable, Equatable {
        public var path: String
        public var summary: String
        public var p: Double
        /// Added by keyword search rather than picked from its batch by the decision model.
        public var fromKeywords: Bool
    }

    public struct Result: Sendable {
        public var hits: [Hit]
        public var searched: Int
        public var total: Int
        public var requests: Int
    }

    public typealias Decide = @Sendable (OrderedJSON, [DecisionQuestion]) async throws -> DecisionResult

    public static let task = "Find where in this file tree the answer to the question is."
    public static let question = "Which candidate most likely contains the information needed to answer the question?"
    static let skipDirs: Set<String> = ["node_modules", "__pycache__", ".venv", "venv", "build", "dist", ".build",
                                        "DerivedData", "Pods", ".git"]

    public var perRequest = 40
    public var keepPerBatch = 5
    public var finalSize = 20
    public var keywordExtra = 10
    /// More files than this: only keyword search's top `maxFiles` are judged (the result says so).
    public var maxFiles = 480

    public init() {}

    // ---- index ----------------------------------------------------------------------------------------

    /// Text files under `dir` (hidden files and build/dependency folders skipped), sorted by path so the
    /// batches, and therefore the server's cached prefixes, are the same from one search to the next.
    public static func index(_ dir: URL, limit: Int = 20_000, readBytes: Int = 200_000) -> [Entry] {
        let fm = FileManager.default
        let base = dir.standardizedFileURL.resolvingSymlinksInPath().path
        let prefix = base.hasSuffix("/") ? base : base + "/"
        guard let en = fm.enumerator(at: dir, includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey, .fileSizeKey],
                                     options: [.skipsHiddenFiles, .skipsPackageDescendants]) else { return [] }
        var out: [Entry] = []
        for case let f as URL in en {
            let v = try? f.resourceValues(forKeys: [.isRegularFileKey, .isDirectoryKey, .fileSizeKey])
            if v?.isDirectory == true {
                if skipDirs.contains(f.lastPathComponent) { en.skipDescendants() }
                continue
            }
            guard v?.isRegularFile == true, let h = try? FileHandle(forReadingFrom: f) else { continue }
            let data = (try? h.read(upToCount: readBytes)) ?? Data()
            try? h.close()
            if data.prefix(4096).contains(0) { continue }          // binary
            let text = String(decoding: data, as: UTF8.self)
            let full = f.standardizedFileURL.resolvingSymlinksInPath().path
            let rel = full.hasPrefix(prefix) ? String(full.dropFirst(prefix.count)) : f.lastPathComponent
            out.append(Entry(path: rel, summary: summary(text), text: text))
            if out.count >= limit { break }
        }
        return out.sorted { $0.path < $1.path }
    }

    static let noise = ["#!", "import ", "from ", "@", "copyright", "spdx", "license", "// swift-tools", "<?xml",
                        "{", "}", "[", "]", "```"]

    static func isNoise(_ s: String) -> Bool {
        let l = s.lowercased()
        return l == "---" || noise.contains { l.hasPrefix($0) }
    }

    /// A file's first meaningful lines: docstring, heading, leading comments or first statements.
    public static func summary(_ text: String, limit: Int = 220) -> String {
        var parts: [String] = []
        var n = 0
        for raw in text.split(separator: "\n", omittingEmptySubsequences: false).prefix(60) {
            let line = raw.trimmingCharacters(in: .whitespaces)
            var s = line.trimmingCharacters(in: CharacterSet(charactersIn: "\"'"))
            while let c = s.first, "#/*-!; ".contains(c) { s.removeFirst() }
            s = s.trimmingCharacters(in: .whitespaces)
            if s.isEmpty || isNoise(line) || isNoise(s) { continue }
            parts.append(s)
            n += s.count
            if n > limit { break }
        }
        let s = parts.joined(separator: " ")
        if s.isEmpty { return "(empty)" }
        return s.count > limit ? String(s.prefix(limit)) + "…" : s
    }

    public static func describe(_ e: Entry) -> String { "file \(e.path) — \(e.summary)" }

    // ---- keyword search (BM25) ---------------------------------------------------------------------------

    static let tokenRE = try! NSRegularExpression(pattern: "[a-z0-9]+|[가-힣]+")
    static let camelRE = try! NSRegularExpression(pattern: "([a-z])([A-Z])")

    static func tokens(_ s: String) -> [String] {
        let split = camelRE.stringByReplacingMatches(in: s, range: NSRange(s.startIndex..., in: s), withTemplate: "$1 $2")
            .lowercased()
        return tokenRE.matches(in: split, range: NSRange(split.startIndex..., in: split)).compactMap {
            Range($0.range, in: split).map { String(split[$0]) }
        }
    }

    /// Entries ranked by BM25 over path + text (k1 1.2, b 0.75), best first.
    public static func keywordRank(_ query: String, _ entries: [Entry]) -> [Int] {
        let docs = entries.map { e -> [String: Int] in
            var c: [String: Int] = [:]
            for t in tokens(e.path.replacingOccurrences(of: "/", with: " ") + " " + e.text) { c[t, default: 0] += 1 }
            return c
        }
        let lens = docs.map { Double($0.values.reduce(0, +)) }
        let avg = max(lens.reduce(0, +) / Double(max(docs.count, 1)), 1)
        var df: [String: Int] = [:]
        for d in docs { for w in d.keys { df[w, default: 0] += 1 } }
        let n = Double(docs.count)
        let q = Set(tokens(query))
        let scores = docs.enumerated().map { i, d -> Double in
            q.reduce(0) { acc, w in
                guard let tf = d[w].map(Double.init), let dfw = df[w] else { return acc }
                let idf = log(1 + (n - Double(dfw) + 0.5) / (Double(dfw) + 0.5))
                return acc + idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * lens[i] / avg))
            }
        }
        return scores.indices.sorted { scores[$0] != scores[$1] ? scores[$0] > scores[$1] : $0 < $1 }
    }

    // ---- the search ------------------------------------------------------------------------------------

    public static func state(_ query: String, _ items: [Entry]) -> OrderedJSON {
        let list = items.enumerated().map { "[\($0.offset + 1)] \(describe($0.element))" }.joined(separator: "\n")
        return .object([("candidates", .string(list)), ("task", .string(task)), ("question", .string(query))])
    }

    /// One choice question over `items` (options are just their numbers; the descriptions are in the state).
    func choose(_ query: String, _ items: [Entry], decide: Decide) async throws -> [Double] {
        let q = DecisionQuestion.choice("best", Self.question, items.indices.map { (String($0 + 1), "") })
        let body = Decision.body(state: Self.state(query, items), questions: [q],
                                 settings: DecisionSettings(reasoning: "never"))
        let r = try await decide(body, [q])
        guard let a = r.answers["best"] else { throw ChatClientError.http(200, "no answer from the decision model") }
        var p = [Double](repeating: 0, count: items.count)
        for o in a.options { if let i = Int(o.key), (1...items.count).contains(i) { p[i - 1] = o.p } }
        return p
    }

    public func locate(_ query: String, in entries: [Entry], top: Int = 5, decide: Decide,
                       progress: (String) -> Void = { _ in }) async throws -> Result {
        guard !entries.isEmpty else { return Result(hits: [], searched: 0, total: 0, requests: 0) }
        let ranked = Self.keywordRank(query, entries)
        let candidates = entries.count <= maxFiles ? Array(entries.indices)
            : Array(ranked.prefix(maxFiles)).sorted { entries[$0].path < entries[$1].path }
        var pool: [(Int, Double)] = []
        var requests = 0
        let batches = stride(from: 0, to: candidates.count, by: perRequest).map {
            Array(candidates[$0..<min($0 + perRequest, candidates.count)])
        }
        for (b, batch) in batches.enumerated() {
            progress("Reading \(candidates.count) files (batch \(b + 1) of \(batches.count))…")
            if batch.count == 1 { pool.append((batch[0], 1)); continue }
            let p = try await choose(query, batch.map { entries[$0] }, decide: decide)
            requests += 1
            pool += zip(batch, p).sorted { $0.1 > $1.1 }.prefix(keepPerBatch).map { ($0.0, $0.1) }
        }
        var finalists = pool.sorted { $0.1 > $1.1 }.prefix(finalSize).map(\.0)
        let fromKeywords = ranked.prefix(keywordExtra).filter { !finalists.contains($0) }
        finalists += fromKeywords
        progress("Comparing the \(finalists.count) best candidates…")
        var p: [Double] = [1]
        if finalists.count > 1 {
            p = try await choose(query, finalists.map { entries[$0] }, decide: decide)
            requests += 1
        }
        let hits = zip(finalists, p).sorted { $0.1 > $1.1 }.prefix(top).map {
            Hit(path: entries[$0.0].path, summary: entries[$0.0].summary, p: $0.1, fromKeywords: fromKeywords.contains($0.0))
        }
        return Result(hits: hits, searched: candidates.count, total: entries.count, requests: requests)
    }

    /// The tool result the model reads.
    public static func report(_ query: String, _ r: Result, folder: String) -> String {
        guard !r.hits.isEmpty else { return "(no files under \(folder))" }
        var s = "Files most likely to answer \"\(query)\" (decision model over \(r.searched) files in \(folder)):\n"
        for (i, h) in r.hits.enumerated() {
            let path = folder == "." ? h.path : (folder as NSString).appendingPathComponent(h.path)
            s += "\(i + 1). \(path) (\(String(format: "%.2f", h.p))) — \(h.summary)\n"
        }
        if r.searched < r.total {
            s += "Only the \(r.searched) of \(r.total) files that keyword search ranked highest were judged; "
                + "pass a narrower path to judge every file.\n"
        }
        return s + "Open the most likely file with read_file to confirm before answering."
    }
}
