import Foundation

/// Finds the files in a folder most likely to hold what a question asks about, with the decision model
/// instead of an embedding index (DecisionRAG, arXiv:2610.35868). Every file is described by its path and
/// opening lines; descriptions go 40 to a request, listed *before* the question, so the server's decision
/// state cache reuses them across searches of the same folder (only the question is read again). Each
/// request is one multiple-choice question that keeps its top 5; the pooled files plus keyword search's
/// top 10 go to one final choice. This is `flatc+bm25` from inference/zoro-1/mlx/filesearch/eval.py, which put the
/// answer file in the top 5 for 94% of its 35 questions about this repository (keyword search alone: 63%).
///
/// Folders with more than `maxFiles` files get a folder step first (as nav2 in the eval): the folder's
/// subfolders, split until each is small enough and described by their main files, are judged the same way,
/// and the files of the best ones go on to the file step. File summaries and keyword counts are kept in an
/// index on disk (by path, size and modification time), so a large folder is read once.
public struct FileLocator: Sendable {
    public struct Entry: Sendable, Equatable {
        public var path: String       // relative to the folder searched
        public var summary: String
        public var terms: [String: Int]   // keyword counts of the path and the file's start
        public var size: Int = 0

        public init(path: String, summary: String, text: String, size: Int = 0) {
            self.init(path: path, summary: summary, terms: FileLocator.termCounts(path, text), size: size)
        }

        public init(path: String, summary: String, terms: [String: Int], size: Int = 0) {
            self.path = path
            self.summary = summary
            self.terms = terms
            self.size = size
        }
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
        /// Large folders: the subfolders the folder step picked (empty when every file was judged).
        public var folders: [String] = []
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
    /// batches, and therefore the server's cached prefixes, are the same from one search to the next. A file
    /// whose size and modification time match the on-disk index (`cache`) is not read again.
    public static func index(_ dir: URL, limit: Int = 20_000, readBytes: Int = 64_000,
                             cache: URL? = FileLocator.defaultIndexURL) -> [Entry] {
        let fm = FileManager.default
        let base = dir.standardizedFileURL.resolvingSymlinksInPath().path
        let prefix = base.hasSuffix("/") ? base : base + "/"
        guard let en = fm.enumerator(at: dir, includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey,
                                                                           .fileSizeKey, .contentModificationDateKey],
                                     options: [.skipsHiddenFiles, .skipsPackageDescendants]) else { return [] }
        var stored = cache.map(IndexStore.load) ?? [:]
        var changed = false
        var out: [Entry] = []
        for case let f as URL in en {
            let v = try? f.resourceValues(forKeys: [.isRegularFileKey, .isDirectoryKey, .fileSizeKey,
                                                    .contentModificationDateKey])
            if v?.isDirectory == true {
                if skipDirs.contains(f.lastPathComponent) { en.skipDescendants() }
                continue
            }
            guard v?.isRegularFile == true else { continue }
            let full = f.standardizedFileURL.resolvingSymlinksInPath().path
            let rel = full.hasPrefix(prefix) ? String(full.dropFirst(prefix.count)) : f.lastPathComponent
            let size = v?.fileSize ?? 0
            let mtime = v?.contentModificationDate?.timeIntervalSince1970 ?? 0
            if let e = stored[full], e.size == size, e.mtime == mtime {
                if !e.binary { out.append(Entry(path: rel, summary: e.summary, terms: e.terms, size: size)) }
            } else {
                guard let h = try? FileHandle(forReadingFrom: f) else { continue }
                let data = (try? h.read(upToCount: readBytes)) ?? Data()
                try? h.close()
                let binary = data.prefix(4096).contains(0)
                let text = binary ? "" : String(decoding: data, as: UTF8.self)
                let entry = Entry(path: rel, summary: summary(text), text: text, size: size)
                stored[full] = IndexStore.Item(size: size, mtime: mtime, binary: binary, summary: entry.summary,
                                               terms: binary ? [:] : Self.topTerms(entry.terms))
                changed = true
                if !binary { out.append(entry) }
            }
            if out.count >= limit { break }
        }
        if changed, let cache { IndexStore.save(stored, to: cache) }
        return out.sorted { $0.path < $1.path }
    }

    public static var defaultIndexURL: URL {
        FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Zoro-1 Chat/file-index.json")
    }

    /// The on-disk index: summary and keyword counts per absolute path, valid while size and mtime match.
    enum IndexStore {
        struct Item: Codable {
            var size: Int
            var mtime: Double
            var binary: Bool
            var summary: String
            var terms: [String: Int]
        }

        static func load(_ url: URL) -> [String: Item] {
            guard let d = try? Data(contentsOf: url) else { return [:] }
            return (try? JSONDecoder().decode([String: Item].self, from: d)) ?? [:]
        }

        static func save(_ items: [String: Item], to url: URL) {
            try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            if let d = try? JSONEncoder().encode(items) { try? d.write(to: url, options: .atomic) }
        }
    }

    /// The 200 most frequent terms (what the index keeps per file; keyword search needs no more).
    static func topTerms(_ t: [String: Int]) -> [String: Int] {
        t.count <= 200 ? t : Dictionary(uniqueKeysWithValues: t.sorted { $0.value > $1.value }.prefix(200).map { ($0.key, $0.value) })
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

    public static func describe(_ e: Entry) -> String {
        (e.path.hasSuffix("/") || e.path.hasPrefix("(") ? "folder " : "file ") + "\(e.path) — \(e.summary)"
    }

    // ---- folder step (large folders) ---------------------------------------------------------------------

    /// Subfolders to judge when there are too many files: the top-level folders (and the files directly in
    /// the folder, as one group), with any group above `maxFiles` split into its own subfolders.
    public static func groups(_ entries: [Entry], maxFiles: Int) -> [(name: String, members: [Int])] {
        func split(_ prefix: String, _ idx: [Int], depth: Int) -> [(name: String, members: [Int])] {
            var byChild: [String: [Int]] = [:]
            var own: [Int] = []
            for i in idx {
                let rest = entries[i].path.dropFirst(prefix.count)
                if let slash = rest.firstIndex(of: "/") {
                    byChild[prefix + rest[..<slash] + "/", default: []].append(i)
                } else {
                    own.append(i)
                }
            }
            var out: [(name: String, members: [Int])] = []
            if !own.isEmpty { out.append((prefix.isEmpty ? "(files at the top level)" : "(files directly in \(prefix))", own)) }
            for (name, m) in byChild.sorted(by: { $0.key < $1.key }) {
                if m.count > maxFiles, depth < 6 { out += split(name, m, depth: depth + 1) } else { out.append((name, m)) }
            }
            return out
        }
        return split("", Array(entries.indices), depth: 0)
    }

    /// A group as a candidate: its file count and what its largest files say about themselves.
    static func groupEntry(_ g: (name: String, members: [Int]), _ entries: [Entry]) -> Entry {
        let main = g.members.sorted { entries[$0].size > entries[$1].size }.prefix(5).map { i -> String in
            let e = entries[i]
            let name = e.path.split(separator: "/").last.map(String.init) ?? e.path
            return name + ": " + String(e.summary.prefix(70))
        }
        return Entry(path: g.name, summary: "\(g.members.count) files. Main files: " + main.joined(separator: "; "),
                     terms: [:])
    }

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

    static func termCounts(_ path: String, _ text: String) -> [String: Int] {
        var c: [String: Int] = [:]
        for t in tokens(path.replacingOccurrences(of: "/", with: " ") + " " + text) { c[t, default: 0] += 1 }
        return c
    }

    /// Entries ranked by BM25 over path + text (k1 1.2, b 0.75), best first.
    public static func keywordRank(_ query: String, _ entries: [Entry]) -> [Int] {
        let docs = entries.map(\.terms)
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
        var requests = 0
        var folders: [String] = []
        var candidates = Array(entries.indices)
        if entries.count > maxFiles {
            // folder step: judge the subfolders like files, keep the best until they hold maxFiles files
            let gs = Self.groups(entries, maxFiles: maxFiles)
            let gEntries = gs.map { Self.groupEntry($0, entries) }
            var gpool: [(Int, Double)] = []
            let gBatches = stride(from: 0, to: gs.count, by: perRequest).map { Array($0..<min($0 + perRequest, gs.count)) }
            for (b, batch) in gBatches.enumerated() {
                progress("Choosing among \(gs.count) folders (batch \(b + 1) of \(gBatches.count))…")
                if batch.count == 1 { gpool.append((batch[0], 1)); continue }
                let p = try await choose(query, batch.map { gEntries[$0] }, decide: decide)
                requests += 1
                gpool += zip(batch, p).sorted { $0.1 > $1.1 }.prefix(3).map { ($0.0, $0.1) }
            }
            var best = gpool.sorted { $0.1 > $1.1 }.prefix(12).map(\.0)
            if best.count > 1 {
                let p = try await choose(query, best.map { gEntries[$0] }, decide: decide)
                requests += 1
                best = zip(best, p).sorted { $0.1 > $1.1 }.map(\.0)
            }
            var chosen: [Int] = []
            for g in best where chosen.count < maxFiles {
                chosen += gs[g].members
                folders.append(gs[g].name)
            }
            if chosen.count > maxFiles {                     // trim by keyword rank, keep path order for batching
                var order: [Int: Int] = [:]
                for (r, i) in ranked.enumerated() { order[i] = r }
                chosen = Array(chosen.sorted { order[$0, default: .max] < order[$1, default: .max] }.prefix(maxFiles))
            }
            candidates = chosen.sorted { entries[$0].path < entries[$1].path }
        }
        var pool: [(Int, Double)] = []
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
        return Result(hits: hits, searched: candidates.count, total: entries.count, requests: requests, folders: folders)
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
            s += "The folder has \(r.total) files: the decision model first picked "
                + (r.folders.isEmpty ? "folders" : r.folders.prefix(6).joined(separator: ", "))
                + " and judged the \(r.searched) files in them (plus keyword search's best). Pass a narrower path to "
                + "judge a folder's files directly.\n"
        }
        return s + "Open the most likely file with read_file to confirm before answering."
    }
}
