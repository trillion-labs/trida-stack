import Foundation

/// A typed decision the app asked the server for while producing a reply (tool routing, a reply check, a
/// decision tool), kept with that reply so the user can mark the right answers. Marked decisions are training
/// data for the pointer head: `kevRecord()` is a line of kev's labelled JSONL (`train.py --decision_data`), and
/// trida-mlx-datamine writes them to decision_labels.jsonl.
public struct DecisionRecord: Identifiable, Equatable, Sendable, Codable {
    public var id = UUID()
    /// What asked: "Tool routing", "Reply check", "decide_choice", ...
    public var kind: String
    /// The /v1/decide request body as sent (JSON text).
    public var request: String
    /// The server's answer (JSON text).
    public var response: String
    /// The user's answers: question id -> answer key (a choice key, "true" / "false", or a level index).
    public var labels: [String: String]? = nil

    public init(kind: String, request: String, response: String) {
        self.kind = kind
        self.request = request
        self.response = response
    }

    struct Q {
        let id: String
        let type: String
        let instructions: String
        let keys: [String]          // answer keys, in option order
        let labels: [String]        // what to show for each
    }

    /// The questions as asked, in order.
    var questions: [Q] {
        guard let r = decision, let qs = r["questions"] as? [String: Any] else { return [] }
        let order = Self.keyOrder(request, path: decisionPath + ["questions"])
        return (order.isEmpty ? Array(qs.keys).sorted() : order).compactMap { id in
            guard let q = qs[id] as? [String: Any], let t = q["type"] as? String else { return nil }
            let ins = q["instructions"] as? String ?? ""
            switch t {
            case "noul": return Q(id: id, type: t, instructions: ins, keys: ["true", "false"], labels: ["yes", "no"])
            case "choice":
                let order = Self.keyOrder(request, path: decisionPath + ["questions", id, "criteria"])
                let c = q["criteria"] as? [String: Any] ?? [:]
                let keys = order.isEmpty ? Array(c.keys).sorted() : order.filter { c[$0] != nil }
                return Q(id: id, type: t, instructions: ins, keys: keys, labels: keys.map { k in
                    let d = c[k] as? String ?? ""
                    return d.isEmpty ? k : "\(k): \(d)"
                })
            default:
                let levels = (q["criteria"] as? [Any])?.map { "\($0)" } ?? []
                return Q(id: id, type: t, instructions: ins, keys: levels.indices.map(String.init), labels: levels)
            }
        }
    }

    /// The model's probability for each answer key of a question.
    func probabilities(_ q: Q) -> [String: Double] {
        guard let r = Self.object(response) else { return [:] }
        let answers = (r["answers"] as? [String: Any]) ?? [:]
        guard let a = answers[q.id] as? [String: Any] else { return [:] }
        if q.type == "noul", let y = (a["noul"] as? NSNumber)?.doubleValue { return ["true": y, "false": 1 - y] }
        let p = a["probabilities"] as? [String: Any] ?? [:]
        return p.compactMapValues { ($0 as? NSNumber)?.doubleValue }
    }

    public struct Option: Identifiable, Sendable {
        public var key: String
        public var label: String
        public var p: Double
        public var id: String { key }
    }

    public struct Row: Identifiable, Sendable {
        public var id: String           // the question id
        public var instructions: String
        public var options: [Option]
        public var label: String?       // the user's answer key
    }

    /// The questions with their options, the model's probabilities and the user's label, for display.
    public var rows: [Row] {
        questions.map { q in
            let p = probabilities(q)
            return Row(id: q.id, instructions: q.instructions,
                       options: zip(q.keys, q.labels).map { Option(key: $0, label: $1, p: p[$0] ?? 0) },
                       label: labels?[q.id])
        }
    }

    /// kev's labelled request for the questions the user answered: {"state", "questions": {id: {type,
    /// instructions, criteria, label}}}; a yes/no label is 1 / 0, a score label the level index. nil if none.
    public func kevRecord() -> [String: Any]? {
        guard let labels, !labels.isEmpty, let r = decision, let qs = r["questions"] as? [String: Any]
        else { return nil }
        var out: [String: Any] = [:]
        for (id, key) in labels {
            guard var q = qs[id] as? [String: Any], let t = q["type"] as? String else { continue }
            switch t {
            case "noul": q["label"] = key == "true" ? 1 : 0
            case "score": q["label"] = Int(key) ?? 0
            default: q["label"] = key
            }
            out[id] = q
        }
        guard !out.isEmpty, let state = r["state"] else { return nil }
        return ["state": state, "questions": out]
    }

    /// The decision itself ({"state", "questions"}): request bodies nest it under "request" (the shared
    /// request format); records saved by older builds have it at the top level.
    var decision: [String: Any]? {
        let r = Self.object(request)
        return (r?["request"] as? [String: Any]) ?? r
    }

    /// Where `decision` sits in the request body, for reading key order from the JSON text.
    var decisionPath: [String] { Self.object(request)?["request"] is [String: Any] ? ["request"] : [] }

    static func object(_ s: String) -> [String: Any]? {
        (try? JSONSerialization.jsonObject(with: Data(s.utf8))) as? [String: Any]
    }

    /// The keys of the object at `path` (e.g. ["questions", "need", "criteria"]) in the JSON text, in the order
    /// written: JSONSerialization's dictionaries forget it, and option order matters to the pointer head and to
    /// the display. Follows the path by searching for each key in turn, which holds for the bodies the app builds.
    static func keyOrder(_ json: String, path: [String]) -> [String] {
        var from = json.startIndex
        for key in path {
            guard let r = json.range(of: "\"\(key)\":", range: from..<json.endIndex) else { return [] }
            from = r.upperBound
        }
        let cs = Array(json[from...])
        var depth = 0, inString = false, escaped = false, readingKey = false, cur = "", keys: [String] = []
        for (k, ch) in cs.enumerated() {
            if inString {
                if escaped { escaped = false; if readingKey { cur.append(ch) }; continue }
                if ch == "\\" { escaped = true; if readingKey { cur.append(ch) }; continue }
                if ch == "\"" {
                    inString = false
                    if readingKey { readingKey = false; keys.append(cur); cur = "" }
                } else if readingKey { cur.append(ch) }
                continue
            }
            switch ch {
            case "\"":
                inString = true
                var b = k - 1                                   // a key: at depth 1, right after "{" or ","
                while b >= 0, cs[b].isWhitespace { b -= 1 }
                readingKey = depth == 1 && b >= 0 && (cs[b] == "{" || cs[b] == ",")
            case "{", "[": depth += 1
            case "}", "]":
                depth -= 1
                if depth == 0 { return keys }
            default:
                if depth == 0, !ch.isWhitespace, ch != "{" { return keys }   // not an object
            }
        }
        return keys
    }
}

/// Undo for a file the agent wrote: the file as it was before (a copy, or that it did not exist).
public enum FileUndo {
    public struct Info: Equatable, Sendable, Codable {
        public var path: String          // absolute
        public var backup: String?       // absolute path of the copy; nil = the file did not exist
    }

    public static var dir: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Zoro-1 Chat/undo", isDirectory: true)
    }

    /// Records `url` as it is now, before a write.
    public static func snapshot(_ url: URL) throws -> Info {
        let fm = FileManager.default
        guard fm.fileExists(atPath: url.path) else { return Info(path: url.path, backup: nil) }
        try fm.createDirectory(at: dir, withIntermediateDirectories: true)
        let copy = dir.appendingPathComponent(UUID().uuidString + "-" + url.lastPathComponent)
        try fm.copyItem(at: url, to: copy)
        return Info(path: url.path, backup: copy.path)
    }

    /// Puts the file back as it was: the saved copy, or no file if there was none.
    public static func restore(_ info: Info) throws {
        let fm = FileManager.default
        let url = URL(fileURLWithPath: info.path)
        if let b = info.backup {
            if fm.fileExists(atPath: url.path) { try fm.removeItem(at: url) }
            try fm.copyItem(at: URL(fileURLWithPath: b), to: url)
        } else if fm.fileExists(atPath: url.path) {
            try fm.removeItem(at: url)
        }
    }
}
