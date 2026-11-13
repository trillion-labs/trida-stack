import Foundation

/// Each chat's scratch folder: a private working directory for the agent's intermediate files (drafts,
/// extracted text, scripts, long tool outputs). Writing there needs no approval; the workspace only gets what
/// the user asked for. Deleted with its chat; folders of deleted chats and folders untouched for a week are
/// removed at launch.
public enum Scratch {
    /// No spaces in the path, so the model can use it in shell commands as is.
    public static let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("co.trillionlabs.zoro1chat/scratch", isDirectory: true)

    /// Tool outputs longer than this are saved to a file; the model gets the start and the path.
    public static let spillChars = 8000
    public static let previewChars = 3000
    public static let maxAge: TimeInterval = 7 * 86400

    static func name(_ chatID: UUID) -> String { String(chatID.uuidString.lowercased().prefix(8)) }

    /// The chat's folder (created on first use by the tools that write into it).
    public static func dir(_ chatID: UUID, base: URL = base) -> URL {
        base.appendingPathComponent(name(chatID), isDirectory: true)
    }

    public static func remove(_ chatID: UUID, base: URL = base) {
        try? FileManager.default.removeItem(at: dir(chatID, base: base))
    }

    /// Removes the folders of chats not in `keep`, and folders with nothing changed for `maxAge`.
    public static func clean(keep: Set<UUID>, base: URL = base, now: Date = Date()) {
        let fm = FileManager.default
        let names = Set(keep.map(name))
        guard let dirs = try? fm.contentsOfDirectory(at: base, includingPropertiesForKeys: nil) else { return }
        for d in dirs {
            let newest = ([d] + files(d)).compactMap {
                try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate
            }.max() ?? .distantPast
            if !names.contains(d.lastPathComponent) || now.timeIntervalSince(newest) > maxAge {
                try? fm.removeItem(at: d)
            }
        }
    }

    /// The files in a scratch folder (recursive), newest first.
    public static func files(_ dir: URL) -> [URL] {
        guard let e = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: [.isRegularFileKey,
                                                         .contentModificationDateKey]) else { return [] }
        let urls = e.compactMap { $0 as? URL }.filter {
            (try? $0.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
        }
        func date(_ u: URL) -> Date {
            (try? u.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
        }
        return urls.sorted { date($0) > date($1) }
    }

    /// A long tool output: saved under `outputs/` in the scratch folder; returns its start and where the rest is.
    /// Returns the text unchanged when it is short or cannot be saved.
    public static func spill(_ text: String, tool: String, in dir: URL, limit: Int = spillChars,
                             preview: Int = previewChars) -> String {
        guard text.count > limit else { return text }
        let f = DateFormatter()
        f.dateFormat = "HHmmss"
        let file = dir.appendingPathComponent("outputs", isDirectory: true)
            .appendingPathComponent("\(tool)-\(f.string(from: Date()))-\(UUID().uuidString.prefix(4).lowercased()).txt")
        do {
            try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try text.write(to: file, atomically: true, encoding: .utf8)
        } catch {
            return text
        }
        return String(text.prefix(preview))
            + "\n\n[… \(text.count) characters in all; the full output is saved to \(file.path). "
            + "Read more with read_file (offset, max_chars) or search it with search_files.]"
    }
}
