import Foundation
#if canImport(ImageIO)
import ImageIO
#endif

/// A tool's result: text, plus images when the tool returns pictures (view_image, screenshot).
/// Images are PNG bytes; the app sends them back to the model inside the tool message.
public struct ToolOutput: Equatable, Sendable {
    public var text: String
    public var images: [Data]

    public init(text: String, images: [Data] = []) {
        self.text = text
        self.images = images
    }
}

/// The agent's tools, scoped to one workspace folder. Same tool set and schemas as
/// trida_mlx/agent.py: list_dir, read_file, search_files, calculator, now, and opt-in
/// run_shell / write_file, which always ask the user first. With a vision model: view_image
/// (an image file from the workspace) and opt-in screenshot (asks first).
public struct AgentTools: Sendable {
    public var root: URL
    public var allowShell: Bool
    public var allowWrite: Bool
    /// The served model has a vision encoder: offer the image tools.
    public var vision: Bool
    public var allowScreenshot: Bool
    /// run_shell commands with the same id share a directory and environment (one per chat).
    public var shellSession: String
    /// The chat's scratch folder (see `Scratch`): readable and writable without approval, outside the workspace.
    public var scratch: URL?

    public init(root: URL, allowShell: Bool, allowWrite: Bool, vision: Bool = false, allowScreenshot: Bool = false,
                shellSession: String = "default", scratch: URL? = nil) {
        self.shellSession = shellSession
        self.root = root.standardizedFileURL.resolvingSymlinksInPath()
        self.scratch = scratch?.standardizedFileURL.resolvingSymlinksInPath()
        self.allowShell = allowShell
        self.allowWrite = allowWrite
        self.vision = vision
        self.allowScreenshot = allowScreenshot
    }

    /// Tools that change something, run code or look at the screen; the app asks before each call
    /// (except `write_file` into the scratch folder: see `writesOnlyScratch`).
    public static let needsApproval: Set<String> = ["run_shell", "write_file", "screenshot", "fetch_url", "open_link"]

    /// The web browsing tools (see WebBrowser.swift).
    public static let webTools: Set<String> = ["fetch_url", "find_in_page", "read_section", "open_link"]

    /// Tools whose long output is saved to the scratch folder (read_file reads slices with offset instead, and
    /// run_shell keeps the start and end of its output).
    static let spills: Set<String> = ["read_pdf", "search_files", "find_files", "list_dir", "read_section"]

    /// Longest side of images handed to the model (the server also caps pixels per image).
    public static let maxImageSide = 1536

    static func fn(_ name: String, _ desc: String, _ props: [String: Any], _ required: [String] = []) -> [String: Any] {
        ["type": "function",
         "function": ["name": name, "description": desc,
                      "parameters": ["type": "object", "properties": props, "required": required]]]
    }

    public var specs: [[String: Any]] {
        let str: [String: Any] = ["type": "string"]
        var s: [[String: Any]] = [
            Self.fn("list_dir", "List files in a directory (relative to the workspace).", ["path": str]),
            Self.fn("read_file", "Read a text file (relative to the workspace, or in your scratch folder). Returns at "
                    + "most max_chars (default 8000) starting at character offset (default 0).",
                    ["path": str, "max_chars": ["type": "integer"], "offset": ["type": "integer"]], ["path"]),
            Self.fn("read_pdf", "Read the text of a PDF (relative to the workspace), page by page. pages is a range "
                    + "such as '1-3' or '5' (default: the first 5 pages). Scanned pages without text come back as "
                    + "images when you can see images.",
                    ["path": str, "pages": str], ["path"]),
            Self.fn("find_files", "Find files by NAME under a directory (recursive). pattern is a glob such as "
                    + "'*.png' or 'report*', or plain text matched anywhere in the name (case-insensitive).",
                    ["pattern": str, "path": str], ["pattern"]),
            Self.fn("search_files", "Search the CONTENTS of text files under a directory for lines containing a "
                    + "substring (not file names: use find_files for names).",
                    ["query": str, "path": str], ["query"]),
            Self.fn("calculator", "Evaluate an arithmetic expression, e.g. '17*24+3'.", ["expression": str], ["expression"]),
            Self.fn("now", "Current local date and time.", [:]),
            Self.fn("fetch_url", "Open a web page: returns its title, its sections, how many links it has and the start "
                    + "of its text (on Wikipedia, the infobox and the first paragraphs). Then use find_in_page, "
                    + "read_section and open_link on it. For facts that change (who holds an office now, news, prices, "
                    + "releases) open a source such as https://en.wikipedia.org/wiki/<Title>, "
                    + "https://ko.wikipedia.org/wiki/<제목>, or a Wikipedia search: "
                    + "https://en.wikipedia.org/w/index.php?search=<words>. For something Korean (a school, a "
                    + "company, a place), search with the words exactly as the user wrote them: "
                    + "https://search.naver.com/search.naver?query=<words>, or https://namu.wiki/w/<제목>. Write "
                    + "Korean and other non-English words in the URL as they are; don't percent-encode them "
                    + "yourself and don't romanize names.", ["url": str], ["url"]),
            Self.fn("find_in_page", "Find text on the page you opened last (case-insensitive): every place it occurs, "
                    + "with the text around it and its section, and the links whose text matches. Use it to find the "
                    + "exact line that answers the question, e.g. 'Incumbent' or 'current president'.",
                    ["query": str], ["query"]),
            Self.fn("read_section", "Read one section of the page you opened last, by its heading (from the "
                    + "Sections list fetch_url gave).", ["heading": str], ["heading"]),
            Self.fn("open_link", "Follow a link on the page you opened last, by its text (e.g. 'Donald Trump') or its "
                    + "number from find_in_page, and open that page like fetch_url.", ["link": str], ["link"]),
        ]
        if allowShell {
            s.append(Self.fn("run_shell", "Run a shell command (zsh) in this chat's terminal; returns stdout/stderr "
                             + "and the exit status. The working directory and exported variables carry over between "
                             + "calls, like one terminal tab (it starts in the workspace). Not interactive: no prompts "
                             + "or editors. timeout is in seconds (default 300, at most 1800).",
                             ["command": str, "timeout": ["type": "integer"]], ["command"]))
        }
        if allowWrite {
            s.append(Self.fn("write_file", "Write a text file (relative to the workspace, or in your scratch folder: "
                             + "no approval needed there).", ["path": str, "content": str], ["path", "content"]))
        } else if scratch != nil {
            s.append(Self.fn("write_file", "Write a text file in your scratch folder (absolute path). Writing to the "
                             + "workspace is turned off.", ["path": str, "content": str], ["path", "content"]))
        }
        if vision {
            s.append(Self.fn("view_image", "Look at an image file (png, jpg, heic, gif, ...) relative to the workspace. "
                             + "The image is shown to you in the tool result.", ["path": str], ["path"]))
            if allowScreenshot {
                s.append(Self.fn("screenshot", "Take a screenshot of the user's main display and look at it. "
                                 + "The user approves each capture.", [:]))
            }
        }
        return s
    }

    public var systemPrompt: String {
        let day = DateFormatter()
        day.locale = Locale(identifier: "en_US_POSIX")
        day.dateFormat = "EEEE, yyyy-MM-dd"
        // the date changes the prompt once a day, so the server's prompt cache barely notices
        var p = "You are a helpful on-device assistant on the user's Mac. Today is \(day.string(from: Date())). "
        p += "What you know comes from training data that stops before today: for anything that may have changed "
            + "since (who holds an office or a job now, news, prices, versions), look it up with fetch_url, or say "
            + "that your information may be out of date. On a web page, find the line that answers the question with "
            + "find_in_page (or read_section, or open_link to follow a link) before answering; answer only from what "
            + "the page says and quote the line you used, and if the page does not say it, say so. "
        p += "Call a tool only when the request needs something you cannot know from the conversation: the contents "
            + "of the user's files, their screen, the current time, current facts from the web, or an exact "
            + "calculation. Answer directly, without "
            + "tools, for conversation, questions about yourself, translation, writing, and explaining or summarizing "
            + "text the user pasted, even when that text mentions file names or paths; do not open files just because "
            + "they are mentioned. If a message is pasted text with no instruction, ask what to do with it, or follow "
            + "the request from the previous turn (for example, translate it). "
            + "Do not list or mention your tools unless the user asks what you can do. "
            + "\"this\" or \"it\" in a request means something already in the conversation (your last reply or "
            + "the user's pasted text), not a file: \"translate this\" / \"이거 번역해줘\" -> translate it directly. "
            + "Examples: \"tell me a joke\", \"translate this\", \"who are you?\", \"write a poem\", \"what is in this "
            + "picture?\" (attached) -> answer directly, no tool. \"what's in notes.txt?\" -> read_file. \"find my "
            + "invoices\" -> find_files. \"what time is it?\" -> now. \"who is the president of Korea now?\" -> "
            + "fetch_url, then find_in_page. \"what's on my screen?\" -> screenshot. "
        p += "The workspace is \(root.path); file paths are relative to it (use \".\" for the workspace itself). "
            + "Use find_files to locate files by name and search_files to search inside files; read PDFs with "
            + "read_pdf. "
            + "If a tool returns an error, read it and change your approach instead of repeating the call."
        if let scratch {
            p += " Your scratch folder for this chat is \(scratch.path) (also $SCRATCH in run_shell). Put "
                + "intermediate files there: drafts, extracted text, notes, scripts, long command output (redirect it "
                + "to a file). Writing there needs no approval; write to the workspace only what the user asked for. "
                + "Long tool outputs are saved there automatically: you get the start and the file's path, and can "
                + "read more with read_file offset."
        }
        if vision {
            p += " You can see images the user attaches to their messages: look at them and answer directly; "
                + "no tool is needed for that."
        }
        if vision && allowScreenshot {
            p += " To see what is on the user's screen, call screenshot."
        } else if vision {
            p += " You cannot see the user's screen; if they ask about it, tell them to enable the screenshot tool "
                + "in Settings → Agent, or to attach a screenshot."
        }
        return p + " Be concise."
    }

    /// One-line description of a call, shown in the approval prompt.
    public static func describe(_ call: ToolCall) -> String {
        let a = call.argumentDict
        switch call.name {
        case "run_shell": return "Run in this chat's terminal:\n\n\(a["command"] as? String ?? "")"
        case "write_file":
            let content = a["content"] as? String ?? ""
            return "Write \(content.count) characters to \(a["path"] as? String ?? "?")"
        case "screenshot": return "Capture your main display and show it to the model."
        case "fetch_url": return "Fetch this web page:\n\n\(a["url"] as? String ?? "?")"
        case "open_link": return "Open this link from the current page:\n\n\(a["link"] as? String ?? "?")"
        default: return "\(call.name)(\(call.arguments))"
        }
    }

    enum ToolError: LocalizedError {
        case escapes, missingArgument(String), disabled(String), outside(String), noSuchDirectory(String)

        var errorDescription: String? {
            switch self {
            case .escapes: return "path escapes the workspace"
            case let .outside(root): return "that path is outside the workspace; only files under \(root) (and your scratch folder) are available"
            case let .noSuchDirectory(p): return "no such directory: \(p) (paths are relative to the workspace; use \".\" for the workspace itself)"
            case let .missingArgument(a): return "missing argument '\(a)'"
            case let .disabled(t): return "\(t) is not enabled"
            }
        }
    }

    /// A path from the model, as a URL inside the workspace. Relative paths are relative to the
    /// workspace; absolute paths (and the workspace path written without its leading "/", or with
    /// "~") are accepted when they point inside it. Anything else is an error, never remapped.
    func resolve(_ rel: String?) throws -> URL {
        var r = (rel ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        if r.hasPrefix("~") { r = (r as NSString).expandingTildeInPath }
        let url: URL
        if r.isEmpty || r == "." || r == "./" {
            url = root
        } else if r.hasPrefix("/") {
            url = URL(fileURLWithPath: r).standardizedFileURL.resolvingSymlinksInPath()
            guard inside(url) || inScratch(url) else { throw ToolError.outside(root.path) }
        } else {
            let asAbsolute = URL(fileURLWithPath: "/" + r).standardizedFileURL.resolvingSymlinksInPath()
            if inside(asAbsolute) || inScratch(asAbsolute) {
                url = asAbsolute  // e.g. "Users/me/Desktop/x" when the workspace is /Users/me/Desktop
            } else {
                url = root.appendingPathComponent(r).standardizedFileURL.resolvingSymlinksInPath()
                guard inside(url) else { throw ToolError.escapes }
            }
        }
        return url
    }

    static func contains(_ base: URL, _ u: URL) -> Bool {
        u.path == base.path || u.path.hasPrefix(base.path.hasSuffix("/") ? base.path : base.path + "/")
    }

    func inside(_ u: URL) -> Bool { Self.contains(root, u) }

    /// The URL is in this chat's scratch folder.
    public func inScratch(_ u: URL) -> Bool { scratch.map { Self.contains($0, u.standardizedFileURL) } ?? false }

    /// A write_file call whose path is in the scratch folder: runs without asking.
    public func writesOnlyScratch(_ call: ToolCall) -> Bool {
        guard call.name == "write_file", let p = call.argumentDict["path"] as? String, let u = try? resolve(p) else {
            return false
        }
        return inScratch(u)
    }

    /// Names in a workspace directory (folders end in "/"), sorted, for decisions over files.
    public func directoryEntries(_ rel: String?, limit: Int = 120) throws -> [String] {
        let dir = try existingDirectory(rel)
        let items = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.isDirectoryKey],
                                                                options: [.skipsHiddenFiles])
        return Array(items.map { u -> String in
            let isDir = (try? u.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
            return u.lastPathComponent + (isDir ? "/" : "")
        }.sorted().prefix(limit))
    }

    /// A workspace folder from a tool argument (relative path, "." or an absolute path inside it).
    public func directory(_ rel: String?) throws -> URL { try existingDirectory(rel) }

    /// A workspace path from a tool argument (it need not exist).
    public func path(_ rel: String?) throws -> URL { try resolve(rel) }

    func existingDirectory(_ rel: String?) throws -> URL {
        let url = try resolve(rel)
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue else {
            throw ToolError.noSuchDirectory(rel ?? ".")
        }
        return url
    }

    /// Runs a tool: image tools return pictures, everything else text (see `call`). Long outputs of the tools in
    /// `spills` are saved to the scratch folder, and the model gets their start and the file's path.
    public func run(_ call: ToolCall) async -> ToolOutput {
        var out = await runTool(call)
        if let scratch, Self.spills.contains(call.name), !out.text.hasPrefix("error") {
            // a web page's facts are rarely in its first lines: show more of it
            let preview = call.name == "fetch_url" ? 6000 : Scratch.previewChars
            out = ToolOutput(text: Scratch.spill(out.text, tool: call.name, in: scratch, preview: preview),
                             images: out.images)
        }
        return out
    }

    func runTool(_ call: ToolCall) async -> ToolOutput {
        let a = call.argumentDict
        switch call.name {
        case "read_pdf":
            guard let p = a["path"] as? String else { return ToolOutput(text: "error: missing argument 'path'") }
            do {
                let url = try resolve(p)
                guard FileManager.default.fileExists(atPath: url.path) else {
                    return ToolOutput(text: "error: no such file: \(p)")
                }
                // with a scratch folder the whole text is kept there (see `run`), so read more of it at once
                let r = PDFText.read(url, pages: a["pages"] as? String, maxChars: scratch == nil ? 24_000 : 400_000,
                                     renderScans: vision)
                return ToolOutput(text: r.text, images: r.images)
            } catch {
                return ToolOutput(text: "error: \(error.localizedDescription)")
            }
        case "view_image":
            guard vision else { return ToolOutput(text: "error: view_image is not enabled") }
            guard let p = a["path"] as? String else { return ToolOutput(text: "error: missing argument 'path'") }
            do {
                let url = try resolve(p)
                guard FileManager.default.fileExists(atPath: url.path) else {
                    return ToolOutput(text: "error: no such file: \(p)")
                }
                guard let png = Self.pngData(fromFile: url, maxSide: Self.maxImageSide) else {
                    return ToolOutput(text: "error: \(p) is not an image macOS can read")
                }
                return ToolOutput(text: "Image \(p):", images: [png])
            } catch {
                return ToolOutput(text: "error: \(error.localizedDescription)")
            }
        case "screenshot":
            guard vision, allowScreenshot else { return ToolOutput(text: "error: screenshot is not enabled") }
            do {
                guard let png = try await Self.captureScreen() else {
                    return ToolOutput(text: "error: the screenshot failed (Screen Recording permission?)")
                }
                return ToolOutput(text: "Screenshot of the main display:", images: [png])
            } catch {
                return ToolOutput(text: "error: \(error.localizedDescription)")
            }
        default:
            return ToolOutput(text: await self.call(call))
        }
    }

    /// PNG bytes of an image file, downscaled so the longest side is at most `maxSide`.
    public static func pngData(fromFile url: URL, maxSide: Int) -> Data? {
        #if canImport(ImageIO)
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        let opts: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maxSide,
        ]
        guard let img = CGImageSourceCreateThumbnailAtIndex(src, 0, opts as CFDictionary) else { return nil }
        let out = NSMutableData()
        guard let dst = CGImageDestinationCreateWithData(out as CFMutableData, "public.png" as CFString, 1, nil) else {
            return nil
        }
        CGImageDestinationAddImage(dst, img, nil)
        return CGImageDestinationFinalize(dst) ? out as Data : nil
        #else
        return nil
        #endif
    }

    /// `screencapture -x -m` (main display, no sound) into a temp file, returned as downscaled PNG.
    /// macOS asks once for Screen Recording permission; without it the capture shows only the
    /// desktop and this app's own windows.
    static func captureScreen() async throws -> Data? {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("zoro-shot-\(UUID().uuidString).png")
        defer { try? FileManager.default.removeItem(at: url) }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
        p.arguments = ["-x", "-m", "-t", "png", url.path]
        let status: Int32 = try await withCheckedThrowingContinuation { cont in
            p.terminationHandler = { proc in cont.resume(returning: proc.terminationStatus) }
            do {
                try p.run()
            } catch {
                p.terminationHandler = nil
                cont.resume(throwing: error)
            }
        }
        guard status == 0 else { return nil }
        return pngData(fromFile: url, maxSide: maxImageSide)
    }

    /// Runs a text tool and returns its result (errors are returned as text, like agent.py).
    public func call(_ call: ToolCall) async -> String {
        let a = call.argumentDict
        do {
            switch call.name {
            case "list_dir":
                let dir = try existingDirectory(a["path"] as? String)
                let items = try FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.isDirectoryKey])
                return items.map { u -> String in
                    let isDir = (try? u.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true
                    return u.lastPathComponent + (isDir ? "/" : "")
                }.sorted().prefix(500).joined(separator: "\n")
            case "read_file":
                guard let p = a["path"] as? String else { throw ToolError.missingArgument("path") }
                let maxChars = max(1, GenStats.int(a["max_chars"]) ?? 8000)
                let offset = max(0, GenStats.int(a["offset"]) ?? 0)
                let url = try resolve(p)
                if url.pathExtension.lowercased() == "pdf" {        // its bytes would be useless as text
                    return PDFText.read(url, pages: nil, maxChars: maxChars).text
                }
                let text = String(decoding: try Data(contentsOf: url), as: UTF8.self)
                return Self.slice(text, offset: offset, maxChars: maxChars)
            case "search_files":
                guard let q = a["query"] as? String, !q.isEmpty else { throw ToolError.missingArgument("query") }
                let dir = try existingDirectory(a["path"] as? String)
                return search(q, in: dir)
            case "find_files":
                guard let pat = a["pattern"] as? String, !pat.isEmpty else { throw ToolError.missingArgument("pattern") }
                let dir = try existingDirectory(a["path"] as? String)
                return find(pat, in: dir)
            case "calculator":
                guard let e = a["expression"] as? String else { throw ToolError.missingArgument("expression") }
                return Calculator.format(try Calculator.evaluate(e))
            case "now":
                let f = DateFormatter()
                f.dateFormat = "yyyy-MM-dd'T'HH:mm:ss"
                return f.string(from: Date())
            case "fetch_url":
                guard let u = a["url"] as? String, !u.isEmpty else { throw ToolError.missingArgument("url") }
                return try await open(WebFetch.fetch(u))
            case "open_link":
                guard let ref = a["link"] as? String, !ref.isEmpty else { throw ToolError.missingArgument("link") }
                guard let page = WebSession.session(shellSession).current else { return Self.noPage }
                guard let link = page.link(ref) else {
                    return "error: no link \"\(ref)\" on \(page.title). find_in_page lists the links whose text matches."
                }
                return try await open(WebFetch.fetch(link.url))
            case "find_in_page":
                guard let q = a["query"] as? String, !q.isEmpty else { throw ToolError.missingArgument("query") }
                guard let page = WebSession.session(shellSession).current else { return Self.noPage }
                return page.find(q)
            case "read_section":
                guard let h = a["heading"] as? String, !h.isEmpty else { throw ToolError.missingArgument("heading") }
                guard let page = WebSession.session(shellSession).current else { return Self.noPage }
                return page.section(h)
            case "run_shell":
                guard allowShell else { throw ToolError.disabled("run_shell") }
                guard let c = a["command"] as? String else { throw ToolError.missingArgument("command") }
                let t = (a["timeout"] as? Double) ?? (a["timeout"] as? Int).map(Double.init) ?? ShellSession.defaultTimeout
                return try await ShellSession.session(shellSession, root: root, scratch: scratch).run(c, timeout: t)
            case "write_file":
                guard allowWrite || scratch != nil else { throw ToolError.disabled("write_file") }
                guard let p = a["path"] as? String else { throw ToolError.missingArgument("path") }
                let content = a["content"] as? String ?? ""
                let url = try resolve(p)
                guard allowWrite || inScratch(url) else { throw ToolError.disabled("write_file outside the scratch folder") }
                try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                try content.write(to: url, atomically: true, encoding: .utf8)
                return "wrote \(content.count) chars to \(p)"
            default:
                return "unknown tool \(call.name)"
            }
        } catch {
            return "error: \(error.localizedDescription)"
        }
    }

    static let noPage = "error: no web page is open in this chat; open one with fetch_url first"

    /// Makes `page` the chat's current page, keeps its whole text in the scratch folder, returns its outline.
    func open(_ page: WebPage) -> String {
        WebSession.session(shellSession).open(page)
        var out = page.outline()
        if let scratch {
            let slug = page.title.lowercased().map { $0.isLetter || $0.isNumber ? $0 : "-" }.prefix(48)
            let file = scratch.appendingPathComponent("pages", isDirectory: true).appendingPathComponent(String(slug) + ".txt")
            if (try? FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)) != nil,
               (try? (page.url + "\n\n" + page.text).write(to: file, atomically: true, encoding: .utf8)) != nil {
                out += "\n(The whole text is saved at \(file.path).)"
            }
        }
        return out
    }

    /// Up to `maxChars` characters from `offset`, with a note on what is left when the text goes on.
    static func slice(_ text: String, offset: Int, maxChars: Int) -> String {
        let total = text.count
        guard offset > 0 || total > maxChars else { return text }
        guard offset < total else { return "(the file has \(total) characters; offset \(offset) is past the end)" }
        let start = text.index(text.startIndex, offsetBy: offset)
        let part = text[start...].prefix(maxChars)
        let end = offset + part.count
        return String(part) + (end < total
            ? "\n\n[characters \(offset)–\(end) of \(total); read_file with offset \(end) for more]" : "")
    }

    /// Workspace-relative names of files whose name matches `pattern` (glob, or substring when the
    /// pattern has no wildcard), newest first, at most 200.
    func find(_ pattern: String, in dir: URL) -> String {
        let p = pattern.trimmingCharacters(in: .whitespaces)
        let glob = (p.contains("*") || p.contains("?") || p.contains("[")) ? p : "*\(p)*"
        let rootPath = root.path.hasSuffix("/") ? root.path : root.path + "/"
        guard let en = FileManager.default.enumerator(at: dir, includingPropertiesForKeys: [.isRegularFileKey, .contentModificationDateKey],
                                                      options: [.skipsHiddenFiles, .skipsPackageDescendants]) else {
            return "(no files)"
        }
        var hits: [(String, Date)] = []
        var scanned = 0
        for case let f as URL in en {
            scanned += 1
            if scanned > 200_000 { break }
            guard fnmatch(glob, f.lastPathComponent, FNM_CASEFOLD) == 0 else { continue }
            let v = try? f.resourceValues(forKeys: [.isRegularFileKey, .contentModificationDateKey])
            guard v?.isRegularFile == true else { continue }
            let rel = f.path.hasPrefix(rootPath) ? String(f.path.dropFirst(rootPath.count)) : f.path
            hits.append((rel, v?.contentModificationDate ?? .distantPast))
        }
        if hits.isEmpty { return "(no files named \(glob) under \(dir.path == root.path ? "the workspace" : dir.path))" }
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd HH:mm"
        let lines = hits.sorted { $0.1 > $1.1 }.prefix(200).map { "\($0.0)  (\(f.string(from: $0.1)))" }
        return lines.joined(separator: "\n") + (hits.count > 200 ? "\n… \(hits.count - 200) more" : "")
    }

    func search(_ q: String, in dir: URL) -> String {
        var hits: [String] = []
        let fm = FileManager.default
        guard let en = fm.enumerator(at: dir, includingPropertiesForKeys: [.isRegularFileKey, .fileSizeKey],
                                     options: [.skipsHiddenFiles, .skipsPackageDescendants]) else { return "(no matches)" }
        let rootPath = root.path.hasSuffix("/") ? root.path : root.path + "/"
        for case let f as URL in en {
            let v = try? f.resourceValues(forKeys: [.isRegularFileKey, .fileSizeKey])
            guard v?.isRegularFile == true, (v?.fileSize ?? 0) < 1_000_000,
                  let data = try? Data(contentsOf: f) else { continue }
            if data.prefix(4096).contains(0) { continue }  // skip binaries
            let rel = f.path.hasPrefix(rootPath) ? String(f.path.dropFirst(rootPath.count)) : f.lastPathComponent
            var n = 0
            for line in String(decoding: data, as: UTF8.self).split(separator: "\n", omittingEmptySubsequences: false) {
                n += 1
                if line.contains(q) {
                    hits.append("\(rel):\(n): \(line.trimmingCharacters(in: .whitespaces).prefix(200))")
                    if hits.count >= 50 { return hits.joined(separator: "\n") }
                }
            }
        }
        return hits.isEmpty ? "(no matches)" : hits.joined(separator: "\n")
    }

    /// `zsh -lc command` in the workspace, 60 s timeout, last 8000 characters of output.
    static func shell(_ command: String, cwd: URL, timeout: TimeInterval = 60) async throws -> String {
        let outURL = FileManager.default.temporaryDirectory.appendingPathComponent("trida-shell-\(UUID().uuidString).log")
        FileManager.default.createFile(atPath: outURL.path, contents: nil)
        defer { try? FileManager.default.removeItem(at: outURL) }
        let handle = try FileHandle(forWritingTo: outURL)
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/zsh")
        p.arguments = ["-lc", command]
        p.currentDirectoryURL = cwd
        p.standardOutput = handle
        p.standardError = handle
        p.standardInput = FileHandle.nullDevice
        let status: Int32 = try await withCheckedThrowingContinuation { cont in
            p.terminationHandler = { proc in cont.resume(returning: proc.terminationStatus) }
            do {
                try p.run()
            } catch {
                p.terminationHandler = nil
                cont.resume(throwing: error)
                return
            }
            DispatchQueue.global().asyncAfter(deadline: .now() + timeout) {
                if p.isRunning { p.terminate() }
            }
        }
        try? handle.close()
        let text = (try? String(contentsOf: outURL, encoding: .utf8)) ?? ""
        let tail = String(text.suffix(8000))
        return tail.isEmpty ? "(exit \(status))" : tail + (status == 0 ? "" : "\n(exit \(status))")
    }
}

/// + - * / % ^ (or **) and parentheses over doubles; no eval, no NSExpression.
public enum Calculator {
    public enum CalcError: LocalizedError {
        case syntax(String)
        public var errorDescription: String? {
            if case let .syntax(m) = self { return "unsupported expression: \(m)" }
            return nil
        }
    }

    public static func evaluate(_ s: String) throws -> Double {
        var p = Parser(chars: Array(s.replacingOccurrences(of: "**", with: "^")))
        let v = try p.expr()
        p.skip()
        guard p.i == p.chars.count else { throw CalcError.syntax("unexpected '\(p.chars[p.i])'") }
        return v
    }

    public static func format(_ v: Double) -> String {
        if v.isFinite, v == v.rounded(), abs(v) < 1e15 { return String(Int64(v)) }
        return String(v)
    }

    struct Parser {
        var chars: [Character]
        var i = 0

        mutating func skip() { while i < chars.count, chars[i] == " " { i += 1 } }

        mutating func peek() -> Character? { skip(); return i < chars.count ? chars[i] : nil }

        mutating func expr() throws -> Double {
            var v = try term()
            while let c = peek(), c == "+" || c == "-" {
                i += 1
                let r = try term()
                v = c == "+" ? v + r : v - r
            }
            return v
        }

        mutating func term() throws -> Double {
            var v = try power()
            while let c = peek(), c == "*" || c == "/" || c == "%" {
                i += 1
                let r = try power()
                switch c {
                case "*": v *= r
                case "/": v /= r
                default: v = v.truncatingRemainder(dividingBy: r)
                }
            }
            return v
        }

        mutating func power() throws -> Double {
            let b = try unary()
            if peek() == "^" {
                i += 1
                let e = try power()  // right-associative
                return pow(b, e)
            }
            return b
        }

        mutating func unary() throws -> Double {
            if peek() == "-" {
                i += 1
                let v = try unary()
                return -v
            }
            if peek() == "+" { i += 1; return try unary() }
            return try atom()
        }

        mutating func atom() throws -> Double {
            if peek() == "(" {
                i += 1
                let v = try expr()
                guard peek() == ")" else { throw CalcError.syntax("missing ')'") }
                i += 1
                return v
            }
            skip()
            let start = i
            while i < chars.count, chars[i].isNumber || chars[i] == "." || chars[i] == "_" { i += 1 }
            let tok = String(chars[start..<i]).replacingOccurrences(of: "_", with: "")
            guard let v = Double(tok) else {
                throw CalcError.syntax(i < chars.count ? "unexpected '\(chars[i])'" : "unexpected end")
            }
            return v
        }
    }
}
