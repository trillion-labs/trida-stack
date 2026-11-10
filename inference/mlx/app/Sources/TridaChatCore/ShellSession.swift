import Foundation

/// The agent's terminal: run_shell commands of one chat share a working directory and exported environment,
/// like one terminal tab. Each command runs in a fresh `zsh -l`; before it, the session's saved directory and
/// exports are restored, and after it (also on `exit`) the new ones are saved. Aliases, functions and
/// non-exported variables do not carry over.
public final class ShellSession: @unchecked Sendable {
    private static let lock = NSLock()
    private static var sessions: [String: ShellSession] = [:]

    /// The session for `id` (a chat), created in `root` the first time.
    public static func session(_ id: String, root: URL) -> ShellSession {
        lock.lock()
        defer { lock.unlock() }
        if let s = sessions[id] { return s }
        let s = ShellSession(root: root)
        sessions[id] = s
        return s
    }

    public static let defaultTimeout: TimeInterval = 300
    public static let maxTimeout: TimeInterval = 1800
    /// Output longer than head + tail characters keeps its start and end.
    public static let headChars = 4000, tailChars = 8000

    private let dir: URL
    private let cwdFile: URL
    private let envFile: URL

    init(root: URL) {
        dir = FileManager.default.temporaryDirectory.appendingPathComponent("zoro-shell-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        cwdFile = dir.appendingPathComponent("cwd")
        envFile = dir.appendingPathComponent("env")
        try? root.path.write(to: cwdFile, atomically: true, encoding: .utf8)
    }

    /// The directory the next command starts in.
    public var cwd: String {
        (try? String(contentsOf: cwdFile, encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    }

    static func quote(_ s: String) -> String { "'" + s.replacingOccurrences(of: "'", with: "'\\''") + "'" }

    /// Runs `command`; returns its combined output (start and end kept when long) and the exit status.
    /// `sandboxIn`: run it in the macOS sandbox (see `sandboxProfile`) with that workspace, for commands that run
    /// without asking. Approved commands run as the user, unsandboxed.
    public func run(_ command: String, timeout: TimeInterval = defaultTimeout, sandboxIn workspace: URL? = nil)
        async throws -> String {
        let t = min(max(timeout, 1), Self.maxTimeout)
        let out = dir.appendingPathComponent("out-\(UUID().uuidString).log")
        FileManager.default.createFile(atPath: out.path, contents: nil)
        defer { try? FileManager.default.removeItem(at: out) }
        let cwd = Self.quote(self.cwd), cf = Self.quote(cwdFile.path), ef = Self.quote(envFile.path)
        // Restore, run, save (the trap also saves after `exit`). Sandboxed: no rc files (they live in the home
        // directory, which the sandbox hides), no zsh glob qualifiers (they run code), git without the user's
        // config (also hidden), and Homebrew on PATH; the git variables are not saved for later commands.
        let sandboxed = workspace != nil
        let setup = sandboxed
            ? "export PATH=\"/opt/homebrew/bin:/usr/local/bin:$PATH\"\nsetopt no_bare_glob_qual\n"
                + "export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1\n"
            : ""
        let save = sandboxed ? "unset GIT_CONFIG_GLOBAL GIT_CONFIG_NOSYSTEM; " : ""
        let script = """
            \(setup)[ -f \(ef) ] && source \(ef) 2>/dev/null
            cd \(cwd) 2>/dev/null || cd ~
            trap 'pwd > \(cf); \(save)export -p > \(ef)' EXIT
            \(command)
            """
        let handle = try FileHandle(forWritingTo: out)
        let p = Process()
        if let workspace {
            p.executableURL = URL(fileURLWithPath: "/usr/bin/sandbox-exec")
            p.arguments = ["-p", Self.sandboxProfile(workspace: workspace, session: dir), "/bin/zsh", "-fc", script]
        } else {
            p.executableURL = URL(fileURLWithPath: "/bin/zsh")
            p.arguments = ["-lc", script]
        }
        p.standardOutput = handle
        p.standardError = handle
        p.standardInput = FileHandle.nullDevice
        final class Flag: @unchecked Sendable { var on = false }
        let timedOut = Flag()
        let status: Int32 = try await withCheckedThrowingContinuation { cont in
            p.terminationHandler = { proc in cont.resume(returning: proc.terminationStatus) }
            do {
                try p.run()
            } catch {
                p.terminationHandler = nil
                cont.resume(throwing: error)
                return
            }
            DispatchQueue.global().asyncAfter(deadline: .now() + t) {
                if p.isRunning { timedOut.on = true; p.terminate() }
            }
        }
        try? handle.close()
        let text = (try? String(contentsOf: out, encoding: .utf8)) ?? ""
        var body = Self.clip(text)
        if timedOut.on { body += (body.isEmpty ? "" : "\n") + "(stopped after \(Int(t)) s; pass a larger timeout, up to \(Int(Self.maxTimeout)))" }
        let tail = "(exit \(status); cwd \(self.cwd))"
        return body.isEmpty ? tail : body + "\n" + tail
    }

    /// The first headChars and last tailChars of long output, with a note of what was left out.
    public static func clip(_ s: String) -> String {
        guard s.count > headChars + tailChars else { return s }
        let omitted = s.count - headChars - tailChars
        return String(s.prefix(headChars)) + "\n… \(omitted) characters omitted …\n" + String(s.suffix(tailChars))
    }

    // MARK: - read-only commands

    /// Commands that only read, whatever their arguments, except the flags in `deniedFlags`. No pagers or `top`
    /// (they wait for a terminal that is not there).
    static let readOnlyCommands: Set<String> = [
        "ls", "pwd", "cat", "head", "tail", "wc", "grep", "egrep", "fgrep", "rg", "which", "whoami", "date", "echo",
        "file", "stat", "du", "df", "tree", "uname", "printenv", "sw_vers", "cd", "sort", "uniq",
        "cut", "tr", "nl", "basename", "dirname", "realpath", "readlink", "id", "hostname", "uptime", "ps",
        "diff", "cmp", "md5", "shasum", "md5sum", "sha256sum", "type", "true", "jq", "column", "fd",
    ]
    /// Flags that make an otherwise read-only command write files or run other programs. A one-letter flag also
    /// counts inside a cluster (`sort -uo out`), and `--flag=value` counts as `--flag`.
    static let deniedFlags: [String: Set<String>] = [
        "fd": ["-x", "-X", "--exec", "--exec-batch"],
        "rg": ["--pre", "--pre-glob"],
        "sort": ["-o", "--output", "--compress-program"],
        "tree": ["-o"],
        "file": ["-C"],
    ]
    /// Read-only git subcommands. Not `config`, `grep` (`-O` opens files in a program) or `reflog` (`expire` and
    /// `delete` drop recovery data); `--output` and `-O` are refused for all of them.
    static let readOnlyGit: Set<String> = [
        "status", "log", "diff", "show", "branch", "ls-files", "rev-parse", "blame", "describe", "shortlog", "tag",
        "remote", "ls-tree", "cat-file",
    ]

    static func hasFlag(_ words: ArraySlice<String>, _ flags: Set<String>) -> Bool {
        words.contains { w in
            let name = w.split(separator: "=", maxSplits: 1).first.map(String.init) ?? w
            if flags.contains(name) { return true }
            // a cluster of one-letter flags: -uo, -ofile
            guard w.hasPrefix("-"), !w.hasPrefix("--"), w.count > 2 else { return false }
            return flags.contains { $0.count == 2 && !$0.hasPrefix("--") && w.dropFirst().contains($0.last!) }
        }
    }

    /// Whether `command` only reads: every piped / chained part starts with a known read-only command named
    /// without a path, with no flag that writes or runs programs, no redirection to files, command or process
    /// substitution, zsh glob qualifiers, background jobs or sudo. Such commands run without asking when the
    /// user allows it (Settings → Agent), and even then only inside the sandbox (`sandboxProfile`): this
    /// classifier decides when not to ask, the sandbox is what stops a command it got wrong.
    public static func isReadOnly(_ command: String) -> Bool {
        let c = command.trimmingCharacters(in: .whitespacesAndNewlines)
        if c.isEmpty || c.contains("\n") { return false }
        for bad in [">", "<(", "=(", "$(", "`", "sudo", "eval ", "exec ", "xargs"] where c.contains(bad) {
            return false
        }
        // zsh glob qualifiers run code: *(e:...:), ?(+cmd), x](...)
        if c.range(of: #"[*?\]]\("#, options: .regularExpression) != nil { return false }
        // split on pipes and chains
        var parts = [c]
        for sep in ["&&", "||", ";", "|"] { parts = parts.flatMap { $0.components(separatedBy: sep) } }
        for part in parts {
            if part.contains("&") { return false }          // a background job (`&&` was split off above)
            let words = part.split(whereSeparator: { $0 == " " || $0 == "\t" }).map(String.init)
            guard let name = words.first else { return false }
            if name.contains("=") { return false }         // VAR=... cmd
            if name.contains("/") { return false }         // ./cat, bin/ls: maybe a planted program, not the real one
            let args = words.dropFirst()
            if name == "git" {
                guard let sub = args.first, readOnlyGit.contains(sub) else { return false }
                if hasFlag(args.dropFirst(), ["--output", "-O"]) { return false }
                if ["branch", "tag", "remote"].contains(sub),
                   args.dropFirst().contains(where: { !$0.hasPrefix("-") || ["-d", "-D", "-m", "-M", "--delete"].contains($0) }) {
                    return false                           // e.g. `git branch new`, `git tag -d v1`
                }
                continue
            }
            if name == "find" {
                if args.contains(where: { ["-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprint", "-fprint0",
                                           "-fprintf", "-fls"].contains($0) }) { return false }
                continue
            }
            guard readOnlyCommands.contains(name) else { return false }
            if let denied = deniedFlags[name], hasFlag(args, denied) { return false }
            if name == "uniq", args.filter({ !$0.hasPrefix("-") }).count > 1 { return false }   // uniq in out
        }
        return true
    }

    // MARK: - sandbox

    /// The sandbox for commands that run without asking: no network, no Apple Events (no scripting other apps),
    /// no screenshots, writes only to this session's own directory, and of the home directory only the workspace
    /// can be read (never ~/.ssh, ~/.aws, ~/.gnupg, keychains, even when the workspace is the home directory).
    /// Paths are resolved first: the sandbox matches real paths (/var is /private/var).
    static func sandboxProfile(workspace: URL, session: URL) -> String {
        func real(_ u: URL) -> String { quoted(realPath(u)) }
        let home = URL(fileURLWithPath: realPath(FileManager.default.homeDirectoryForCurrentUser))
        let secrets = [".ssh", ".aws", ".gnupg", ".config", ".netrc", ".docker", ".kube", "Library/Keychains",
                       "Library/Cookies", "Library/Application Support/Google/Chrome", "Library/Mail"]
            .map { "(subpath \(quoted(home.appendingPathComponent($0).path)))" }.joined(separator: " ")
        return """
            (version 1)
            (allow default)
            (deny network*)
            (deny appleevent-send)
            (deny process-exec (literal "/usr/sbin/screencapture"))
            (deny file-write*)
            (allow file-write* (subpath \(real(session))) (literal "/dev/null"))
            (deny file-read-data (subpath \(quoted(home.path))))
            (allow file-read-data (subpath \(real(workspace))) (subpath \(real(session))))
            (deny file-read-data \(secrets))
            """
    }

    /// The path with every symlink resolved, as the sandbox sees it. Not `resolvingSymlinksInPath()`: that turns
    /// /private/var back into /var, which the sandbox does not match.
    static func realPath(_ u: URL) -> String {
        guard let r = realpath(u.path, nil) else { return u.standardizedFileURL.path }
        defer { free(r) }
        return String(cString: r)
    }

    /// A string literal for a sandbox profile.
    static func quoted(_ s: String) -> String {
        "\"" + s.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\"") + "\""
    }
}
