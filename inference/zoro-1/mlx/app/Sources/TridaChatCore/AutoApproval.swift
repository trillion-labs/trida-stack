import Foundation

/// Auto mode (as in Claude Code): writes and commands run without asking, except risky ones. A fixed list of
/// risky patterns always asks; everything else is judged by the decision model (the app asks when it is not
/// confident the action is safe).
public enum AutoApproval {
    /// The decision model's P(safe) at or above this runs the action; below it, the app asks.
    public static let threshold = 0.8

    public static let question = "Is it safe to run this action without asking the user? Safe means: it does "
        + "what the user asked for (or is a clear step toward it), and it does not delete or overwrite data the "
        + "user may want to keep outside the files the request is about, change system or account settings, "
        + "install or remove software, or send the user's data off this computer."

    /// Shell commands (or parts of a pipeline) that always ask, with the reason shown in the prompt.
    static let shellRules: [(pattern: String, reason: String)] = [
        (#"(^|[\s;&|(])(sudo|su|doas)\s"#, "runs as administrator"),
        (#"(^|[\s;&|(])(rm|rmdir|unlink|shred|trash|srm)\s"#, "deletes files"),
        (#"(^|[\s;&|(])mv\s"#, "moves or overwrites files"),
        (#"-delete\b|\bxargs\s+(-\S+\s+)*rm\b"#, "deletes files"),
        (#"\bgit\s+(push|reset\s+--hard|clean|checkout\s+--|restore|rebase|filter-branch|branch\s+-D|stash\s+(drop|clear))"#,
         "changes git history or discards changes"),
        (#"--force\b|\s-f\s*$"#, "forces an operation"),
        (#"(curl|wget)[^|]*\|\s*(sudo\s+)?(sh|bash|zsh|python3?)\b"#, "runs a script from the internet"),
        (#"(curl|wget)\s.*(\s-d\s|--data|\s-F\s|--form|\s-T\s|--upload-file|-X\s*(POST|PUT))"#, "sends data off this Mac"),
        (#"\b(scp|rsync|sftp|ssh|ftp|nc|ncat)\s"#, "connects to another machine"),
        (#"\b(kill|killall|pkill)\s"#, "stops processes"),
        (#"\b(chmod|chown|chflags|xattr)\s"#, "changes file permissions"),
        (#"\b(launchctl|defaults\s+(write|delete)|systemsetup|networksetup|scutil|csrutil|spctl|pmset|tccutil)\b"#,
         "changes system settings"),
        (#"\b(shutdown|reboot|halt)\b"#, "restarts or shuts down the Mac"),
        (#"\b(dd|mkfs|diskutil|hdiutil\s+(erase|detach))\b"#, "works on disks"),
        (#"\b(brew|pip3?|npm|pnpm|yarn|gem|cargo|uv\s+pip)\s+(uninstall|remove|rm)\b"#, "removes software"),
        (#"\b(npm|pnpm|yarn)\s+publish\b|\btwine\s+upload\b|\bgh\s+(pr|release|repo)\s+(create|delete|merge)"#,
         "publishes something"),
        (#"\bosascript\b"#, "controls other apps"),
        (#"(^|[\s;&|])>\s*~?/(etc|usr|System|Library|bin|sbin)/|~/\.[A-Za-z]"#, "writes outside the project"),
        (#"\beval\b|\bsource\s+<\(|\bbase64\s+(-d|--decode)"#, "runs generated code"),
    ]

    /// Why the call always asks in auto mode, or nil when the decision model may judge it.
    public static func risk(_ call: ToolCall, tools: AgentTools) -> String? {
        let a = call.argumentDict
        switch call.name {
        case "screenshot":
            return "it captures your screen"
        case "fetch_url":
            guard let u = WebFetch.url(a["url"] as? String ?? "") else { return "it is not a web address" }
            return WebFetch.isPrivate(u) ? "it reaches a computer on your own network (\(u.host ?? ""))" : nil
        case "run_shell":
            let c = a["command"] as? String ?? ""
            for r in shellRules where c.range(of: r.pattern, options: .regularExpression) != nil { return r.reason }
            return nil
        case "write_file":
            guard let p = a["path"] as? String, let url = try? tools.path(p) else { return "the path is not valid" }
            if tools.inScratch(url) { return nil }
            let rel = url.path.dropFirst(tools.root.path.count)
            if rel.split(separator: "/").contains(where: { $0.hasPrefix(".") }) {
                return "it writes a hidden file or folder (\(url.lastPathComponent))"
            }
            return nil
        default:
            return nil
        }
    }
}
