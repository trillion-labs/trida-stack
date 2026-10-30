import AppKit
import ApplicationServices
import Foundation
import TridaChatCore

/// One element of the target app's window, as the macOS accessibility tree describes it.
struct UIElement {
    let n: Int
    let ax: AXUIElement
    let role: String
    let roleDescription: String
    let label: String
    let state: String
    let frame: CGRect
    let isTextEntry: Bool
    let isSecure: Bool

    /// What the decision head sees for this element, like the screen-agent demo's labels:
    /// "button 'Send'", "checkbox 'Dark mode' (on)", "text field 'To' (empty)".
    var describe: String {
        let name = label.isEmpty ? roleDescription : "\(roleDescription) '\(label)'"
        return state.isEmpty ? name : "\(name) (\(state))"
    }
}

/// Reads and operates the app the user is working in (the last app other than this one that was
/// frontmost) through the Accessibility API: list its elements, press one, type, send keys.
/// macOS asks once for the Accessibility permission (System Settings → Privacy & Security).
@MainActor
final class UIAgent {
    static let shared = UIAgent()

    private(set) var target: NSRunningApplication? {
        didSet { onTargetChange?(targetName) }
    }
    /// Called with the target app's name whenever it changes.
    var onTargetChange: ((String) -> Void)?
    private var observer: NSObjectProtocol?

    private init() {
        let me = ProcessInfo.processInfo.processIdentifier
        if let front = NSWorkspace.shared.frontmostApplication, front.processIdentifier != me {
            target = front
        }
        observer = NSWorkspace.shared.notificationCenter.addObserver(
            forName: NSWorkspace.didActivateApplicationNotification, object: nil, queue: .main
        ) { [weak self] note in
            guard let app = note.userInfo?[NSWorkspace.applicationUserInfoKey] as? NSRunningApplication,
                  app.processIdentifier != me else { return }
            MainActor.assumeIsolated { self?.target = app }
        }
    }

    var targetName: String { target?.localizedName ?? "no app" }

    static var trusted: Bool { AXIsProcessTrusted() }

    /// Shows macOS's "allow Accessibility" prompt (once; afterwards it opens nothing).
    static func requestTrust() {
        let key = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
        _ = AXIsProcessTrustedWithOptions([key: true] as CFDictionary)
    }

    static func openAccessibilitySettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility") {
            NSWorkspace.shared.open(url)
        }
    }

    // MARK: - accessibility helpers

    private func value(_ el: AXUIElement, _ name: String) -> CFTypeRef? {
        var v: CFTypeRef?
        guard AXUIElementCopyAttributeValue(el, name as CFString, &v) == .success else { return nil }
        return v
    }

    private func string(_ el: AXUIElement, _ name: String) -> String? {
        (value(el, name) as? String).flatMap { $0.isEmpty ? nil : $0 }
    }

    private func bool(_ el: AXUIElement, _ name: String) -> Bool? {
        guard let v = value(el, name) else { return nil }
        if let b = v as? Bool { return b }
        if let n = v as? NSNumber { return n.boolValue }
        return nil
    }

    private func element(_ el: AXUIElement, _ name: String) -> AXUIElement? {
        guard let v = value(el, name), CFGetTypeID(v) == AXUIElementGetTypeID() else { return nil }
        return (v as! AXUIElement)
    }

    private func children(_ el: AXUIElement, _ name: String = "AXChildren") -> [AXUIElement] {
        (value(el, name) as? [AXUIElement]) ?? []
    }

    private func frame(_ el: AXUIElement) -> CGRect {
        var p = CGPoint.zero, s = CGSize.zero
        if let v = value(el, "AXPosition"), CFGetTypeID(v) == AXValueGetTypeID() {
            _ = AXValueGetValue(v as! AXValue, .cgPoint, &p)
        }
        if let v = value(el, "AXSize"), CFGetTypeID(v) == AXValueGetTypeID() {
            _ = AXValueGetValue(v as! AXValue, .cgSize, &s)
        }
        return CGRect(origin: p, size: s)
    }

    private func actions(_ el: AXUIElement) -> [String] {
        var names: CFArray?
        guard AXUIElementCopyActionNames(el, &names) == .success else { return [] }
        return (names as? [String]) ?? []
    }

    /// First static text below `el` (a row's or a cell's visible label).
    private func innerText(_ el: AXUIElement, depth: Int = 0) -> String? {
        if depth > 3 { return nil }
        for c in children(el) {
            if string(c, "AXRole") == "AXStaticText", let t = string(c, "AXValue") ?? string(c, "AXTitle") { return t }
            if let t = innerText(c, depth: depth + 1) { return t }
        }
        return nil
    }

    static let clickableRoles: Set<String> = [
        "AXButton", "AXCheckBox", "AXRadioButton", "AXPopUpButton", "AXMenuButton", "AXLink", "AXTextField",
        "AXTextArea", "AXComboBox", "AXDisclosureTriangle", "AXIncrementor", "AXSlider", "AXMenuItem", "AXRow",
        "AXCell", "AXMenuBarItem", "AXTab",
    ]
    static let textRoles: Set<String> = ["AXTextField", "AXTextArea", "AXComboBox"]

    private func short(_ s: String, _ n: Int = 60) -> String {
        let one = s.replacingOccurrences(of: "\n", with: " ").trimmingCharacters(in: .whitespaces)
        return one.count > n ? String(one.prefix(n)) + "…" : one
    }

    private func describeState(_ el: AXUIElement, role: String, subrole: String, secure: Bool) -> String {
        var parts: [String] = []
        if Self.textRoles.contains(role) {
            if secure {
                parts.append("password field")
            } else if let v = string(el, "AXValue") {
                parts.append("contains '\(short(v, 40))'")
            } else {
                parts.append("empty")
            }
        } else if role == "AXCheckBox" || subrole == "AXSwitch" {
            if let n = value(el, "AXValue") as? NSNumber { parts.append(n.intValue != 0 ? "on" : "off") }
        } else if role == "AXRadioButton" || role == "AXTab" {
            if let n = value(el, "AXValue") as? NSNumber, n.intValue != 0 { parts.append("selected") }
        }
        if bool(el, "AXSelected") == true, !parts.contains("selected") { parts.append("selected") }
        if bool(el, "AXFocused") == true { parts.append("focused") }
        return parts.joined(separator: ", ")
    }

    // MARK: - reading the screen

    struct Screen {
        var app: String
        var window: String
        var elements: [UIElement]

        /// Numbered list for the model: "[3] button 'Send'".
        var listing: String {
            elements.isEmpty ? "(no clickable elements found)"
                : elements.map { "[\($0.n)] \($0.describe)" }.joined(separator: "\n")
        }
    }

    /// The target app's focused window (plus its menu bar), at most `limit` elements.
    func screen(limit: Int = 90) -> Screen? {
        guard let app = target, !app.isTerminated else { return nil }
        let axApp = AXUIElementCreateApplication(app.processIdentifier)
        AXUIElementSetMessagingTimeout(axApp, 1.0)
        let window = element(axApp, "AXFocusedWindow") ?? children(axApp, "AXWindows").first
        var out: [UIElement] = []
        var seen = Set<String>()

        func add(_ el: AXUIElement, role: String) {
            let subrole = string(el, "AXSubrole") ?? ""
            let f = frame(el)
            guard f.width >= 2, f.height >= 2, bool(el, "AXEnabled") != false else { return }
            let secure = subrole == "AXSecureTextField"
            var label = string(el, "AXTitle") ?? string(el, "AXDescription") ?? ""
            if label.isEmpty, !Self.textRoles.contains(role), let v = string(el, "AXValue") { label = v }
            if label.isEmpty { label = string(el, "AXPlaceholderValue") ?? string(el, "AXHelp") ?? innerText(el) ?? "" }
            let roleDesc = string(el, "AXRoleDescription") ?? role.replacingOccurrences(of: "AX", with: "").lowercased()
            let item = UIElement(n: out.count + 1, ax: el, role: role, roleDescription: roleDesc, label: short(label),
                                 state: describeState(el, role: role, subrole: subrole, secure: secure), frame: f,
                                 isTextEntry: Self.textRoles.contains(role), isSecure: secure)
            let key = "\(item.describe)@\(Int(f.midX)),\(Int(f.midY))"
            if seen.insert(key).inserted { out.append(item) }
        }

        func walk(_ el: AXUIElement, depth: Int) {
            if out.count >= limit || depth > 30 { return }
            let role = string(el, "AXRole") ?? ""
            if Self.clickableRoles.contains(role) || actions(el).contains("AXPress") {
                add(el, role: role)
                if role == "AXRow" || role == "AXCell" || role == "AXMenuItem" { return }   // don't list their insides too
            }
            for c in children(el) { walk(c, depth: depth + 1) }
        }

        if let window { walk(window, depth: 0) }
        if let bar = element(axApp, "AXMenuBar") {
            for item in children(bar).dropFirst() where out.count < limit + 12 {   // skip the Apple menu
                add(item, role: "AXMenuBarItem")
            }
        }
        let title = window.flatMap { string($0, "AXTitle") } ?? ""
        return Screen(app: app.localizedName ?? "app", window: title, elements: out)
    }

    // MARK: - acting

    func activateTarget() async {
        guard let app = target else { return }
        if NSWorkspace.shared.frontmostApplication?.processIdentifier != app.processIdentifier {
            app.activate()
            try? await Task.sleep(nanoseconds: 350_000_000)
        }
    }

    /// AXPress, or a real mouse click at the element's centre when it has no press action.
    func press(_ e: UIElement) async -> Bool {
        await activateTarget()
        if AXUIElementPerformAction(e.ax, "AXPress" as CFString) == .success { return true }
        return click(at: CGPoint(x: e.frame.midX, y: e.frame.midY))
    }

    func click(at p: CGPoint) -> Bool {
        guard let down = CGEvent(mouseEventSource: nil, mouseType: .leftMouseDown, mouseCursorPosition: p,
                                 mouseButton: .left),
              let up = CGEvent(mouseEventSource: nil, mouseType: .leftMouseUp, mouseCursorPosition: p,
                               mouseButton: .left) else { return false }
        down.post(tap: .cghidEventTap)
        usleep(40_000)
        up.post(tap: .cghidEventTap)
        return true
    }

    /// Whether keyboard focus is in a password field of the target app (typing there is refused).
    func focusIsSecure() -> Bool {
        guard let app = target else { return false }
        let axApp = AXUIElementCreateApplication(app.processIdentifier)
        guard let f = element(axApp, "AXFocusedUIElement") else { return false }
        return string(f, "AXSubrole") == "AXSecureTextField"
    }

    /// Types `text` into `field` (focused first), or wherever the keyboard focus is.
    func type(_ text: String, into field: UIElement?) async -> String? {
        await activateTarget()
        if let field {
            if field.isSecure { return "refused: that is a password field; ask the user to type it" }
            AXUIElementSetAttributeValue(field.ax, "AXFocused" as CFString, kCFBooleanTrue)
            try? await Task.sleep(nanoseconds: 120_000_000)
        }
        if focusIsSecure() { return "refused: the focused field is a password field; ask the user to type it" }
        let units = Array(text.utf16)
        var i = 0
        while i < units.count {
            var chunk = Array(units[i..<min(i + 16, units.count)])
            for down in [true, false] {
                guard let ev = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: down) else { return "could not create key events" }
                ev.keyboardSetUnicodeString(stringLength: chunk.count, unicodeString: &chunk)
                ev.post(tap: .cghidEventTap)
            }
            usleep(15_000)
            i += 16
        }
        return nil
    }

    static let keyCodes: [String: CGKeyCode] = [
        "return": 36, "enter": 36, "tab": 48, "space": 49, "delete": 51, "backspace": 51, "esc": 53, "escape": 53,
        "left": 123, "right": 124, "down": 125, "up": 126, "home": 115, "end": 119, "pageup": 116, "pagedown": 121,
        "forwarddelete": 117,
        "a": 0, "s": 1, "d": 2, "f": 3, "h": 4, "g": 5, "z": 6, "x": 7, "c": 8, "v": 9, "b": 11, "q": 12, "w": 13,
        "e": 14, "r": 15, "y": 16, "t": 17, "1": 18, "2": 19, "3": 20, "4": 21, "6": 22, "5": 23, "=": 24, "9": 25,
        "7": 26, "-": 27, "8": 28, "0": 29, "]": 30, "o": 31, "u": 32, "[": 33, "i": 34, "p": 35, "l": 37, "j": 38,
        "'": 39, "k": 40, ";": 41, "\\": 42, ",": 43, "/": 44, "n": 45, "m": 46, ".": 47, "`": 50,
    ]

    /// "cmd+s", "shift+tab", "return", "cmd+shift+n". Returns an error message or nil.
    func key(_ combo: String) async -> String? {
        let parts = combo.lowercased().split(separator: "+").map { $0.trimmingCharacters(in: .whitespaces) }
        guard let last = parts.last, let code = Self.keyCodes[last] else { return "unknown key in '\(combo)'" }
        var flags: CGEventFlags = []
        for m in parts.dropLast() {
            switch m {
            case "cmd", "command", "⌘": flags.insert(.maskCommand)
            case "shift", "⇧": flags.insert(.maskShift)
            case "opt", "option", "alt", "⌥": flags.insert(.maskAlternate)
            case "ctrl", "control", "⌃": flags.insert(.maskControl)
            default: return "unknown modifier '\(m)' in '\(combo)'"
            }
        }
        await activateTarget()
        for down in [true, false] {
            guard let ev = CGEvent(keyboardEventSource: nil, virtualKey: code, keyDown: down) else { return "could not create key events" }
            ev.flags = flags
            ev.post(tap: .cghidEventTap)
        }
        return nil
    }

    /// Opens (or brings forward) an app by name and makes it the target.
    func openApp(_ name: String) async -> String? {
        let wanted = name.lowercased().replacingOccurrences(of: ".app", with: "")
        let fm = FileManager.default
        var found: URL?
        for dir in ["/Applications", "/System/Applications", "/System/Applications/Utilities",
                    fm.homeDirectoryForCurrentUser.appendingPathComponent("Applications").path] {
            for item in (try? fm.contentsOfDirectory(atPath: dir)) ?? [] where item.hasSuffix(".app") {
                if item.dropLast(4).lowercased() == wanted { found = URL(fileURLWithPath: dir).appendingPathComponent(item) }
            }
            if found != nil { break }
        }
        guard let url = found else { return "no app named '\(name)' in Applications" }
        do {
            let app = try await NSWorkspace.shared.openApplication(at: url, configuration: NSWorkspace.OpenConfiguration())
            target = app
            try? await Task.sleep(nanoseconds: 1_200_000_000)
            return nil
        } catch {
            return "could not open \(name): \(error.localizedDescription)"
        }
    }

    /// PNG of the target app's frontmost window (screencapture -l), for decisions that look at pixels.
    func windowScreenshot() async -> Data? {
        guard let pid = target?.processIdentifier,
              let info = CGWindowListCopyWindowInfo([.optionOnScreenOnly, .excludeDesktopElements], kCGNullWindowID)
                as? [[String: Any]],
              let win = info.first(where: { ($0[kCGWindowOwnerPID as String] as? Int32) == pid
                                            && ($0[kCGWindowLayer as String] as? Int) == 0 }),
              let wid = win[kCGWindowNumber as String] as? Int else { return nil }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("zoro-win-\(UUID().uuidString).png")
        defer { try? FileManager.default.removeItem(at: url) }
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/sbin/screencapture")
        p.arguments = ["-x", "-o", "-l", String(wid), "-t", "png", url.path]
        do { try p.run() } catch { return nil }
        p.waitUntilExit()
        return AgentTools.pngData(fromFile: url, maxSide: 896)
    }
}
