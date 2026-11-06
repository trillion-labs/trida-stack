import AppKit
import SwiftUI
import TridaChatCore

/// What the mascot shows: what the model is doing, or how it reacts to you.
enum MascotState: Equatable {
    // the model
    case offline, idle, watching, sleepy, reading, thinking, deciding, tool(String), answering, checking, tidying,
         done, error
    // reactions (a few seconds each)
    case love, notes, scribble, wave, hearts

    var caption: String {
        switch self {
        case .offline: return "Waiting for the server"
        case .idle: return "Ready"
        case .watching: return "Listening"
        case .sleepy: return "Dozing"
        case .reading: return "Reading your message"
        case .thinking: return "Thinking"
        case .deciding: return "Deciding"
        case let .tool(name): return MascotProp.of(name).caption
        case .answering: return "Answering"
        case .checking: return "Checking the answer"
        case .tidying: return "Tidying up the chat"
        case .done: return "Done"
        case .error: return "Something went wrong"
        case .love: return "Glad that helped"
        case .notes: return "Noted, I'll learn from it"
        case .scribble: return "Learning from your edit"
        case .wave: return "Hi!"
        case .hearts: return "Hehe"
        }
    }
}

/// What the mascot holds while a tool runs.
enum MascotProp {
    case magnifier, page, folder, terminal, pen, camera, globe, wrench

    static func of(_ tool: String) -> MascotProp {
        switch tool {
        case "search_files", "find_files", "locate_files", "pick_file": return .magnifier
        case "read_file", "read_pdf", "view_image": return .page
        case "list_dir": return .folder
        case "run_shell": return .terminal
        case "write_file": return .pen
        case "screenshot": return .camera
        case "fetch_url", "find_in_page", "read_section", "open_link": return .globe
        default: return .wrench
        }
    }

    var caption: String {
        switch self {
        case .magnifier: return "Searching files"
        case .page: return "Reading a file"
        case .folder: return "Looking in a folder"
        case .terminal: return "Running a command"
        case .pen: return "Writing a file"
        case .camera: return "Taking a screenshot"
        case .globe: return "Reading a web page"
        case .wrench: return "Using a tool"
        }
    }
}

/// Everything that decides a frame besides the time: the state, the accessories earned by labeling decisions
/// (trainer level), the hour (nightcap, morning coffee) and whether drafts were being accepted at full speed.
struct MascotLook: Equatable {
    var state: MascotState
    var level = 0
    var hour = 12
    var fast = false
}

enum MascotReaction { case love, notes, scribble, wave, hearts }

extension AppState {
    /// Derived from what the app already tracks: server status, compaction, the prompt prefill, the latest
    /// reply's routing note, tool runs, content and review line, running /decide cards, reactions to the
    /// user's feedback and clicks, the composer, and how long it has been quiet.
    var mascotState: MascotState {
        switch status {
        case .failed: return .error
        case .starting, .stopped: return .offline
        default: break
        }
        let last = currentChatIndex.flatMap { chats[$0].messages.last { $0.role == .assistant } }
        if isCompacting { return .tidying }
        if decisionCards.values.contains(where: { $0.running }) { return .deciding }
        if let c = last?.check, c.hasPrefix("Checking the reply") { return .checking }
        if isGenerating {
            if prefill != nil { return .reading }
            if let m = last {
                if let r = m.route, r.hasPrefix("Choosing tools") { return .deciding }
                if let run = m.toolRuns.last(where: { $0.status == .running || $0.status == .waitingForApproval }) {
                    return DecisionTools.names.contains(run.call.name) && run.call.name != "locate_files"
                        ? .deciding : .tool(run.call.name)
                }
                if !m.content.isEmpty { return .answering }
            }
            return .thinking
        }
        if let r = mascotReaction, Date().timeIntervalSince(r.at) < 3 {
            switch r.kind {
            case .love: return .love
            case .notes: return .notes
            case .scribble: return .scribble
            case .wave: return .wave
            case .hearts: return .hearts
            }
        }
        if let e = last?.error, e != "Stopped." { return .error }
        if let t = lastFinished, Date().timeIntervalSince(t) < 2.5 { return .done }
        if !draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { return .watching }
        if Date().timeIntervalSince(lastActivity) > 300 { return .sleepy }
        return .idle
    }

    var mascotLook: MascotLook {
        MascotLook(state: mascotState, level: MascotTrainer.level(labeledDecisionCount),
                   hour: Calendar.current.component(.hour, from: Date()), fast: (lastAcceptance ?? 0) >= 0.7)
    }

    /// A short line under the caption, with the numbers behind the state.
    var mascotDetail: String? {
        let last = currentChatIndex.flatMap { chats[$0].messages.last { $0.role == .assistant } }
        switch mascotState {
        case .reading:
            return prefill.map { "\($0.total.formatted()) tokens to read" }
        case .tool:
            guard let run = last?.toolRuns.last(where: { $0.status == .running || $0.status == .waitingForApproval })
            else { return nil }
            let a = run.call.argumentDict
            let arg = (a["query"] ?? a["path"] ?? a["command"] ?? a["pattern"] ?? a["url"] ?? a["heading"] ?? a["link"])
                as? String ?? ""
            return arg.isEmpty ? nil : String(arg.prefix(32)) + (arg.count > 32 ? "…" : "")
        case .done:
            guard let st = last?.stats else { return nil }
            if st.mode.hasPrefix("self-spec"), st.tokensPerForward > 0 {
                return String(format: "%.1f tokens per step", st.tokensPerForward)
            }
            return String(format: "%.0f tokens a second", st.decodeTokensPerSecond)
        case .tidying: return "Summarizing the earlier conversation"
        case .sleepy: return "Type to wake me"
        default: return nil
        }
    }

    /// The mascot was clicked: a wave, or hearts after a few quick clicks.
    func petMascot() {
        let now = Date()
        mascotPets = mascotPets.filter { now.timeIntervalSince($0) < 2 } + [now]
        mascotReaction = (mascotPets.count >= 3 ? .hearts : .wave, now)
        lastActivity = now
    }

    func react(_ kind: MascotReaction) { mascotReaction = (kind, Date()) }

    /// Tokens kept per forward pass over the draft length N ("self-spec(N=4)"), for the last reply.
    func acceptanceOfLastReply() -> Double? {
        guard let st = currentChatIndex.flatMap({ chats[$0].messages.last { $0.role == .assistant } })?.stats,
              st.tokensPerForward > 0,
              let r = st.mode.range(of: #"N=(\d+)"#, options: .regularExpression),
              let n = Double(st.mode[r].dropFirst(2)), n > 0 else { return nil }
        return st.tokensPerForward / n
    }
}

/// Labeling decisions (the "N decisions" under a reply) levels the mascot up; each level adds an accessory.
enum MascotTrainer {
    static let thresholds = [5, 15, 30, 60]
    static let accessories = ["a scarf", "glasses", "a leaf crown", "a golden Z"]

    static func level(_ marked: Int) -> Int { thresholds.filter { marked >= $0 }.count }

    static func progress(_ marked: Int) -> String {
        let l = level(marked)
        guard l < thresholds.count else { return "Level \(l): every accessory unlocked" }
        let need = thresholds[l] - marked
        return "Level \(l). Mark \(need) more decision answer\(need == 1 ? "" : "s") for \(accessories[l])"
    }
}

/// Small achievements, from what is already in the chats (plus the days the app was used).
struct MascotBadge: Identifiable {
    let id: String
    let title: String
    let detail: String
    let earned: Bool

    @MainActor static func all(_ s: AppState) -> [MascotBadge] {
        let msgs = s.chats.flatMap(\.messages)
        let replies = msgs.filter { $0.role == .assistant && $0.stats != nil }.count
        let runs = msgs.flatMap(\.toolRuns)
        let found = runs.contains { $0.call.name == "locate_files" && $0.status == .done }
        let pdf = runs.contains { $0.call.name == "read_pdf" && $0.status == .done }
        let undone = runs.contains { $0.undone == true }
        let marked = s.labeledDecisionCount
        let days = UsageDays.count
        return [
            MascotBadge(id: "first", title: "Hello there", detail: "Get a first reply", earned: replies >= 1),
            MascotBadge(id: "hundred", title: "Chatterbox", detail: "100 replies (\(min(replies, 100))/100)", earned: replies >= 100),
            MascotBadge(id: "finder", title: "Found it", detail: "Find a file with locate_files", earned: found),
            MascotBadge(id: "pdf", title: "Bookworm", detail: "Read a PDF", earned: pdf),
            MascotBadge(id: "undo", title: "Second thoughts", detail: "Undo a file write", earned: undone),
            MascotBadge(id: "teacher", title: "Teacher", detail: "Mark a decision answer", earned: marked >= 1),
            MascotBadge(id: "week", title: "A week together", detail: "Use the app on 7 days (\(min(days, 7))/7)", earned: days >= 7),
        ]
    }
}

/// The days the app was used (for the "A week together" badge), kept in UserDefaults.
enum UsageDays {
    static let key = "TridaChat.usageDays"

    static func record(_ date: Date = Date()) {
        let f = DateFormatter()
        f.dateFormat = "yyyy-MM-dd"
        var days = UserDefaults.standard.stringArray(forKey: key) ?? []
        let today = f.string(from: date)
        if !days.contains(today) { days = Array((days + [today]).suffix(400)) }
        UserDefaults.standard.set(days, forKey: key)
    }

    static var count: Int { (UserDefaults.standard.stringArray(forKey: key) ?? []).count }
}

/// The Zoro-1 mascot: a 32 x 31 pixel figure (head, bent arms, crossed legs, a Z on its belly), drawn on a
/// Canvas and animated at 6 frames per second by moving its parts (eyes, mouth, hands, the whole body) and a few
/// effect and prop pixels around it. With Reduce Motion on, it holds one still pose per state.
struct MascotView: View {
    let look: MascotLook
    var fps = 6.0

    var body: some View {
        TimelineView(.periodic(from: .now, by: 1 / fps)) { tl in
            let still = NSWorkspace.shared.accessibilityDisplayShouldReduceMotion
            MascotFrame(look: look, step: still ? 0 : Int(tl.date.timeIntervalSinceReferenceDate * fps))
        }
        .accessibilityElement()
        .accessibilityLabel("Zoro-1: \(look.state.caption)")
    }
}

/// One frame of the mascot (also what the Dock icon is drawn from).
struct MascotFrame: View {
    let look: MascotLook
    let step: Int

    // '#' body, 'g' highlight, 'E' eyes, 'M' mouth, 'Z' the belly letter
    static let sprite: [[Character]] = ([
        ".........gg..........gg.........",
        "........g###........###g........",
        ".......g####........####g.......",
        "......g######......######g......",
        ".....g########....########g.....",
        ".....##########..##########.....",
        "......####################......",
        ".......##################.......",
        ".......####EE######EE####.......",
        "........###EE#M##M#EE###........",
        ".........######MM######.........",
        ".........##############.........",
        "................................",
        "..ggggg#................#ggggg..",
        ".g#######....ZZZZZZ....#######g.",
        "##########.......ZZ...##########",
        "###########....ZZ....###########",
        "###########..ZZ......###########",
        "############.ZZZZZZ.############",
        ".###########........###########.",
        ".........###........###.........",
        "..........###......###..........",
        ".........#####....#####.........",
        "........#######..#######........",
        "........################........",
        ".......##################.......",
        "......########....########......",
        "......########....########......",
        ".....########......########.....",
        "....#ggg####........####ggg#....",
        "....########........########...."
    ] as [String]).map(Array.init)

    static let body = Theme.dynamic(light: 0x1C2B23, dark: 0xD6E4DC)
    static let highlight = Theme.dynamic(light: 0x5D7066, dark: 0xFFFFFF)
    static let face = Theme.dynamic(light: 0xF6F1E3, dark: 0x0F1F19)
    static let heart = Theme.dynamic(light: 0xD9485F, dark: 0xF2778A)
    static let gold = Theme.dynamic(light: 0xC9962B, dark: 0xF0C25A)
    static let scarf = Theme.dynamic(light: 0xC2410C, dark: 0xF08A5D)

    /// Room around the sprite for effects: 9 rows above (the thought bubble), 2 columns each side, 1 row below.
    static let margin = (top: 9, side: 2, bottom: 1)

    var body: some View {
        Canvas { gc, size in draw(Pose(look: look, step: step), in: gc, size: size) }
    }

    typealias Px = (row: Int, col: Int, color: Color)

    // A frame: how far each part is moved, which face, and the effect pixels.
    struct Pose {
        enum Eyes { case open, closed, slanted, focused, happy }
        enum Mouth { case smile, open, flat, frown, none }
        var bob = 0, head = 0, leftHand = 0, rightHand = 0
        var eyes = Eyes.open, eyeDX = 0, eyeDY = 0
        var mouth = Mouth.smile
        var zGlow = 0.85
        var effects: [Px] = []

        init(look: MascotLook, step s: Int) {
            let blink = s % 30 == 0
            switch look.state {
            case .offline:
                eyes = .closed; mouth = .flat; zGlow = 0.35
                effects = Pose.drifting(s)
            case .idle:
                bob = s / 4 % 2; eyes = blink ? .closed : .open
                if (6..<11).contains(look.hour) {                    // morning coffee in the left hand
                    leftHand = -1
                    effects = Pose.coffee(s)
                }
            case .watching:                                         // looking toward the composer, nodding
                eyeDX = 1; eyeDY = 1; eyes = blink ? .closed : .open; head = s / 6 % 2
            case .sleepy:
                eyes = .closed; mouth = .flat; bob = 1; zGlow = 0.5
                effects = Pose.drifting(s)
            case .reading:
                eyeDX = s / 3 % 2 == 0 ? -1 : 1; eyeDY = 1; mouth = .none     // the lowered eyes need the room
                leftHand = -1; rightHand = -1
                effects = Pose.page(s)
            case .thinking:
                eyeDY = -1; eyeDX = 1; mouth = .flat; bob = s / 6 % 2
                effects = Pose.thoughtBubble(dots: s / 2 % 4, float: s / 6 % 2)
            case .deciding:
                let left = s / 4 % 2 == 0                           // weighing one option against the other
                leftHand = left ? -2 : 0; rightHand = left ? 0 : -2
                eyeDX = left ? -1 : 1; mouth = .flat
                effects = Pose.square(row: 9 + (left ? -2 : 0), col: 2, color: left ? Theme.accent : Theme.muted.opacity(0.5))
                    + Pose.square(row: 9 + (left ? 0 : -2), col: 28, color: left ? Theme.muted.opacity(0.5) : Theme.accent)
                    + Pose.question(color: s / 3 % 2 == 0 ? Theme.accent : Theme.accent.opacity(0.45))
            case let .tool(name):
                mouth = .flat; eyes = .focused
                (rightHand, effects) = Pose.prop(MascotProp.of(name), s)
            case .answering:
                mouth = s % 2 == 0 ? .open : .smile; zGlow = s % 2 == 0 ? 1 : 0.6
                bob = s / 3 % 2
                if look.fast { effects = Pose.speedLines(s) }        // drafts accepted at (nearly) full speed
            case .checking:
                rightHand = -2; eyeDX = 1; mouth = .flat
                let on = s / 3 % 2 == 0
                effects = [(7, 26), (8, 27), (7, 28), (6, 29), (5, 30)].map {
                    (row: $0.0, col: $0.1, color: on ? Theme.accent : Theme.accent.opacity(0.4))
                }
            case .tidying:                                          // sweeping with a broom, dust flying
                let a = s % 2 == 0
                rightHand = a ? -1 : 0; eyes = .focused; mouth = .flat
                effects = Pose.broom(a, s)
            case .done:
                if look.fast {                                      // a quick run with speed lines
                    bob = s % 2 == 0 ? -1 : 0; leftHand = s % 2 == 0 ? -1 : 0; rightHand = s % 2 == 0 ? 0 : -1
                    eyes = .happy
                    effects = Pose.speedLines(s)
                } else {
                    bob = s % 4 < 2 ? -1 : 0; leftHand = -2; rightHand = -2
                    if s % 2 == 0 { effects = [(-1, 3, Theme.accent), (-2, 28, Theme.accent), (2, 0, Theme.warning)] }
                }
            case .error:
                eyes = .slanted; mouth = .frown; head = 1; zGlow = 0.4
                effects = [(3, 29, Theme.danger), (4, 29, Theme.danger), (6, 29, Theme.danger)]   // "!"
            case .love:
                eyes = .happy; bob = s % 2 == 0 ? -1 : 0
                effects = Pose.heart(row: -5 - s % 2, col: 14, color: MascotFrame.heart)
            case .notes:                                            // a little sad, writing it down
                eyes = .slanted; mouth = .flat; leftHand = -1; rightHand = -1
                effects = Pose.notebook(s)
            case .scribble:
                eyes = .focused; mouth = .smile; leftHand = -1; rightHand = s % 2 == 0 ? -2 : -1
                effects = Pose.notebook(s)
            case .wave:
                eyes = .happy; rightHand = s % 2 == 0 ? -3 : -1
            case .hearts:
                eyes = .happy; bob = s % 2 == 0 ? -1 : 0
                for (i, c) in [4, 14, 24].enumerated() {
                    effects += Pose.heart(row: -2 - (s + 2 * i) % 7, col: c, color: MascotFrame.heart.opacity(i == 1 ? 1 : 0.7))
                }
            }
        }

        static func px(_ cells: [(Int, Int)], _ r: Int, _ c: Int, _ color: Color) -> [Px] {
            cells.map { (row: r + $0.0, col: c + $0.1, color: color) }
        }

        static func square(row: Int, col: Int, color: Color) -> [Px] { px([(0, 0), (0, 1), (1, 0), (1, 1)], row, col, color) }

        /// A "?" above the head, rows -6 to -1.
        static func question(color: Color) -> [Px] {
            px([(-6, 15), (-6, 16), (-5, 14), (-5, 17), (-4, 17), (-3, 16), (-1, 16)], 0, 0, color)
        }

        static func heart(row: Int, col: Int, color: Color) -> [Px] {
            px([(0, 0), (0, 1), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (2, 3), (3, 2)],
               row, col, color)
        }

        static func letterZ(row: Int, col: Int, color: Color) -> [Px] {
            px([(0, 0), (0, 1), (0, 2), (1, 1), (2, 0), (2, 1), (2, 2)], row, col, color)
        }

        /// A "z" drifting up from the head (asleep, or waiting for the server).
        static func drifting(_ s: Int) -> [Px] {
            let k = s / 3 % 3
            return letterZ(row: -2 - 2 * k, col: 25 + k, color: Theme.muted.opacity(1 - Double(k) * 0.3))
        }

        /// A page held in front of the belly, its text lines scrolling up.
        static func page(_ s: Int) -> [Px] {
            var out: [Px] = []
            for r in 13...18 { for c in 10...21 { out.append((r, c, MascotFrame.face)) } }
            for i in 0..<3 {
                let r = 14 + (i * 2 + 5 - s / 2 % 5) % 5
                for c in 11...(i == 1 ? 18 : 20) { out.append((r, c, Theme.muted)) }
            }
            return out
        }

        /// A small notebook held in front of the belly, with a pen moving over it.
        static func notebook(_ s: Int) -> [Px] {
            var out: [Px] = []
            for r in 14...18 { for c in 11...20 { out.append((r, c, MascotFrame.face)) } }
            for r in [15, 17] { for c in 12...(12 + (s / 2 + r) % 7) { out.append((r, c, Theme.muted)) } }
            let pc = 14 + s % 5                                     // the pen's tip moving along the line
            out += px([(0, 0), (-1, 1), (-2, 2)], 16, pc, MascotFrame.highlight)
            return out
        }

        /// A cream cloud at the upper right with a two-dot trail from the head, filling in with dots.
        static func thoughtBubble(dots: Int, float: Int) -> [Px] {
            var out: [Px] = []
            let top = -9 + float                                    // bobs down a row: row -9 is the canvas top
            let outline = Theme.muted
            for c in 24...30 { out.append((top, c, outline)); out.append((top + 6, c, outline)) }
            for (dr, l, r) in [(1, 23, 31), (2, 22, 32), (3, 22, 32), (4, 22, 32), (5, 23, 31)] {
                out.append((top + dr, l, outline)); out.append((top + dr, r, outline))
                for c in (l + 1)..<r { out.append((top + dr, c, MascotFrame.face)) }
            }
            out.append((top + 7, 22, outline))
            out.append((top + 8, 21, outline))
            for i in 0..<dots { out += square(row: top + 2, col: 24 + 3 * i, color: Theme.accent) }
            return out
        }

        /// Motion lines streaming off the left side (fast self-spec).
        static func speedLines(_ s: Int) -> [Px] {
            var out: [Px] = []
            for (i, r) in [15, 19, 24].enumerated() {
                let len = 2 + (s + i) % 3
                for c in 0..<len { out.append((r, -2 + c - (s + i) % 2, Theme.accent.opacity(0.6 - 0.15 * Double(c)))) }
            }
            return out
        }

        /// Morning coffee: a mug at the left hand with steam rising.
        static func coffee(_ s: Int) -> [Px] {
            var out = px([(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)], 9, 1, MascotFrame.face)
            out += px([(1, 3)], 9, 1, MascotFrame.face)             // the handle
            out += px([(0, 0), (0, 1), (0, 2)], 9, 1, Theme.warning.opacity(0.8))   // coffee
            let k = s / 2 % 3
            out += px([(-1 - k, 1 + k % 2), (-2 - k, 2 - k % 2)], 9, 1, Theme.muted.opacity(0.6))
            return out
        }

        /// A broom in the right hand, sweeping; dust near the feet.
        static func broom(_ a: Bool, _ s: Int) -> [Px] {
            var out: [Px] = []
            let shift = a ? 0 : 1
            for i in 0..<9 { out.append((12 + i, 27 - i / 3 + shift, Theme.warning)) }   // the handle
            for c in 21...26 { out.append((21, c + shift, MascotFrame.gold)); out.append((22, c + shift, MascotFrame.gold)) }
            if s % 2 == 0 { out += px([(0, 0), (-1, 2), (1, 4)], 28, 31, Theme.muted.opacity(0.7)) }
            return out
        }

        /// The tool's prop near the right hand (and how high the hand is).
        static func prop(_ p: MascotProp, _ s: Int) -> (Int, [Px]) {
            let a = s % 2 == 0
            switch p {
            case .magnifier:                                        // sweeping left and right
                let c = 26 + (s / 2 % 3)
                var out = px([(0, 1), (0, 2), (1, 0), (1, 3), (2, 0), (2, 3), (3, 1), (3, 2), (4, 3), (5, 4)], 5, c,
                             MascotFrame.highlight)
                out += px([(1, 1), (1, 2), (2, 1), (2, 2)], 5, c, Theme.accentSoft)
                return (-2, out)
            case .page:
                var out: [Px] = []
                for r in 5...10 { for c in 27...31 { out.append((r, c, MascotFrame.face)) } }
                for (i, r) in [6, 8, 10].enumerated() where (s / 2 + i) % 4 != 3 {
                    for c in 28...30 { out.append((r, c, Theme.muted)) }
                }
                return (-2, out)
            case .folder:
                var out = px([(0, 0), (0, 1), (0, 2)], 6, 26, MascotFrame.gold)              // the tab
                for r in 7...10 { for c in 26...31 { out.append((r, c, MascotFrame.gold)) } }
                if a { out += px([(0, 1), (0, 2), (0, 3)], 6, 28, MascotFrame.face) }      // a sheet peeking out
                return (-2, out)
            case .terminal:
                var out: [Px] = []
                for r in 5...10 { for c in 25...32 { out.append((r, c, r == 5 || r == 10 || c == 25 || c == 32
                                                                ? Theme.muted : MascotFrame.body)) } }
                out += px([(0, 0), (1, 1), (2, 0)], 6, 26, Theme.accent)                    // ">"
                if a { out += px([(0, 0), (0, 1)], 8, 28, Theme.accent) }                   // a blinking cursor
                return (-2, out)
            case .pen:
                let d = s % 3
                var out = px([(0, 0), (1, -1), (2, -2), (3, -3)], 5 + d, 30, MascotFrame.highlight)
                out += px([(4, -4)], 5 + d, 30, Theme.accent)                               // the nib
                return (-2 + (a ? 0 : 1), out)
            case .camera:
                var out: [Px] = []
                for r in 6...9 { for c in 26...31 { out.append((r, c, Theme.muted)) } }
                out += px([(0, 0), (0, 1), (1, 0), (1, 1)], 7, 28, MascotFrame.face)          // the lens
                if s % 4 == 0 { out += px([(-2, 0), (-1, -1), (-1, 1), (-3, 0)], 5, 33, MascotFrame.gold) }   // flash
                return (-2, out)
            case .globe:                                            // a globe in the raised hand, turning
                let ring = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 4), (2, 0), (2, 4), (3, 0), (3, 4), (4, 1), (4, 2), (4, 3)]
                var out = px(ring, 5, 27, MascotFrame.highlight)
                let fill = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3)]
                out += px(fill, 5, 27, Theme.accent.opacity(0.55))
                let m = 1 + s / 2 % 3                               // the meridian moves across
                out += px([(1, m), (2, m), (3, m)], 5, 27, MascotFrame.face)
                return (-2, out)
            case .wrench:                                           // hammering, with a gear turning
                var out = px([(0, 0), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1), (4, 1)], a ? 6 : 9, 27,
                             MascotFrame.highlight)
                if !a { out += [(14, 31, Theme.warning), (13, 31, Theme.warning), (12, 30, Theme.warning)] }
                let g1 = [(0, 2), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (4, 2)]
                let g2 = [(0, 0), (0, 4), (1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3), (4, 0), (4, 4)]
                out += px(s / 2 % 2 == 1 ? g2 : g1, 6, 1, Theme.muted)
                return (a ? -3 : 0, out)
            }
        }
    }

    func draw(_ p: Pose, in gc: GraphicsContext, size: CGSize) {
        let rows = Self.sprite.count + Self.margin.top + Self.margin.bottom
        let cols = Self.sprite[0].count + 2 * Self.margin.side
        let cell = max(1, floor(min(size.width / CGFloat(cols), size.height / CGFloat(rows))))
        let ox = (size.width - cell * CGFloat(cols)) / 2 + cell * CGFloat(Self.margin.side)
        let oy = (size.height - cell * CGFloat(rows)) / 2 + cell * CGFloat(Self.margin.top)
        func fill(_ r: Int, _ c: Int, _ color: Color) {
            gc.fill(Path(CGRect(x: ox + CGFloat(c) * cell, y: oy + CGFloat(r) * cell, width: cell, height: cell)),
                    with: .color(color))
        }
        let golden = look.level >= 4
        for (r, line) in Self.sprite.enumerated() {
            for (c, ch) in line.enumerated() where ch != "." {
                var dy = p.bob
                if r <= 11 { dy += p.head }
                if (13...19).contains(r), c <= 7 { dy += p.leftHand }
                if (13...19).contains(r), c >= 24 { dy += p.rightHand }
                switch ch {
                case "g": fill(r + dy, c, Self.highlight)
                case "Z": fill(r + dy, c, (golden ? Self.gold : Theme.accent).opacity(p.zGlow))
                default: fill(r + dy, c, Self.body)            // eyes and mouth are drawn over the head below
                }
            }
        }
        let hy = p.bob + p.head
        for (r, c) in eyeCells(p) { fill(r + hy, c, Self.face) }
        for (r, c) in mouthCells(p.mouth) { fill(r + hy, c, Self.face) }
        for (r, c, color) in accessories(p) { fill(r, c, color) }
        for e in p.effects { fill(e.row + p.bob, e.col, e.color) }
    }

    /// Earned by labeling decisions (MascotTrainer), plus a nightcap late at night.
    func accessories(_ p: Pose) -> [Px] {
        var out: [Px] = []
        let hy = p.bob + p.head
        if look.level >= 1 {                                        // a scarf, its tail hanging down
            for c in 10...21 { out.append((12 + p.bob, c, Self.scarf)) }
            out += [(13 + p.bob, 20, Self.scarf), (14 + p.bob, 20, Self.scarf), (15 + p.bob, 21, Self.scarf)]
        }
        if look.level >= 2 {                                        // glasses around the eyes
            for x in [10, 18] {
                for c in x...(x + 3) { out.append((7 + hy, c, Self.gold)); out.append((10 + hy, c, Self.gold)) }
                out += [(8 + hy, x, Self.gold), (9 + hy, x, Self.gold), (8 + hy, x + 3, Self.gold), (9 + hy, x + 3, Self.gold)]
            }
            for c in 14...17 { out.append((8 + hy, c, Self.gold)) }
        }
        if look.level >= 3 {                                        // a crown of leaves on the head
            for x in [7, 21] {
                out += [(-1 + hy, x, Theme.accent), (-2 + hy, x + 1, Theme.accent), (-1 + hy, x + 2, Theme.accent),
                        (-2 + hy, x + 3, Theme.accent)]
            }
        }
        if look.hour >= 23 || look.hour < 5 {                       // a nightcap on the right lobe
            for (r, a, b) in [(0, 20, 25), (-1, 21, 25), (-2, 22, 25), (-3, 24, 26)] {
                for c in a...b { out.append((r + hy, c, Theme.accent.opacity(0.85))) }
            }
            out.append((-4 + hy, 27, Self.face))
        }
        return out
    }

    func eyeCells(_ p: Pose) -> [(Int, Int)] {
        let dx = p.eyeDX, dy = p.eyeDY
        switch p.eyes {
        case .open:
            return [11, 12, 19, 20].flatMap { c in [(8 + dy, c + dx), (9 + dy, c + dx)] }
        case .closed:
            return [11, 12, 19, 20].map { (9, $0) }
        case .slanted:                                          // sad: inner corners up
            return [(8, 12), (9, 11), (8, 19), (9, 20)]
        case .focused:                                          // one row, glancing right
            return [(9, 12), (9, 13), (9, 20), (9, 21)]
        case .happy:                                            // ^ ^
            return [(9, 11), (8, 12), (9, 13), (9, 18), (8, 19), (9, 20)]
        }
    }

    func mouthCells(_ m: Pose.Mouth) -> [(Int, Int)] {
        switch m {
        case .smile: return [(9, 14), (9, 17), (10, 15), (10, 16)]
        case .open: return [(9, 15), (9, 16), (10, 15), (10, 16)]
        case .flat: return [(10, 14), (10, 15), (10, 16), (10, 17)]
        case .frown: return [(9, 15), (9, 16), (10, 14), (10, 17)]
        case .none: return []
        }
    }

    /// The Dock icon: one still frame on a rounded sage tile.
    @MainActor static func icon(_ look: MascotLook, size: CGFloat = 512) -> NSImage? {
        let view = ZStack {
            RoundedRectangle(cornerRadius: size * 0.22, style: .continuous).fill(Color(nsColor: NSColor(hex: 0xE2E9DF)))
            MascotFrame(look: look, step: 0).padding(size * 0.08)
        }
        .frame(width: size, height: size)
        .environment(\.colorScheme, .light)
        return ImageRenderer(content: view).nsImage
    }
}

/// The mascot: the figure (click it), what it is doing, the numbers behind it, and the trainer level (click for
/// badges). It stands right beside the latest reply when the window leaves room for it (`.message`, smaller,
/// in place of the reply's avatar), in the middle of an empty chat (`.margin`), and otherwise at the bottom of
/// the sidebar (`.sidebar`).
struct MascotPanel: View {
    enum Placement { case sidebar, margin, message }

    @EnvironmentObject var state: AppState
    var placement: Placement = .sidebar

    /// Resting states move at half speed, so the figure does not pull the eye while you read.
    static func calm(_ s: MascotState) -> Bool { [.idle, .watching, .sleepy, .done].contains(s) }

    var body: some View {
        if placement == .message { beside } else { panel }
    }

    /// Beside a reply: the figure at 4 pt a cell and the caption under it (the numbers in the tooltip, the
    /// badges in the context menu).
    private var beside: some View {
        TimelineView(.periodic(from: .now, by: 0.5)) { _ in
            let look = state.mascotLook
            VStack(spacing: 2) {
                MascotView(look: look, fps: Self.calm(look.state) ? 3 : 6)
                    .frame(width: 168, height: 168)        // 36 x 41 cells at 4 pt each
                    .contentShape(Rectangle())
                    .onTapGesture { state.petMascot() }
                Text(look.state.caption)
                    .font(.callout.weight(.medium))
                    .foregroundStyle(look.state == .error ? Theme.danger : Theme.muted)
                    .multilineTextAlignment(.center)
                    .lineLimit(2)
            }
            .frame(width: ChatLayout.besideWidth)
            .help([look.state.caption, state.mascotDetail, "Level \(look.level): click me"]
                .compactMap { $0 }.joined(separator: "\n"))
            .contextMenu { Button("Level \(look.level) and Badges…") { state.showBadges = true } }
            .popover(isPresented: $state.showBadges, arrowEdge: .leading) { BadgesView() }
        }
    }

    private var panel: some View {
        TimelineView(.periodic(from: .now, by: 0.5)) { _ in      // re-derive the state (reactions and "done" expire)
            let look = state.mascotLook
            VStack(spacing: 4) {
                MascotView(look: look, fps: Self.calm(look.state) ? 3 : 6)
                    .frame(width: 168, height: 168)       // 36 x 41 cells at 4 pt each (pixels stay whole points)
                    .contentShape(Rectangle())
                    .onTapGesture { state.petMascot() }
                    .help("Click me")
                Text(look.state.caption)
                    .font(.body.weight(.semibold))
                    .foregroundStyle(look.state == .error ? Theme.danger : Theme.text)
                Text(state.mascotDetail ?? " ")
                    .font(.callout)
                    .foregroundStyle(Theme.muted)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Button { state.showBadges = true } label: {
                    Text("Level \(look.level)").font(.caption.weight(.semibold)).foregroundStyle(Theme.accent)
                }
                .buttonStyle(.plain)
                .help(MascotTrainer.progress(state.labeledDecisionCount))
                .popover(isPresented: $state.showBadges, arrowEdge: .trailing) { BadgesView() }
            }
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 12)
            .padding(.vertical, 14)
            .background(placement == .sidebar ? Theme.sidebar : Color.clear)
            .overlay(alignment: .top) {
                if placement == .sidebar { Rectangle().fill(Theme.border).frame(height: 1) }
            }
        }
    }
}

struct BadgesView: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Zoro-1's level and badges").font(.headline)
            Text(MascotTrainer.progress(state.labeledDecisionCount))
                .font(.callout).foregroundStyle(Theme.text)
            Text("Mark the right answers in a reply's decisions to level up: \(MascotTrainer.thresholds.map(String.init).joined(separator: ", ")) marked answers unlock "
                 + MascotTrainer.accessories.joined(separator: ", ") + ".")
                .font(.caption).foregroundStyle(Theme.muted).fixedSize(horizontal: false, vertical: true)
            Divider()
            ForEach(MascotBadge.all(state)) { b in
                HStack(spacing: 10) {
                    Image(systemName: b.earned ? "rosette" : "circle.dashed")
                        .foregroundStyle(b.earned ? MascotFrame.gold : Theme.muted)
                        .frame(width: 18)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(b.title).font(.callout.weight(b.earned ? .semibold : .regular))
                            .foregroundStyle(b.earned ? Theme.text : Theme.muted)
                        Text(b.detail).font(.caption).foregroundStyle(Theme.muted)
                    }
                }
            }
        }
        .padding(16)
        .frame(width: 300)
    }
}
