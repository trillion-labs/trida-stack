import AppKit
import SwiftUI
import TridaChatCore

/// What the model is doing, as the mascot shows it.
enum MascotState: Equatable {
    case offline, idle, reading, thinking, deciding, tool, answering, checking, done, error

    var caption: String {
        switch self {
        case .offline: return "Waiting for the server"
        case .idle: return "Ready"
        case .reading: return "Reading your message"
        case .thinking: return "Thinking"
        case .deciding: return "Deciding"
        case .tool: return "Using a tool"
        case .answering: return "Answering"
        case .checking: return "Checking the answer"
        case .done: return "Done"
        case .error: return "Something went wrong"
        }
    }
}

extension AppState {
    /// Derived from what the app already tracks: server status, the prompt prefill, the latest reply's
    /// routing note, tool runs, content and review line, and running /decide cards.
    var mascotState: MascotState {
        switch status {
        case .failed: return .error
        case .starting, .stopped: return .offline
        default: break
        }
        let last = currentChatIndex.flatMap { chats[$0].messages.last { $0.role == .assistant } }
        if decisionCards.values.contains(where: { $0.running }) { return .deciding }
        if let c = last?.check, c.hasPrefix("Checking the reply") { return .checking }
        if isGenerating {
            if prefill != nil { return .reading }
            if let m = last {
                if let r = m.route, r.hasPrefix("Choosing tools") { return .deciding }
                if let run = m.toolRuns.last(where: { $0.status == .running || $0.status == .waitingForApproval }) {
                    return DecisionTools.names.contains(run.call.name) ? .deciding : .tool
                }
                if !m.content.isEmpty { return .answering }
            }
            return .thinking
        }
        if let e = last?.error, e != "Stopped." { return .error }
        if let t = lastFinished, Date().timeIntervalSince(t) < 2.5 { return .done }
        return .idle
    }
}

/// The Zoro-1 mascot: a 32 x 31 pixel figure (head, bent arms, crossed legs, a Z on its belly), drawn on a
/// Canvas and animated at 6 frames per second by moving its parts: eyes, mouth, hands, the whole body, and a
/// few effect pixels around it. With Reduce Motion on, it holds one still pose per state.
struct MascotView: View {
    let state: MascotState
    var fps = 6.0

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

    var body: some View {
        TimelineView(.periodic(from: .now, by: 1 / fps)) { tl in
            let still = NSWorkspace.shared.accessibilityDisplayShouldReduceMotion
            let step = still ? 0 : Int(tl.date.timeIntervalSinceReferenceDate * fps)
            Canvas { gc, size in
                draw(Pose(state: state, step: step), in: gc, size: size)
            }
        }
        .accessibilityElement()
        .accessibilityLabel("Zoro-1: \(state.caption)")
    }

    // A frame: how far each part is moved and which face and effects are on.
    struct Pose {
        enum Eyes { case open, closed, slanted, focused }
        enum Mouth { case smile, open, flat, frown, none }
        var bob = 0, head = 0, leftHand = 0, rightHand = 0
        var eyes = Eyes.open, eyeDX = 0, eyeDY = 0
        var mouth = Mouth.smile
        var zGlow = 0.85
        var effects: [(row: Int, col: Int, color: Color)] = []

        init(state: MascotState, step s: Int) {
            let blink = s % 30 == 0
            switch state {
            case .offline:
                eyes = .closed; mouth = .flat; zGlow = 0.35
                let k = s / 3 % 3                                  // a "z" drifting up from the head
                effects = Pose.letterZ(row: -2 - 2 * k, col: 25 + k, color: Theme.muted.opacity(1 - Double(k) * 0.3))
            case .idle:
                bob = s / 4 % 2; eyes = blink ? .closed : .open
            case .reading:
                // holds a page in front of its belly; eyes down and scanning, the lines scroll up
                eyeDX = s / 3 % 2 == 0 ? -1 : 1; eyeDY = 1; mouth = .none     // the lowered eyes need the room
                leftHand = -1; rightHand = -1
                for r in 13...18 { for c in 10...21 { effects.append((r, c, MascotView.face)) } }
                for i in 0..<3 {
                    let r = 14 + (i * 2 + 5 - s / 2 % 5) % 5
                    for c in 11...(i == 1 ? 18 : 20) { effects.append((r, c, Theme.muted)) }
                }
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
            case .tool:
                // hammers with a wrench (right hand) while a gear turns by its left shoulder
                let up = s % 2 == 0
                rightHand = up ? -3 : 0; eyes = .focused; mouth = .flat
                let wrench = [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1), (4, 1)]   // jaws on a shaft
                effects = wrench.map { (row: (up ? 6 : 9) + $0.0, col: 27 + $0.1, color: MascotView.highlight) }
                if !up { effects += [(14, 31, Theme.warning), (13, 31, Theme.warning), (12, 30, Theme.warning)] }
                effects += Pose.gear(row: 6, col: 1, turned: s / 2 % 2 == 1, color: Theme.muted)
            case .answering:
                mouth = s % 2 == 0 ? .open : .smile; zGlow = s % 2 == 0 ? 1 : 0.6
                bob = s / 3 % 2
            case .checking:
                rightHand = -2; eyeDX = 1; mouth = .flat
                let on = s / 3 % 2 == 0
                effects = [(7, 26), (8, 27), (7, 28), (6, 29), (5, 30)].map {
                    (row: $0.0, col: $0.1, color: on ? Theme.accent : Theme.accent.opacity(0.4))
                }
            case .done:
                bob = s % 4 < 2 ? -1 : 0; leftHand = -2; rightHand = -2
                if s % 2 == 0 { effects = [(-1, 3, Theme.accent), (-2, 28, Theme.accent), (2, 0, Theme.warning)] }
            case .error:
                eyes = .slanted; mouth = .frown; head = 1; zGlow = 0.4
                effects = [(3, 29, Theme.danger), (4, 29, Theme.danger), (6, 29, Theme.danger)]   // "!"
            }
        }

        static func square(row: Int, col: Int, color: Color) -> [(row: Int, col: Int, color: Color)] {
            [(row, col, color), (row, col + 1, color), (row + 1, col, color), (row + 1, col + 1, color)]
        }

        /// A thought bubble at the upper right (rows -9 to -3, columns 22 to 32): a cream cloud with an outline,
        /// a two-dot trail from the head, and up to three 2 x 2 dots filling in inside it.
        static func thoughtBubble(dots: Int, float: Int) -> [(row: Int, col: Int, color: Color)] {
            var out: [(row: Int, col: Int, color: Color)] = []
            let top = -9 + float                                  // bobs down a row: row -9 is the top of the canvas
            let outline = Theme.muted
            for c in 24...30 { out.append((top, c, outline)); out.append((top + 6, c, outline)) }
            for (dr, l, r) in [(1, 23, 31), (2, 22, 32), (3, 22, 32), (4, 22, 32), (5, 23, 31)] {
                out.append((top + dr, l, outline)); out.append((top + dr, r, outline))
                for c in (l + 1)..<r { out.append((top + dr, c, MascotView.face)) }
            }
            out.append((top + 7, 22, outline))                      // the trail toward the head
            out.append((top + 8, 21, outline))
            for i in 0..<dots {
                let c = 24 + 3 * i
                out += [(top + 2, c, Theme.accent), (top + 2, c + 1, Theme.accent),
                        (top + 3, c, Theme.accent), (top + 3, c + 1, Theme.accent)]
            }
            return out
        }

        /// A "?" above the head, rows -6 to -1.
        static func question(color: Color) -> [(row: Int, col: Int, color: Color)] {
            [(-6, 15), (-6, 16), (-5, 14), (-5, 17), (-4, 17), (-3, 16), (-1, 16)].map { (row: $0.0, col: $0.1, color: color) }
        }

        /// A 5 x 5 gear in two rotations, its centre left open.
        static func gear(row: Int, col: Int, turned: Bool, color: Color) -> [(row: Int, col: Int, color: Color)] {
            let a = [(0, 2), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (4, 2)]
            let b = [(0, 0), (0, 4), (1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3), (4, 0), (4, 4)]
            return (turned ? b : a).map { (row: row + $0.0, col: col + $0.1, color: color) }
        }

        static func letterZ(row: Int, col: Int, color: Color) -> [(row: Int, col: Int, color: Color)] {
            [(row, col), (row, col + 1), (row, col + 2), (row + 1, col + 1), (row + 2, col), (row + 2, col + 1),
             (row + 2, col + 2)].map { (row: $0.0, col: $0.1, color: color) }
        }
    }

    /// Room around the sprite for effects: 9 rows above (the thought bubble), 2 columns each side, 1 row below.
    static let margin = (top: 9, side: 2, bottom: 1)

    func draw(_ p: Pose, in gc: GraphicsContext, size: CGSize) {
        let rows = Self.sprite.count + Self.margin.top + Self.margin.bottom
        let cols = Self.sprite[0].count + 2 * Self.margin.side
        let cell = floor(min(size.width / CGFloat(cols), size.height / CGFloat(rows)))
        let ox = (size.width - cell * CGFloat(cols)) / 2 + cell * CGFloat(Self.margin.side)
        let oy = (size.height - cell * CGFloat(rows)) / 2 + cell * CGFloat(Self.margin.top)
        func fill(_ r: Int, _ c: Int, _ color: Color) {
            gc.fill(Path(CGRect(x: ox + CGFloat(c) * cell, y: oy + CGFloat(r) * cell, width: cell, height: cell)),
                    with: .color(color))
        }
        for (r, line) in Self.sprite.enumerated() {
            for (c, ch) in line.enumerated() where ch != "." {
                var dy = p.bob
                if r <= 11 { dy += p.head }
                if (13...19).contains(r), c <= 7 { dy += p.leftHand }
                if (13...19).contains(r), c >= 24 { dy += p.rightHand }
                switch ch {
                case "g": fill(r + dy, c, Self.highlight)
                case "Z": fill(r + dy, c, Theme.accent.opacity(p.zGlow))
                default: fill(r + dy, c, Self.body)            // eyes and mouth are drawn over the head below
                }
            }
        }
        let hy = p.bob + p.head
        for (r, c) in eyeCells(p) { fill(r + hy, c, Self.face) }
        for (r, c) in mouthCells(p.mouth) { fill(r + hy, c, Self.face) }
        for e in p.effects { fill(e.row + p.bob, e.col, e.color) }
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
        case .focused:                                          // narrowed, looking at the work
            return [(9, 12), (9, 13), (9, 20), (9, 21)]           // one row, glancing right
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
}

/// The mascot's corner of the sidebar: the figure and what it is doing, in plain words.
struct MascotPanel: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        TimelineView(.periodic(from: .now, by: 0.5)) { _ in      // re-derive the state (e.g. "done" expiring)
            let s = state.mascotState
            VStack(spacing: 6) {
                MascotView(state: s)
                    .frame(width: 124, height: 124)       // 36 x 41 cells at 3 pt each (pixels stay whole points)
                Text(s.caption)
                    .font(.callout.weight(.medium))
                    .foregroundStyle(s == .error ? Theme.danger : Theme.text)
                    .animation(nil, value: s)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
            .background(Theme.sidebar)
            .overlay(alignment: .top) { Rectangle().fill(Theme.border).frame(height: 1) }
        }
    }
}
