import AppKit
import SwiftUI
import TridaChatCore

struct MessageView: View {
    @EnvironmentObject var state: AppState
    let message: ChatMessage
    let isLast: Bool
    /// The mascot stands beside this reply, in place of its avatar (see MascotPanel).
    var hostsMascot = false

    var body: some View {
        switch (message.role, message.decide) {
        case (.user, _): userBubble
        case (_, .prompt?): decidePrompt
        case let (_, .card(id)?): DecisionCardView(card: state.decisionCards[id])
        default: assistant
        }
    }

    private var userBubble: some View {
        HStack {
            Spacer(minLength: 80)
            VStack(alignment: .trailing, spacing: 6) {
                if !message.images.isEmpty {
                    HStack {
                        ForEach(Array(message.images.enumerated()), id: \.offset) { _, d in
                            Thumbnail(data: d, side: 120)
                        }
                    }
                }
                if !message.content.isEmpty {
                    Text(message.content)
                        .foregroundStyle(Theme.userText)
                        .textSelection(.enabled)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 9)
                        .background(RoundedRectangle(cornerRadius: 16, style: .continuous).fill(Theme.userBubble))
                }
            }
        }
    }

    private var generating: Bool { isLast && state.isGenerating }

    /// A /decide set-up message from the app, with the step's buttons on the latest one.
    private var decidePrompt: some View {
        VStack(alignment: .leading, spacing: 8) {
            Label { MarkdownText(message.content) } icon: {
                Image(systemName: message.compaction != nil || message.content.hasPrefix("Summarizing") ? "rectangle.compress.vertical" : "chart.bar.xaxis")
            }
            if let c = message.compaction {
                DisclosureGroup("Summary the model sees") {
                    Text(c).font(.callout).foregroundStyle(.secondary).textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            if isLast, let f = state.decideFlow, f.chatID == state.selectedChatID {
                HStack(spacing: 8) {
                    switch f.step {
                    case .options:
                        Button("Yes / no question") { state.decideYesNo() }
                    case .more:
                        Button("Add Question") { state.decideAddQuestion() }
                        Button("Decide") { state.runDecideFlow() }.keyboardShortcut(.return, modifiers: .command)
                    default:
                        EmptyView()
                    }
                    Button("Cancel") { state.decideCancel() }
                }
                .controlSize(.small)
            }
        }
        .padding(10)
        .card(radius: 10, fill: Theme.surfaceRaised)
    }

    @ViewBuilder
    private var assistant: some View {
        if message.superseded == true {
            DisclosureGroup {
                MarkdownText(message.content).foregroundStyle(.secondary)
                if let ds = message.decisions, !ds.isEmpty { DecisionsView(messageID: message.id, records: ds) }
            } label: {
                Label("Draft replaced after review" + (message.check.map { " — " + $0 } ?? ""), systemImage: "arrow.uturn.backward")
                    .font(.caption).foregroundStyle(.secondary)
            }
        } else {
            HStack(alignment: .top, spacing: 12) {
                if hostsMascot { Color.clear.frame(width: 26, height: 26) } else { Avatar() }
                assistantBody
            }
            // bottom-aligned, so it stays next to the newest text while the reply streams in
            .overlay(alignment: .bottomLeading) {
                if hostsMascot {
                    MascotPanel(placement: .message)
                        .offset(x: 26 - ChatLayout.besideWidth)
                        .transition(.opacity)
                }
            }
        }
    }

    private var assistantBody: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let r = message.route {
                Label(r, systemImage: "arrow.triangle.branch")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .help("Which tools the decision model allowed for this message (Settings → Agent → Tool routing)")
            }
            if !message.reasoning.isEmpty {
                ReasoningView(text: message.reasoning, live: generating && message.content.isEmpty && message.toolRuns.isEmpty)
            }
            ForEach(message.toolRuns) { run in
                ToolRunView(run: run, messageID: message.id)
            }
            if !message.content.isEmpty {
                MarkdownText(message.content)
            } else if generating && message.reasoning.isEmpty && message.toolRuns.isEmpty {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.small)
                    Text(state.prefill.map { "Reading prompt… \($0.done.formatted())/\($0.total.formatted())" } ?? "Thinking…")
                        .foregroundStyle(.secondary)
                }
            }
            if let err = message.error {
                Label(err, systemImage: err == "Stopped." ? "stop.circle" : "exclamationmark.triangle")
                    .font(.caption)
                    .foregroundStyle(err == "Stopped." ? Theme.muted : Theme.danger)
            }
            if let c = message.check, message.stats == nil {
                CheckBadge(text: c)
            }
            // one quiet footer: how it was decoded, whether the checks passed, and the reply's actions
            if let st = message.stats {
                HStack(spacing: 10) {
                    DraftMeter(stats: st)
                    Text(String(format: "%.0f tok/s", st.decodeTokensPerSecond))
                        .monospacedDigit()
                        .help(st.summary)
                    if let c = message.check { CheckBadge(text: c) }
                    if isLast && !state.isGenerating {
                        Button { state.retry() } label: { Image(systemName: "arrow.clockwise") }
                            .buttonStyle(.borderless)
                            .help("Regenerate")
                    }
                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(message.content, forType: .string)
                    } label: { Image(systemName: "doc.on.doc") }
                        .buttonStyle(.borderless)
                        .help("Copy reply")
                    if !(isLast && state.isGenerating) {
                        Button { state.rate(message.id, "up") } label: {
                            Image(systemName: message.feedback == "up" ? "hand.thumbsup.fill" : "hand.thumbsup")
                        }
                        .buttonStyle(.borderless)
                        .help("Good reply (kept for training-data export)")
                        Button { state.rate(message.id, "down") } label: {
                            Image(systemName: message.feedback == "down" ? "hand.thumbsdown.fill" : "hand.thumbsdown")
                        }
                        .buttonStyle(.borderless)
                        .help("Bad reply (kept for training-data export)")
                        Button { state.beginEdit(message) } label: {
                            Image(systemName: message.editedFrom != nil ? "pencil.circle.fill" : "pencil")
                        }
                        .buttonStyle(.borderless)
                        .help("Edit this reply into what it should have been (the model sees your version from now on)")
                    }
                }
                .font(.caption)
                .foregroundStyle(.secondary)
            }
            if let ds = message.decisions, !ds.isEmpty, !(isLast && state.isGenerating) {
                DecisionsView(messageID: message.id, records: ds)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

struct ReasoningView: View {
    let text: String
    let live: Bool

    var body: some View {
        DisclosureGroup {
            Text(text)
                .font(.callout)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.top, 4)
        } label: {
            HStack(spacing: 6) {
                if live { ProgressView().controlSize(.mini) }
                Text(live ? "Thinking…" : "Thought for \(text.split(whereSeparator: \.isWhitespace).count) words")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
        }
    }
}

struct ToolRunView: View {
    @EnvironmentObject var state: AppState
    let run: ToolRun
    let messageID: UUID

    var body: some View {
        DisclosureGroup {
            VStack(alignment: .leading, spacing: 6) {
                Text(run.call.arguments)
                    .font(.system(size: 11, design: .monospaced))
                    .textSelection(.enabled)
                if !run.images.isEmpty {
                    HStack {
                        ForEach(Array(run.images.enumerated()), id: \.offset) { _, d in
                            Thumbnail(data: d, side: 160)
                        }
                    }
                }
                if !run.result.isEmpty {
                    Divider()
                    ScrollView {
                        Text(run.result)
                            .font(.system(size: 11, design: .monospaced))
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .frame(maxHeight: 220)
                }
            }
            .padding(.top, 4)
        } label: {
            HStack(spacing: 6) {
                statusIcon
                Text(run.call.name).font(.system(.callout, design: .monospaced)).bold()
                Text(argPreview)
                    .font(.system(.callout, design: .monospaced))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
                    .truncationMode(.tail)
                if run.call.name == "write_file", run.status == .done, !run.result.hasPrefix("error") {
                    Spacer(minLength: 8)
                    Button {
                        state.viewerTarget = ViewerTarget(messageID: messageID, runID: run.id)
                        state.showViewer = true
                    } label: {
                        Label("View", systemImage: "sidebar.right").font(.caption)
                    }
                    .buttonStyle(.borderless)
                    .help("Show this write in the code viewer")
                }
                if let note = run.approval {
                    Text(note).font(.caption2).foregroundStyle(Theme.muted).lineLimit(1)
                }
                if run.undo != nil {
                    Spacer(minLength: 8)
                    if run.undone == true {
                        Text("Undone").font(.caption).foregroundStyle(Theme.muted)
                    } else {
                        Button {
                            if let chat = state.selectedChatID {
                                state.undoWrite(chatID: chat, messageID: messageID, runID: run.id)
                            }
                        } label: {
                            Label("Undo", systemImage: "arrow.uturn.backward").font(.caption)
                        }
                        .buttonStyle(.borderless)
                        .help(run.undo?.backup == nil ? "Remove the file this call created"
                                                      : "Put the file back as it was before this write")
                    }
                }
            }
        }
        .padding(.leading, 10)
        .padding(.vertical, 2)
        .overlay(alignment: .leading) { Rectangle().fill(Theme.accent.opacity(0.45)).frame(width: 2) }
    }

    private var argPreview: String {
        let a = run.call.argumentDict
        if let c = a["command"] as? String { return c }
        if let p = a["path"] as? String { return p }
        if let q = a["query"] as? String { return q }
        if let u = a["url"] as? String { return u }
        if let h = a["heading"] as? String { return h }
        if let l = a["link"] as? String { return l }
        if let e = a["expression"] as? String { return e }
        return run.call.arguments == "{}" ? "" : run.call.arguments
    }

    @ViewBuilder private var statusIcon: some View {
        switch run.status {
        case .pending: Image(systemName: "circle.dotted").foregroundStyle(.secondary)
        case .waitingForApproval: Image(systemName: "hand.raised.fill").foregroundStyle(Theme.warning)
        case .running: ProgressView().controlSize(.mini)
        case .done: Image(systemName: "checkmark.circle.fill").foregroundStyle(Theme.success)
        case .denied: Image(systemName: "xmark.circle.fill").foregroundStyle(Theme.danger)
        }
    }
}

/// A /decide result in the chat: one probability bar per option, the reasoning collapsed.
struct DecisionCardView: View {
    let card: DecisionCard?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if let card {
                HStack(spacing: 6) {
                    if card.running { ProgressView().controlSize(.small) }
                    Label(card.phase, systemImage: "chart.bar.xaxis").foregroundStyle(.secondary)
                }
                if let r = card.result {
                    ForEach(card.questions, id: \.key) { q in
                        if let a = r.answers[q.key] {
                            AnswerView(question: q, answer: a, direct: r.escalated && r.cut == nil ? r.direct[q.key] : nil)
                        }
                    }
                    Text(String(format: "%.0f ms of decision time", r.timingMs)).font(.caption).foregroundStyle(.secondary)
                }
                if let e = card.error, e != "Stopped." { Text(e).foregroundStyle(.red).font(.caption) }
                let reasoning = card.result?.reasoning ?? card.liveReasoning
                if !reasoning.isEmpty {
                    ReasoningView(text: reasoning, live: card.running)
                }
            } else {
                Text("Decision").foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .card(radius: 10)
    }
}

/// The reply checks, quiet when they all pass: a seal (details on hover). Otherwise only the checks that did not
/// pass are spelled out, in amber (unsure, "?") or terracotta (failed, "✗").
struct CheckBadge: View {
    let text: String

    private var items: [String] {
        guard text.hasPrefix("Check: ") else { return [] }
        return text.dropFirst("Check: ".count).components(separatedBy: " · ")
    }

    var body: some View {
        let open = items.filter { !$0.hasPrefix("✓") }
        Group {
            if items.isEmpty {
                Text(text).foregroundStyle(Theme.muted)           // "Checking the reply…", errors
            } else if open.isEmpty {
                Image(systemName: "checkmark.seal").foregroundStyle(Theme.accent)
            } else {
                HStack(spacing: 6) {
                    ForEach(open, id: \.self) { item in
                        Text(item).foregroundStyle(item.hasPrefix("✗") ? Theme.danger : Theme.warning)
                    }
                }
            }
        }
        .lineLimit(1)
        .help("Reply review by the decision model: " + (items.isEmpty ? text : items.joined(separator: ", ")))
    }
}

/// Self-speculative decoding at a glance: one cell per token the model may commit per step (the generation
/// block N from the mode, e.g. "self-spec(N=4)"), filled up to the reply's average tokens per forward. A full
/// bar means the diffusion drafts were almost always accepted; one cell is plain AR speed.
struct DraftMeter: View {
    let stats: GenStats

    private var block: Int? {
        guard stats.mode.hasPrefix("self-spec"), let r = stats.mode.range(of: "N="),
              let n = Int(stats.mode[r.upperBound...].prefix { $0.isNumber }), n > 1 else { return nil }
        return min(n, 16)
    }

    var body: some View {
        if let n = block, stats.tokensPerForward > 0 {
            HStack(spacing: 2) {
                ForEach(0..<n, id: \.self) { i in
                    let fill = min(max(stats.tokensPerForward - Double(i), 0), 1)
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 1.5).fill(Theme.accentSoft)
                        RoundedRectangle(cornerRadius: 1.5).fill(Theme.accent).frame(width: 6 * fill)
                    }
                    .frame(width: 6, height: 10)
                }
            }
            .help(String(format: "%.2f of %d tokens per step (self-speculative decoding)", stats.tokensPerForward, n))
            .accessibilityLabel(String(format: "%.1f of %d tokens per step", stats.tokensPerForward, n))
        }
    }
}

/// The model's mark beside its replies.
struct Avatar: View {
    var body: some View {
        Text("Z")
            .font(.system(size: 13, weight: .bold, design: .rounded))
            .foregroundStyle(Theme.onAccent)
            .frame(width: 26, height: 26)
            .background(Circle().fill(Theme.accent))
            .padding(.top, 1)
            .help("Zoro-1")
    }
}
