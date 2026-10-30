import AppKit
import SwiftUI
import TridaChatCore

struct MessageView: View {
    @EnvironmentObject var state: AppState
    let message: ChatMessage
    let isLast: Bool

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
                        .textSelection(.enabled)
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(RoundedRectangle(cornerRadius: 12).fill(Color.accentColor.opacity(0.15)))
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
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.secondary.opacity(0.08)))
    }

    @ViewBuilder
    private var assistant: some View {
        if message.superseded == true {
            DisclosureGroup {
                MarkdownText(message.content).foregroundStyle(.secondary)
            } label: {
                Label("Draft replaced after review" + (message.check.map { " — " + $0 } ?? ""), systemImage: "arrow.uturn.backward")
                    .font(.caption).foregroundStyle(.secondary)
            }
        } else {
            assistantBody
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
                ToolRunView(run: run)
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
                    .foregroundStyle(err == "Stopped." ? Color.secondary : Color.red)
            }
            if let c = message.check {
                Label(c, systemImage: c.contains("✗") ? "exclamationmark.triangle" : "checkmark.seal")
                    .font(.caption)
                    .foregroundStyle(c.contains("✗") ? Color.orange : Color.secondary)
                    .help("Reply review by the decision model (Settings → Agent → Check replies)")
            }
            if let st = message.stats {
                HStack(spacing: 10) {
                    Text(st.summary)
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
    let run: ToolRun

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
            }
        }
        .padding(8)
        .background(RoundedRectangle(cornerRadius: 8).fill(Color.secondary.opacity(0.08)))
    }

    private var argPreview: String {
        let a = run.call.argumentDict
        if let c = a["command"] as? String { return c }
        if let p = a["path"] as? String { return p }
        if let q = a["query"] as? String { return q }
        if let e = a["expression"] as? String { return e }
        return run.call.arguments == "{}" ? "" : run.call.arguments
    }

    @ViewBuilder private var statusIcon: some View {
        switch run.status {
        case .pending: Image(systemName: "circle.dotted").foregroundStyle(.secondary)
        case .waitingForApproval: Image(systemName: "hand.raised.fill").foregroundStyle(.orange)
        case .running: ProgressView().controlSize(.mini)
        case .done: Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
        case .denied: Image(systemName: "xmark.circle.fill").foregroundStyle(.red)
        }
    }
}

/// Inline Markdown (bold, italics, code, links) per paragraph; fenced code blocks in monospace.
struct MarkdownText: View {
    let blocks: [Block]

    enum Block: Hashable {
        case text(String)
        case code(String)
    }

    init(_ source: String) {
        var out: [Block] = []
        var buf: [Substring] = []
        var inCode = false
        func flush() {
            let s = buf.joined(separator: "\n")
            if inCode {
                out.append(.code(s))
            } else if !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                out.append(.text(s))
            }
            buf = []
        }
        for line in source.split(separator: "\n", omittingEmptySubsequences: false) {
            if line.trimmingCharacters(in: .whitespaces).hasPrefix("```") {
                flush()
                inCode.toggle()
            } else {
                buf.append(line)
            }
        }
        flush()
        blocks = out
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, b in
                switch b {
                case let .text(s):
                    Text(Self.attributed(s))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                case let .code(s):
                    ScrollView(.horizontal) {
                        Text(s)
                            .font(.system(size: 12, design: .monospaced))
                            .textSelection(.enabled)
                            .padding(10)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(RoundedRectangle(cornerRadius: 8).fill(Color.secondary.opacity(0.1)))
                }
            }
        }
    }

    static func attributed(_ s: String) -> AttributedString {
        let opts = AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        return (try? AttributedString(markdown: s, options: opts)) ?? AttributedString(s)
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
        .padding(10)
        .background(RoundedRectangle(cornerRadius: 8).stroke(Color.secondary.opacity(0.25)))
    }
}
