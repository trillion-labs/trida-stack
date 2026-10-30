import AppKit
import SwiftUI
import TridaChatCore
import UniformTypeIdentifiers

/// One question as edited in the Decide panel.
struct EditableQuestion: Identifiable {
    let id = UUID()
    var key: String
    var type: DecisionType
    var instructions: String
    /// choice: "key: description" per line; score: one level per line; yes/no: unused.
    var optionsText: String = ""

    var question: DecisionQuestion {
        DecisionQuestion(key: key.trimmingCharacters(in: .whitespaces), type: type, instructions: instructions,
                         options: DecisionQuestion.parseOptions(optionsText, type: type))
    }
}

struct DecidePreset {
    var name: String
    var state: String
    var questions: [EditableQuestion]
}

/// State of the Decide window: a situation, typed questions about it, and the decision model's answers.
/// (Kept in an ObservableObject rather than @State; see AppState.)
@MainActor
final class DecidePanel: ObservableObject {
    @Published var state = ""
    @Published var questions: [EditableQuestion] = []
    @Published var image: Data?
    @Published var reasoning = "auto"
    @Published var threshold = 0.7
    @Published var maxReasoningTokens = 8192

    @Published var running = false
    @Published var phase = ""
    @Published var liveReasoning = ""
    @Published var result: DecisionResult?
    @Published var resultQuestions: [DecisionQuestion] = []
    @Published var error: String?
    private var task: Task<Void, Never>?

    init() {}   // starts empty; Examples loads a sample

    func load(_ p: DecidePreset) {
        state = p.state
        questions = p.questions
        image = nil
        clearResult()
    }

    func clearResult() {
        result = nil
        liveReasoning = ""
        phase = ""
        error = nil
    }

    func addQuestion() {
        questions.append(EditableQuestion(key: "q\(questions.count + 1)", type: .choice, instructions: ""))
    }

    func remove(_ id: UUID) { questions.removeAll { $0.id == id } }

    /// Problems that would make the server reject the request, in plain words.
    var problem: String? {
        if state.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && image == nil { return "Describe the situation (or add an image)." }
        if questions.isEmpty { return "Add a question." }
        var keys = Set<String>()
        for q in questions.map(\.question) {
            if q.key.isEmpty { return "Every question needs a key." }
            if !keys.insert(q.key).inserted { return "Two questions share the key \"\(q.key)\"." }
            if q.type == .choice && q.options.count < 2 { return "\(q.key): a choice needs at least two options." }
            if q.type == .score && q.options.count < 2 { return "\(q.key): a score needs at least two levels." }
            if q.type == .choice && Set(q.options.map(\.0)).count != q.options.count { return "\(q.key): option keys repeat." }
        }
        return nil
    }

    func run(client: ChatClient, selfSpec: Bool) {
        guard !running, problem == nil else { return }
        let qs = questions.map(\.question)
        var settings = DecisionSettings(reasoning: reasoning, threshold: threshold,
                                        maxReasoningTokens: maxReasoningTokens, selfSpec: selfSpec)
        settings.maxImagePixels = 448 * 448
        let body = Decision.body(state: .string(state), questions: qs, image: image, settings: settings, stream: true)
        clearResult()
        resultQuestions = qs
        running = true
        phase = "Deciding…"
        task = Task {
            defer { running = false }
            do {
                for try await ev in client.decideStream(body, questions: qs) {
                    switch ev {
                    case let .direct(reason):
                        phase = reason.map { "Unsure: \($0). Reasoning…" } ?? "Decided directly."
                    case let .reasoning(t):
                        liveReasoning += t
                    case let .result(r):
                        result = r
                        phase = r.escalated ? (r.cut == nil ? "Decided after reasoning." : "Reasoning cut off; direct decision kept.")
                                            : "Decided directly (no reasoning needed)."
                    }
                }
            } catch {
                if !Task.isCancelled { self.error = error.localizedDescription }
                phase = Task.isCancelled ? "Stopped." : "Failed."
            }
        }
    }

    func cancel() {
        task?.cancel()
        task = nil
    }

    func chooseImage() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.image]
        if panel.runModal() == .OK, let url = panel.url { image = ImageUtil.pngData(contentsOf: url) }
    }

    static let presets: [DecidePreset] = [
        DecidePreset(name: "Support triage", state: """
            Customer message: "I was charged twice for my October invoice and nobody answered the email I sent \
            last week. Please fix this today, I'm about to cancel."
            Customer plan: Business, 3 years.
            """, questions: [
                EditableQuestion(key: "category", type: .choice, instructions: "What is the message about?",
                                 optionsText: "billing: billing or payments\ntechnical: something is broken\n"
                                     + "account: login or account settings\nother: anything else"),
                EditableQuestion(key: "urgency", type: .score, instructions: "How urgent is it?",
                                 optionsText: "low\nmedium\nhigh"),
                EditableQuestion(key: "human", type: .noul, instructions: "Should a human agent take this over?"),
            ]),
        DecidePreset(name: "Phishing check", state: """
            From: "IT Service Desk" <helpdesk@rnicrosoft-support.co>
            Subject: Your mailbox will be deleted in 24 hours
            Body: We detected unusual activity. Verify your password at http://rnicrosoft-support.co/verify \
            within 24 hours or your mailbox and all files will be permanently removed.
            """, questions: [
                EditableQuestion(key: "phishing", type: .noul, instructions: "Is this email phishing?"),
                EditableQuestion(key: "action", type: .choice, instructions: "What should the recipient do?",
                                 optionsText: "report: report it to the security team\nreply: reply and ask for details\n"
                                     + "click: follow the link and verify\nignore: ignore it"),
            ]),
        DecidePreset(name: "Review rating", state: """
            Review: "Battery easily lasts two days and the screen is great outside. The camera is only OK at night \
            and the charger is not in the box, which is annoying at this price."
            """, questions: [
                EditableQuestion(key: "stars", type: .score, instructions: "How many stars did the reviewer give?",
                                 optionsText: "1 star\n2 stars\n3 stars\n4 stars\n5 stars"),
                EditableQuestion(key: "topic", type: .choice, instructions: "What does the review talk about most?",
                                 optionsText: "battery\nscreen\ncamera\nprice\naccessories"),
                EditableQuestion(key: "recommend", type: .noul, instructions: "Would the reviewer recommend it?"),
            ]),
        DecidePreset(name: "Cheapest plan (needs reasoning)", state: """
            A customer uses 12 GB of data and 300 minutes of calls a month.
            Plan A: $20 a month, 10 GB included, $5 per extra GB, unlimited calls.
            Plan B: $28 a month, unlimited data, unlimited calls.
            Plan C: $15 a month, 5 GB included, $4 per extra GB, 200 minutes included, $0.05 per extra minute.
            """, questions: [
                EditableQuestion(key: "plan", type: .choice, instructions: "Which plan is cheapest for this customer?",
                                 optionsText: "A: Plan A\nB: Plan B\nC: Plan C"),
            ]),
    ]
}

struct DecideView: View {
    @EnvironmentObject var app: AppState
    @ObservedObject var panel: DecidePanel

    var body: some View {
        HSplitView {
            editor.frame(minWidth: 380, idealWidth: 460)
            results.frame(minWidth: 340, idealWidth: 420)
        }
        .frame(minWidth: 760, minHeight: 520)
    }

    // MARK: editor

    private var editor: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Menu("Examples") {
                    ForEach(DecidePanel.presets.indices, id: \.self) { i in
                        Button(DecidePanel.presets[i].name) { panel.load(DecidePanel.presets[i]) }
                    }
                }
                .frame(width: 130)
                Spacer()
                if !app.supportsDecision {
                    Label(app.status.isUsable ? "This model has no decision head" : "Server not ready",
                          systemImage: "exclamationmark.triangle").foregroundStyle(.orange).font(.caption)
                }
            }
            Text("Situation (state)").font(.headline)
            TextEditor(text: $panel.state)
                .font(.body)
                .frame(minHeight: 110)
                .overlay(RoundedRectangle(cornerRadius: 4).stroke(Color.secondary.opacity(0.3)))
            HStack {
                if let img = panel.image, let ns = NSImage(data: img) {
                    Image(nsImage: ns).resizable().scaledToFit().frame(height: 48)
                    Button("Remove Image") { panel.image = nil }
                } else {
                    Button("Add Image…") { panel.chooseImage() }
                        .disabled(!app.supportsVision)
                }
                Spacer()
            }
            HStack {
                Text("Questions").font(.headline)
                Spacer()
                Button { panel.addQuestion() } label: { Label("Add", systemImage: "plus") }
            }
            ScrollView {
                VStack(spacing: 10) {
                    ForEach($panel.questions) { $q in
                        QuestionEditor(q: $q) { panel.remove(q.id) }
                    }
                }
            }
            Divider()
            HStack {
                Picker("Reasoning", selection: $panel.reasoning) {
                    Text("When unsure").tag("auto")
                    Text("Always").tag("always")
                    Text("Never").tag("never")
                }
                .frame(width: 220)
                if panel.reasoning == "auto" {
                    Text("below")
                    TextField("", value: $panel.threshold, format: .number.precision(.fractionLength(2)))
                        .frame(width: 44)
                }
                Spacer()
            }
            HStack {
                Stepper("Reasoning budget: \(panel.maxReasoningTokens) tokens", value: $panel.maxReasoningTokens,
                        in: 128...8192, step: 128)
                Spacer()
                Text(app.settings.selfSpec ? "self-spec" : "AR").foregroundStyle(.secondary).font(.caption)
            }
            HStack {
                if let p = panel.problem { Text(p).font(.caption).foregroundStyle(.orange) }
                Spacer()
                if panel.running {
                    Button("Stop") { panel.cancel() }.keyboardShortcut(".")
                } else {
                    Button("Decide") { panel.run(client: app.client, selfSpec: app.settings.selfSpec) }
                        .keyboardShortcut(.return, modifiers: .command)
                        .disabled(panel.problem != nil || !app.supportsDecision)
                }
            }
        }
        .padding(14)
    }

    // MARK: results

    private var results: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                HStack {
                    if panel.running { ProgressView().controlSize(.small) }
                    Text(panel.phase.isEmpty ? "Answers appear here." : panel.phase).foregroundStyle(.secondary)
                }
                if let e = panel.error { Text(e).foregroundStyle(.red).textSelection(.enabled) }
                if let r = panel.result {
                    ForEach(panel.resultQuestions, id: \.key) { q in
                        if let a = r.answers[q.key] {
                            AnswerView(question: q, answer: a, direct: r.escalated && r.cut == nil ? r.direct[q.key] : nil)
                        }
                    }
                    Text(String(format: "%.0f ms of decision time", r.timingMs)).font(.caption).foregroundStyle(.secondary)
                    if let c = r.cut { Text(c).font(.caption).foregroundStyle(.orange) }
                }
                if !panel.liveReasoning.isEmpty || panel.result?.reasoning != nil {
                    DisclosureGroup("Reasoning") {
                        Text(panel.result?.reasoning ?? panel.liveReasoning)
                            .font(.callout)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                }
            }
            .padding(14)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}

private struct QuestionEditor: View {
    @Binding var q: EditableQuestion
    var onDelete: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                TextField("key", text: $q.key).frame(width: 110).textFieldStyle(.roundedBorder)
                Picker("", selection: $q.type) {
                    ForEach(DecisionType.allCases, id: \.self) { Text($0.label).tag($0) }
                }
                .frame(width: 110)
                Spacer()
                Button(role: .destructive, action: onDelete) { Image(systemName: "trash") }.buttonStyle(.borderless)
            }
            TextField("Question, e.g. Which team handles it?", text: $q.instructions).textFieldStyle(.roundedBorder)
            if q.type != .noul {
                Text(q.type == .choice ? "Options, one per line: key: description" : "Levels, lowest first, one per line")
                    .font(.caption).foregroundStyle(.secondary)
                TextEditor(text: $q.optionsText)
                    .font(.system(.callout, design: .monospaced))
                    .frame(height: 70)
                    .overlay(RoundedRectangle(cornerRadius: 4).stroke(Color.secondary.opacity(0.3)))
            }
        }
        .padding(8)
        .background(RoundedRectangle(cornerRadius: 6).fill(Color.secondary.opacity(0.07)))
    }
}

struct AnswerView: View {
    let question: DecisionQuestion
    let answer: DecisionAnswer
    /// The direct (pre-reasoning) answer, when the shown one came after reasoning.
    let direct: DecisionAnswer?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(alignment: .firstTextBaseline) {
                Text(question.key).font(.headline)
                Text(question.instructions).font(.caption).foregroundStyle(.secondary).lineLimit(2)
            }
            Text(answer.summary).font(.body.weight(.semibold))
            ForEach(answer.options.indices, id: \.self) { i in
                let o = answer.options[i]
                let picked = o.label == answer.pick || o.key == answer.pick
                HStack(spacing: 8) {
                    Text(o.label).lineLimit(1).frame(width: 150, alignment: .leading)
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 3).fill(Color.secondary.opacity(0.15)).frame(width: 160, height: 10)
                        RoundedRectangle(cornerRadius: 3).fill(picked ? Color.accentColor : Color.secondary.opacity(0.6))
                            .frame(width: max(2, 160 * o.p), height: 10)
                    }
                    Text(String(format: "%.2f", o.p)).monospacedDigit().font(.caption)
                    if let d = direct, i < d.options.count {
                        Text(String(format: "(direct %.2f)", d.options[i].p)).font(.caption2).foregroundStyle(.secondary)
                    }
                }
            }
        }
    }
}

/// Opens the Decide window (menu item and toolbar button).
struct OpenDecideButton: View {
    @Environment(\.openWindow) private var openWindow
    var label = "Decide…"

    var body: some View {
        Button { openWindow(id: "decide") } label: { Label(label, systemImage: "chart.bar.xaxis") }
            .help("Typed decisions with calibrated probabilities (Zoro-1 decision head)")
    }
}
