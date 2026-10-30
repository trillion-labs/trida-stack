import AppKit
import Foundation
import TridaChatCore

struct Chat: Identifiable, Equatable, Codable {
    var id = UUID()
    var title: String = "New chat"
    var messages: [ChatMessage] = []
    /// When it was last saved with changes (sorts the sidebar after a restart).
    var updated = Date()
}

/// A pending "allow this tool call?" question shown as a sheet.
struct ApprovalRequest: Identifiable {
    let id = UUID()
    var toolName: String
    var detail: String
    /// UI actions: offer "Allow all UI actions for this message".
    var offerAllowAll = false
    /// Shown under the detail (default: the workspace).
    var caption: String?
}

enum ServerStatus: Equatable {
    case stopped
    case starting(String)
    case ready(String)            // served model name
    case external(String)         // a server we did not launch already answers on the port
    case failed(String)

    var isUsable: Bool {
        switch self {
        case .ready, .external: return true
        default: return false
        }
    }

    var label: String {
        switch self {
        case .stopped: return "Server stopped"
        case let .starting(s): return s
        case let .ready(m): return "Ready · \(m)"
        case let .external(m): return "Connected to running server · \(m)"
        case let .failed(e): return e
        }
    }
}

@MainActor
final class AppState: ObservableObject {
    @Published var settings = AppSettings.load() {
        didSet { if settings != oldValue { settings.save() } }
    }
    @Published var status: ServerStatus = .stopped
    @Published var serverLog = ""
    @Published var models: [LocalModel] = []
    @Published var supportsVision = false
    /// The served checkpoint has a pointer head for typed decisions (POST /v1/decide).
    @Published var supportsDecision = false

    @Published var chats: [Chat] = [Chat()] {
        didSet { chatStore.scheduleSave(chats, cards: decisionCards) }
    }
    /// Chats are saved to ~/Library/Application Support/Zoro-1 Chat/chats and reloaded at launch.
    let chatStore: ChatStore
    @Published var selectedChatID: Chat.ID?
    @Published var isGenerating = false
    @Published var prefill: (done: Int, total: Int)? = nil
    @Published var approval: ApprovalRequest?

    // View state kept here rather than in @State: with only the Command Line Tools installed,
    // the SwiftUI macro plugin that implements @State is missing.
    @Published var draft = ""
    @Published var draftImages: [Data] = []
    @Published var importingImage = false
    @Published var dropTargeted = false
    @Published var showLog = false

    // "Get a Model" sheet (download from Hugging Face + convert to MLX).
    @Published var showModelSetup = false
    @Published var setupRepo = ModelRecipe.known[0].repo
    @Published var setupBits = 8
    @Published var setupToken = ""
    @Published var setupDeleteRaw = false
    @Published var setupRunning = false
    @Published var setupPhase = ""
    @Published var setupLog = TerminalLog()
    @Published var setupError: String?
    private let setupRunner = CommandRunner()
    private var setupTask: Task<Void, Never>?

    let server = ServerProcess()
    var generationTask: Task<Void, Never>?
    private var approvalContinuation: CheckedContinuation<Bool, Never>?
    private var startupTask: Task<Void, Never>?
    private var launchedByUs = false

    /// Request + raw model output logs (server --log-requests), one JSON file per request.
    static let requestLogDir = FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Logs/Zoro-1 Chat/requests")

    /// Agent tool: screenshot (vision models only; asks before every capture).
    @Published var allowScreenshot = UserDefaults.standard.object(forKey: "TridaChat.allowScreenshot") as? Bool ?? true {
        didSet { UserDefaults.standard.set(allowScreenshot, forKey: "TridaChat.allowScreenshot") }
    }

    /// UI agent (beta): operate another app through the Accessibility API; clicks are typed decisions.
    @Published var uiAgentEnabled = UserDefaults.standard.bool(forKey: "TridaChat.uiAgent") {
        didSet { UserDefaults.standard.set(uiAgentEnabled, forKey: "TridaChat.uiAgent") }
    }
    /// Send the target window's screenshot with each click decision (the element list is always sent).
    @Published var uiDecisionScreenshot = UserDefaults.standard.bool(forKey: "TridaChat.uiScreenshot") {
        didSet { UserDefaults.standard.set(uiDecisionScreenshot, forKey: "TridaChat.uiScreenshot") }
    }
    /// The app the UI agent operates: the last app other than this one that was frontmost.
    @Published var uiTarget = "no app"
    @Published var headDownloading = false
    @Published var headMessage: String?
    var uiSession = UISession()
    /// The /decide flow in progress, and the decision cards shown in chats.
    @Published var decideFlow: DecideFlow?
    /// Highlighted row of the composer's slash-command menu.
    @Published var slashSelection = 0
    /// Local control API for simulated users (Automation.swift); off unless enabled in Settings.
    @Published var automationEnabled = UserDefaults.standard.bool(forKey: "TridaChat.automation") {
        didSet {
            UserDefaults.standard.set(automationEnabled, forKey: "TridaChat.automation")
            updateAutomation()
        }
    }
    @Published var automationError: String?
    private var automation: AutomationServer?

    func updateAutomation() {
        if automationEnabled {
            if automation == nil { automation = AutomationServer(state: self) }
            do {
                try automation?.start()
                automationError = nil
            } catch {
                automationError = error.localizedDescription
            }
        } else {
            automation?.stop()
            automation = nil
        }
    }

    /// The reply being edited, and the text in the edit sheet.
    @Published var editTarget: EditTarget?
    @Published var editText = ""
    @Published var decisionCards: [UUID: DecisionCard] = [:] {
        didSet { chatStore.scheduleSave(chats, cards: decisionCards) }
    }
    var decideTask: Task<Void, Never>?
    /// The Decide window's state.
    let decidePanel: DecidePanel
    private var headLog = ""
    private var stopMonitors: [Any] = []

    /// Tool routing by the decision model (agent mode, with a decision head).
    @Published var toolRouting = ToolRouting(rawValue: UserDefaults.standard.string(forKey: "TridaChat.toolRouting") ?? "") ?? .reason {
        didSet { UserDefaults.standard.set(toolRouting.rawValue, forKey: "TridaChat.toolRouting") }
    }

    /// run_shell: run read-only commands (ls, cat, git status, …) without asking.
    @Published var autoApproveReadOnlyShell = UserDefaults.standard.object(forKey: "TridaChat.autoReadOnlyShell") as? Bool ?? true {
        didSet { UserDefaults.standard.set(autoApproveReadOnlyShell, forKey: "TridaChat.autoReadOnlyShell") }
    }

    /// Check each finished reply with the decision model (language, does what was asked, complete, not a
    /// repeat) and rewrite it once when a check fails.
    @Published var reviewReplies = UserDefaults.standard.object(forKey: "TridaChat.reviewReplies") as? Bool ?? true {
        didSet { UserDefaults.standard.set(reviewReplies, forKey: "TridaChat.reviewReplies") }
    }

    @Published var logRequests = UserDefaults.standard.object(forKey: "TridaChat.logRequests") as? Bool ?? true {
        didSet { UserDefaults.standard.set(logRequests, forKey: "TridaChat.logRequests") }
    }

    init() {
        decidePanel = DecidePanel()
        chatStore = ChatStore()
        let saved = chatStore.load()
        chats = [Chat()] + saved.chats   // a fresh chat on top; earlier ones in the sidebar
        decisionCards = saved.cards
        selectedChatID = chats.first?.id
        // One-time move to the tool-use sampling defaults (temperature 0.6, top-k 20) for
        // settings saved by earlier builds with temperature 1.0.
        // One-time move to a 64K reply budget (thinking + answer) for settings saved with the old 4,096.
        // One-time: run_shell is on by default (read-only commands run without asking, the rest ask).
        if !UserDefaults.standard.bool(forKey: "TridaChat.shellDefaultOn") {
            settings.allowShell = true
            settings.save()
            UserDefaults.standard.set(true, forKey: "TridaChat.shellDefaultOn")
        }
        // One-time move to a 128K context for settings saved with an earlier default.
        if !UserDefaults.standard.bool(forKey: "TridaChat.context128k") {
            if settings.contextLength < 131072 { settings.contextLength = 131072 }
            settings.save()
            UserDefaults.standard.set(true, forKey: "TridaChat.context128k")
        }
        if !UserDefaults.standard.bool(forKey: "TridaChat.maxTokens64k") {
            if settings.maxTokens <= 4096 { settings.maxTokens = 65536 }
            settings.save()
            UserDefaults.standard.set(true, forKey: "TridaChat.maxTokens64k")
        }
        if !UserDefaults.standard.bool(forKey: "TridaChat.sampling06") {
            if settings.temperature == 1.0 { settings.temperature = 0.6 }
            settings.topK = 20
            settings.save()  // didSet does not fire inside init
            UserDefaults.standard.set(true, forKey: "TridaChat.sampling06")
        }
        server.onOutput = { [weak self] text in
            Task { @MainActor in self?.appendLog(text) }
        }
        server.onExit = { [weak self] code in
            Task { @MainActor in self?.serverExited(code) }
        }
        refreshModels()
        // Start tracking which app the user works in (the UI agent's target).
        uiTarget = UIAgent.shared.targetName
        UIAgent.shared.onTargetChange = { [weak self] name in self?.uiTarget = name }
    }

    // MARK: - models and server

    var selectedModel: LocalModel? {
        models.first { $0.url.path == settings.modelPath }
    }

    var servedName: String {
        (selectedModel?.name ?? "trida").lowercased()
    }

    var client: ChatClient { ChatClient(baseURL: settings.baseURL) }

    /// Development builds can run the server from a trida-stack checkout even when the app
    /// carries its own Python runtime. Stored outside AppSettings so old settings still decode.
    @Published var useProjectRuntime = UserDefaults.standard.bool(forKey: "TridaChat.useProjectRuntime") {
        didSet { UserDefaults.standard.set(useProjectRuntime, forKey: "TridaChat.useProjectRuntime") }
    }

    var hasBundledRuntime: Bool { PythonRuntime.bundledPython() != nil }

    var runtime: PythonRuntime {
        PythonRuntime.resolve(project: settings.mlxProject, preferProject: useProjectRuntime)
    }

    func refreshModels() {
        models = ModelCatalog.scan(settings.modelsDir)
        if selectedModel == nil, let m = ModelCatalog.preferred(models) {
            settings.modelPath = m.url.path
        }
    }

    func selectModel(_ m: LocalModel) {
        guard m.url.path != settings.modelPath else { return }
        settings.modelPath = m.url.path
        if launchedByUs || status == .stopped || isFailed { startServer() }
    }

    private var isFailed: Bool {
        if case .failed = status { return true }
        return false
    }

    private func appendLog(_ s: String) {
        serverLog += s
        if serverLog.count > 200_000 { serverLog = String(serverLog.suffix(150_000)) }
        if case .starting = status,
           let last = s.split(separator: "\n").last(where: { !$0.trimmingCharacters(in: .whitespaces).isEmpty }) {
            status = .starting(String(last.prefix(160)))
        }
    }

    private func serverExited(_ code: Int32) {
        launchedByUs = false
        switch status {
        case .stopped: break
        default: status = .failed("Server exited (code \(code)). Open the server log for details.")
        }
    }

    /// Starts the app-managed server for the selected model, or attaches to one that is
    /// already listening on the port (e.g. started from a terminal).
    func startServer() {
        startupTask?.cancel()
        guard let model = selectedModel else {
            status = .failed("No model in \(settings.modelsDirPath). Use Server → Get a Model…, or change the folder in Settings.")
            showModelSetup = true
            return
        }
        let cfg = settings
        let name = servedName
        startupTask = Task {
            if launchedByUs {
                status = .starting("Stopping the previous model…")
                let s = server
                await Task.detached { s.stop() }.value
                launchedByUs = false
            } else if let card = await client.modelCard() {
                attach(card, external: true)
                return
            }
            serverLog = ""
            status = .starting("Launching \(model.name)…")
            // Server-side sampling defaults apply to clients that send none (e.g. Hermes pointed at
            // this server); the app's own requests always send the Settings values.
            var extra = ["--context-length", String(cfg.contextLength),
                         "--temperature", String(cfg.temperature), "--top-k", String(cfg.topK),
                         "--top-p", String(cfg.topP)]
            if logRequests {
                try? FileManager.default.createDirectory(at: Self.requestLogDir, withIntermediateDirectories: true)
                extra += ["--log-requests", Self.requestLogDir.path]
            }
            // The server's generation guards are off by default (evals, speedrun); the app runs with them.
            extra += ["--loop-guard", "--tool-call-gate", "0.7"]
            if cfg.genBlock > 0 { extra += ["--gen-block", String(cfg.genBlock)] }
            do {
                try server.start(.init(runtime: runtime, model: model.url, port: cfg.port,
                                       servedName: name, extraArgs: extra))
                launchedByUs = true
            } catch {
                status = .failed(error.localizedDescription)
                return
            }
            // Loading a 4B q8 model takes a few seconds; the first `uv run` may also sync the venv.
            for _ in 0..<1200 {
                try? await Task.sleep(nanoseconds: 500_000_000)
                if Task.isCancelled || !launchedByUs { return }
                if let card = await client.modelCard() {
                    attach(card, external: false)
                    return
                }
            }
            status = .failed("Server did not come up within 10 minutes. See the server log.")
        }
    }

    private func attach(_ card: [String: Any], external: Bool) {
        let id = card["id"] as? String ?? "model"
        supportsVision = card["supports_vision"] as? Bool ?? false
        supportsDecision = card["decision"] as? Bool ?? false
        status = external ? .external(id) : .ready(id)
    }

    func openRequestLogs() {
        try? FileManager.default.createDirectory(at: Self.requestLogDir, withIntermediateDirectories: true)
        NSWorkspace.shared.open(Self.requestLogDir)
    }

    func stopServer() {
        startupTask?.cancel()
        cancelGeneration()
        status = .stopped
        let s = server
        launchedByUs = false
        Task.detached { s.stop() }
    }

    /// Called from applicationWillTerminate: blocks briefly so the Python process never outlives the app.
    func shutdown() {
        chatStore.save(chats, cards: decisionCards)
        startupTask?.cancel()
        generationTask?.cancel()
        server.stop()
    }

    // MARK: - model setup

    var setupRecipeName: String {
        String(setupRepo.trimmingCharacters(in: .whitespaces).split(separator: "/").last ?? "model")
    }

    var setupOutputFolder: String {
        ModelRecipe(repo: setupRepo, note: "").convertedFolder(bits: setupBits)
    }

    var setupAlreadyConverted: Bool {
        FileManager.default.fileExists(atPath: settings.modelsDir
            .appendingPathComponent(setupOutputFolder).appendingPathComponent("config.json").path)
    }

    /// Download `setupRepo` into the models folder, convert it to MLX at `setupBits`, then load it.
    func prepareModel() {
        guard !setupRunning else { return }
        let repo = setupRepo.trimmingCharacters(in: .whitespaces)
        guard repo.contains("/") else {
            setupError = "Enter a Hugging Face repo id such as trillionlabs/Zoro-1."
            return
        }
        let recipe = ModelRecipe(repo: repo, note: "")
        let cfg = settings
        let bits = setupBits
        let raw = cfg.modelsDir.appendingPathComponent(recipe.rawFolder())
        let out = cfg.modelsDir.appendingPathComponent(recipe.convertedFolder(bits: bits))
        let token = setupToken.trimmingCharacters(in: .whitespacesAndNewlines)
        let deleteRaw = setupDeleteRaw && bits < 16
        setupError = nil
        setupLog = TerminalLog()
        setupRunning = true
        let runner = setupRunner
        setupTask = Task {
            defer { setupRunning = false }
            // Converting loads the full bf16 checkpoint; don't hold a second model in memory meanwhile.
            if launchedByUs {
                setupPhase = "Stopping the running model…"
                startupTask?.cancel()
                status = .stopped
                launchedByUs = false
                let s = server
                await Task.detached { s.stop() }.value
            }
            try? FileManager.default.createDirectory(at: cfg.modelsDir, withIntermediateDirectories: true)
            let rt = runtime
            do {
                try rt.validate()
            } catch {
                setupPhase = "Failed."
                setupError = error.localizedDescription
                return
            }
            var env = rt.environment
            if !token.isEmpty { env["HF_TOKEN"] = token }
            let onOutput: @Sendable (String) -> Void = { text in
                Task { @MainActor in self.setupLog.append(text) }
            }
            do {
                setupPhase = "Downloading \(repo) from Hugging Face…"
                let (dlExe, dlArgs) = rt.download(repo: repo, to: raw)
                let dl = try await runner.run(dlExe, dlArgs, cwd: rt.workingDirectory, extraEnv: env, onOutput: onOutput)
                if Task.isCancelled { setupPhase = "Cancelled."; return }
                guard dl == 0 else {
                    setupPhase = "Download failed."
                    setupError = "The download exited with code \(dl). If the log mentions 401, 403 or gated, accept the "
                        + "model's terms on huggingface.co/\(repo) and enter an access token."
                    return
                }
                if bits < 16 {
                    setupPhase = "Converting to MLX \(bits)-bit… (a few minutes)"
                    let (cvExe, cvArgs) = rt.convert(["--model", raw.path, "--out", out.path, "--bits", String(bits)])
                    let cv = try await runner.run(cvExe, cvArgs, cwd: rt.workingDirectory, extraEnv: rt.environment,
                                                  onOutput: onOutput)
                    if Task.isCancelled { setupPhase = "Cancelled."; return }
                    guard cv == 0 else {
                        setupPhase = "Conversion failed."
                        setupError = "The conversion exited with code \(cv). See the log above."
                        return
                    }
                    if deleteRaw {
                        setupPhase = "Removing the original download…"
                        try? FileManager.default.removeItem(at: raw)
                    }
                }
            } catch {
                setupPhase = "Failed."
                setupError = error.localizedDescription
                return
            }
            setupPhase = "Done. Loading \(out.lastPathComponent)…"
            refreshModels()
            if let m = models.first(where: { $0.url.standardizedFileURL.path == out.standardizedFileURL.path }) {
                settings.modelPath = m.url.path
            }
            showModelSetup = false
            startServer()
        }
    }

    func cancelPrepare() {
        setupTask?.cancel()
        setupRunner.cancel()
    }

    // MARK: - chats

    var currentChatIndex: Int? {
        chats.firstIndex { $0.id == selectedChatID }
    }

    func newChat() {
        if let i = currentChatIndex, chats[i].messages.isEmpty { return }
        let c = Chat()
        chats.insert(c, at: 0)
        selectedChatID = c.id
    }

    func deleteChat(_ id: Chat.ID) {
        if isGenerating, id == selectedChatID { cancelGeneration() }
        if decideFlow?.chatID == id { decideFlow = nil }
        chats.removeAll { $0.id == id }
        chatStore.delete(id)
        if chats.isEmpty { chats = [Chat()] }
        if selectedChatID == id || currentChatIndex == nil { selectedChatID = chats.first?.id }
    }

    func update(_ chatID: Chat.ID, _ msgID: UUID, _ f: (inout ChatMessage) -> Void) {
        guard let ci = chats.firstIndex(where: { $0.id == chatID }),
              let mi = chats[ci].messages.firstIndex(where: { $0.id == msgID }) else { return }
        f(&chats[ci].messages[mi])
    }

    func append(_ chatID: Chat.ID, _ m: ChatMessage) {
        guard let ci = chats.firstIndex(where: { $0.id == chatID }) else { return }
        chats[ci].messages.append(m)
    }

    // MARK: - generation

    func send(_ text: String, images: [Data]) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !isGenerating, status.isUsable, !(trimmed.isEmpty && images.isEmpty),
              let ci = currentChatIndex else { return }
        let chatID = chats[ci].id
        // /decide: a decision about the conversation, set up step by step in the chat.
        if decideFlow?.chatID == chatID {
            if !trimmed.isEmpty { handleDecideInput(trimmed, chatID: chatID) }
            return
        }
        if trimmed.hasPrefix("/"), runSlash(trimmed) { return }
        if chats[ci].messages.isEmpty {
            chats[ci].title = trimmed.isEmpty ? "Image" : String(trimmed.prefix(48))
        }
        chats[ci].messages.append(ChatMessage(role: .user, content: trimmed, images: supportsVision ? images : []))
        generationTask = Task { await runTurn(chatID) }
    }

    func cancelGeneration() {
        decideTask?.cancel()
        decideTask = nil
        generationTask?.cancel()
        generationTask = nil
        resolveApproval(false)
    }

    /// Regenerate the last assistant reply.
    func retry() {
        guard !isGenerating, let ci = currentChatIndex else { return }
        let chatID = chats[ci].id
        while let last = chats[ci].messages.last, last.role != .user {
            chats[ci].messages.removeLast()
        }
        guard !chats[ci].messages.isEmpty else { return }
        generationTask = Task { await runTurn(chatID) }
    }

    /// Who the model is, first in every system prompt (Zoro-1 checkpoints only; e.g. "Zoro-1-mlx-q8").
    var modelIdentity: String? {
        guard servedName.contains("zoro") else { return nil }
        var s = "You are Zoro-1, a language model made by Trillion Labs. You run on-device on the user's Mac."
        if supportsVision { s += " You can see the images the user attaches, including earlier ones in this conversation." }
        return s + " Reply in the language the user writes in."
    }

    func history(_ chatID: Chat.ID, tools: AgentTools?) -> [[String: Any]] {
        guard let chat = chats.first(where: { $0.id == chatID }) else { return [] }
        var out: [[String: Any]] = []
        var system = settings.systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if var p = tools?.systemPrompt {
            if decisionToolsActive { p += " " + DecisionTools.systemPrompt }
            if uiToolsActive { p += " " + UITools.systemPrompt(target: UIAgent.shared.targetName) }
            system = system.isEmpty ? p : p + "\n\n" + system
        }
        if let id = modelIdentity { system = system.isEmpty ? id : id + " " + system }
        // After a compaction the model sees its summary instead of the messages before it.
        let start = compactionIndex(chat)
        if let start, let summary = chat.messages[start].compaction {
            system += (system.isEmpty ? "" : "\n\n") + "Summary of the earlier conversation (the messages before "
                + "it were summarized to free up context):\n" + summary
        }
        if !system.isEmpty { out.append(["role": "system", "content": system]) }
        for m in chat.messages[(start.map { $0 + 1 } ?? 0)...] where (m.error == nil || !m.content.isEmpty) && m.decide != .prompt {
            if case .card(_)? = m.decide, m.content.isEmpty { continue }   // a decision that did not finish
            if m.superseded == true { continue }                            // a draft rewritten after review
            out.append(m.openAIMessage())
        }
        return out
    }

    private func runTurn(_ chatID: Chat.ID) async {
        isGenerating = true
        defer {
            isGenerating = false
            prefill = nil
        }
        let s = settings
        let tools: AgentTools? = s.agentMode
            ? AgentTools(root: s.workspace, allowShell: s.allowShell, allowWrite: s.allowWrite,
                         vision: supportsVision, allowScreenshot: allowScreenshot,
                         shellSession: chatID.uuidString) : nil
        let client = self.client
        let ui = uiToolsActive
        uiSession = UISession()
        if ui { installStopMonitor() }
        defer { removeStopMonitor() }
        let dec = decisionToolsActive
        let specs = tools.map { $0.specs + (dec ? DecisionTools.specs(ui: ui) : []) + (ui ? UITools.specs : []) }
        let lastIsUser = chats.first { $0.id == chatID }?.messages.last?.role == .user
        // Nearly full context: summarize the earlier conversation first (see Compaction.swift).
        if lastIsUser, needsCompaction(chatID) {
            let pct = Int((Double(contextUsed(chatID)) / Double(s.contextLength) * 100).rounded())
            await compact(chatID, keepLastUser: true, reason: "the context was \(pct)% full")
            if Task.isCancelled { return }
        }
        var overflowCompacted = false
        // Tool routing: the decision model picks which tools this message may use (see ToolRouter).
        var allowed: Set<String>? = nil
        var pending: ChatMessage? = nil
        if let specs, !specs.isEmpty, lastIsUser, toolRouting != .off, supportsDecision {
            let first = ChatMessage(role: .assistant, content: "")
            append(chatID, first)
            pending = first
            let route = await routeTools(chatID: chatID, offered: ToolRouter.names(specs), mode: toolRouting) { [weak self] t in
                self?.update(chatID, first.id) { $0.route = t }
            }
            if Task.isCancelled {
                update(chatID, first.id) { $0.error = "Stopped." }
                return
            }
            allowed = route.allowed
            update(chatID, first.id) { $0.route = route.note.isEmpty ? nil : route.note }
        }
        for step in 0..<max(1, s.agentMode ? s.maxAgentSteps : 1) {
            var body = ChatClient.requestBody(messages: history(chatID, tools: tools), tools: specs,
                                              options: s.generation, model: servedName)
            // No tool needed: the tools stay in the prompt (prompt cache) but cannot be called.
            if allowed?.isEmpty == true { body["tool_choice"] = "none" }
            // The tool-call gate (server default 0.7) guards against stray calls. When the router has decided
            // that tools are needed, or the model is continuing after a tool result, a call is expected: relax it.
            else if (allowed.map { !$0.isEmpty } ?? false) || step > 0 { body["tool_call_min_prob"] = 0.2 }
            let reply = pending ?? ChatMessage(role: .assistant, content: "")
            if pending == nil { append(chatID, reply) }
            pending = nil
            var calls: [ToolCall] = []
            do {
                for try await event in client.stream(body: body) {
                    switch event {
                    case let .prefill(done, total):
                        prefill = (done, total)
                    case let .content(t):
                        prefill = nil
                        update(chatID, reply.id) { $0.content += t }
                    case let .reasoning(t):
                        prefill = nil
                        update(chatID, reply.id) { $0.reasoning += t }
                    case let .toolCalls(cs):
                        calls += cs
                        update(chatID, reply.id) { m in m.toolRuns += cs.map { ToolRun(call: $0) } }
                    case .finish:
                        break
                    case let .usage(st):
                        update(chatID, reply.id) { $0.stats = st }
                    }
                }
            } catch {
                let cancelled = Task.isCancelled || (error as? URLError)?.code == .cancelled || error is CancellationError
                // The prompt no longer fits: summarize the conversation once and try this step again.
                if !cancelled, !overflowCompacted, Self.isContextOverflow(error) {
                    overflowCompacted = true
                    if let ci = chats.firstIndex(where: { $0.id == chatID }) {
                        chats[ci].messages.removeAll { $0.id == reply.id }
                    }
                    if await compact(chatID, keepLastUser: true, reason: "the prompt was longer than the context") {
                        continue
                    }
                }
                update(chatID, reply.id) { $0.error = cancelled ? "Stopped." : error.localizedDescription }
                return
            }
            if Task.isCancelled {
                update(chatID, reply.id) { $0.error = "Stopped." }
                return
            }
            // A degenerate reply can repeat one call many times (e.g. calculator("1") until max_tokens; the
            // server's loop guard stops it after a few). Run each distinct call once, at most 8 per step.
            var seen = Set<String>()
            let distinct = calls.filter { seen.insert($0.name + "\u{0}" + $0.arguments).inserted }
            if distinct.count != calls.count || distinct.count > 8 {
                calls = Array(distinct.prefix(8))
                let keep = Set(calls.map(\.id))
                update(chatID, reply.id) { m in m.toolRuns.removeAll { !keep.contains($0.id) } }
            }
            if calls.isEmpty {
                // A finished text reply: check it, and rewrite it once if a check fails (ReplyReview.swift).
                if !Task.isCancelled { await reviewAndRevise(chatID, replyID: reply.id, specs: specs) }
                return
            }
            guard let tools else { return }

            for call in calls {
                if Task.isCancelled { return }
                var result: String
                if let allowed, !allowed.contains(call.name) {
                    let text = "error: \(call.name) is not needed for this message (the tool router allowed "
                        + (allowed.isEmpty ? "no tools" : allowed.sorted().joined(separator: ", "))
                        + "). Answer from the conversation, or use an allowed tool."
                    setRun(chatID, reply.id, call.id) { $0.status = .denied; $0.result = text }
                    append(chatID, toolMessage(call, text))
                    continue
                }
                if dec, DecisionTools.names.contains(call.name), call.name != "ui_check" || ui {
                    setRun(chatID, reply.id, call.id) { $0.status = .running }
                    let output = await runDecisionTool(call, chatID: chatID, tools: tools) { [weak self] text in
                        self?.setRun(chatID, reply.id, call.id) { $0.result = text }
                    }
                    setRun(chatID, reply.id, call.id) { run in
                        run.status = .done
                        run.result = output.text
                    }
                    append(chatID, toolMessage(call, output.text))
                    continue
                }
                if ui, UITools.names.contains(call.name) {
                    setRun(chatID, reply.id, call.id) { $0.status = .running }
                    let output = await runUITool(call) { [weak self] text in
                        self?.setRun(chatID, reply.id, call.id) { $0.result = text }
                    }
                    setRun(chatID, reply.id, call.id) { run in
                        run.status = output.text.hasPrefix("user denied") ? .denied : .done
                        run.result = output.text
                    }
                    append(chatID, toolMessage(call, output.text))
                    continue
                }
                let readOnlyShell = call.name == "run_shell" && autoApproveReadOnlyShell
                    && ShellSession.isReadOnly(call.argumentDict["command"] as? String ?? "")
                if AgentTools.needsApproval.contains(call.name), !readOnlyShell {
                    setRun(chatID, reply.id, call.id) { $0.status = .waitingForApproval }
                    let ok = await requestApproval(call)
                    if !ok {
                        result = "user denied"
                        setRun(chatID, reply.id, call.id) { $0.status = .denied; $0.result = result }
                        append(chatID, toolMessage(call, result))
                        continue
                    }
                }
                setRun(chatID, reply.id, call.id) { $0.status = .running }
                let output = await tools.run(call)
                result = output.text
                setRun(chatID, reply.id, call.id) { run in
                    run.status = .done
                    run.result = output.text
                    run.images = output.images
                }
                append(chatID, toolMessage(call, result, images: output.images))
            }
        }
        append(chatID, {
            var m = ChatMessage(role: .assistant, content: "")
            m.error = "Stopped after \(s.maxAgentSteps) agent steps."
            return m
        }())
    }

    private func toolMessage(_ call: ToolCall, _ result: String, images: [Data] = []) -> ChatMessage {
        var m = ChatMessage(role: .tool, content: result, images: images)
        m.toolCallID = call.id
        m.toolName = call.name
        return m
    }

    func setRun(_ chatID: Chat.ID, _ msgID: UUID, _ callID: String, _ f: (inout ToolRun) -> Void) {
        update(chatID, msgID) { m in
            if let i = m.toolRuns.firstIndex(where: { $0.id == callID }) { f(&m.toolRuns[i]) }
        }
    }

    private func requestApproval(_ call: ToolCall) async -> Bool {
        await requestApproval(ApprovalRequest(toolName: call.name, detail: AgentTools.describe(call)))
    }

    func requestApproval(_ req: ApprovalRequest) async -> Bool {
        resolveApproval(false)
        if Task.isCancelled { return false }
        return await withCheckedContinuation { (cont: CheckedContinuation<Bool, Never>) in
            approvalContinuation = cont
            approval = req
        }
    }

    /// `allowAll`: the user chose "Allow all UI actions for this message".
    func resolveApproval(_ ok: Bool, allowAll: Bool = false) {
        if ok && allowAll { uiSession.allowAll = true }
        approval = nil
        if let c = approvalContinuation {
            approvalContinuation = nil
            c.resume(returning: ok)
        }
    }

    // MARK: - UI agent

    /// ⌘⎋ stops the turn, also while another app is frontmost (the global monitor needs the
    /// Accessibility permission, which the UI agent has anyway).
    private func installStopMonitor() {
        removeStopMonitor()
        let isStop: (NSEvent) -> Bool = { $0.keyCode == 53 && $0.modifierFlags.contains(.command) }
        if let g = NSEvent.addGlobalMonitorForEvents(matching: .keyDown, handler: { [weak self] e in
            if isStop(e) { Task { @MainActor in self?.cancelGeneration() } }
        }) { stopMonitors.append(g) }
        if let l = NSEvent.addLocalMonitorForEvents(matching: .keyDown, handler: { [weak self] e in
            guard isStop(e) else { return e }
            Task { @MainActor in self?.cancelGeneration() }
            return nil
        }) { stopMonitors.append(l) }
    }

    private func removeStopMonitor() {
        for m in stopMonitors { NSEvent.removeMonitor(m) }
        stopMonitors = []
    }

    /// Decision tools (decide_choice, decide_yes_no, pick_file, ui_check) in agent mode, with a decision head.
    var decisionToolsActive: Bool { settings.agentMode && supportsDecision }

    /// Whether the selected model folder has the decision head the server loads.
    var modelHasDecisionHead: Bool {
        guard let m = selectedModel else { return false }
        return ["jev_head.pt", "jev_head.safetensors"].contains {
            FileManager.default.fileExists(atPath: m.url.appendingPathComponent($0).path)
        }
    }

    /// Fetches jev_head.pt from trillionlabs/Zoro-1 into the selected model folder (checkpoints converted
    /// before the decision support did not carry it), then restarts the server so it loads the head.
    func downloadDecisionHead() {
        guard !headDownloading, let model = selectedModel else { return }
        headDownloading = true
        headMessage = "Downloading jev_head.pt from trillionlabs/Zoro-1…"
        let rt = runtime
        let runner = CommandRunner()
        Task {
            defer { headDownloading = false }
            let (exe, args) = rt.downloadFile(repo: "trillionlabs/Zoro-1", file: "jev_head.pt", to: model.url)
            headLog = ""
            let code = (try? await runner.run(exe, args, cwd: rt.workingDirectory, extraEnv: rt.environment,
                                              onOutput: { t in Task { @MainActor in self.headLog += t } })) ?? -1
            try? await Task.sleep(nanoseconds: 100_000_000)   // let the last output land
            if code == 0, modelHasDecisionHead {
                headMessage = "Downloaded. Restarting the server…"
                if launchedByUs || status == .stopped || isFailed { startServer() }
                else { headMessage = "Downloaded. Restart the server you started so it loads the head." }
            } else {
                headMessage = "Download failed (exit \(code)). " + String(headLog.suffix(300))
            }
        }
    }

    // MARK: - pickers

    func chooseWorkspace() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.directoryURL = settings.workspace
        panel.prompt = "Use as Workspace"
        if panel.runModal() == .OK, let url = panel.url {
            settings.workspacePath = url.path
        }
    }

    func chooseFolder(_ keyPath: WritableKeyPath<AppSettings, String>) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.directoryURL = AppSettings.url(settings[keyPath: keyPath])
        if panel.runModal() == .OK, let url = panel.url {
            settings[keyPath: keyPath] = url.path
            if keyPath == \AppSettings.modelsDirPath { refreshModels() }
        }
    }
}
