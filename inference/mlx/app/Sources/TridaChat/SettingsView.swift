import SwiftUI
import TridaChatCore

struct SettingsView: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        TabView {
            serverTab.tabItem { Label("Server", systemImage: "server.rack") }
            generationTab.tabItem { Label("Generation", systemImage: "slider.horizontal.3") }
            agentTab.tabItem { Label("Agent", systemImage: "wrench.and.screwdriver") }
        }
        .padding(20)
        .frame(width: 560)
    }

    private func folderRow(_ title: String, _ keyPath: WritableKeyPath<AppSettings, String>) -> some View {
        LabeledContent(title) {
            HStack {
                TextField("", text: Binding(get: { state.settings[keyPath: keyPath] },
                                              set: { state.settings[keyPath: keyPath] = $0 }))
                    .textFieldStyle(.roundedBorder)
                Button("Choose…") { state.chooseFolder(keyPath) }
            }
        }
    }

    private var serverTab: some View {
        Form {
            LabeledContent("Runtime") {
                Text(state.runtime.label).lineLimit(2).truncationMode(.middle)
            }
            if state.hasBundledRuntime {
                Toggle("Use a trida-stack checkout instead (development)", isOn: $state.useProjectRuntime)
            }
            if !state.hasBundledRuntime || state.useProjectRuntime {
                folderRow("MLX project", \.mlxProjectPath)
                Text("The trida-stack/inference/mlx folder. The app runs its .venv (after `uv sync`) or `uv run`.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            folderRow("Models folder", \.modelsDirPath)
            Text("Converted checkpoints, e.g. ~/models/Zoro-1-mlx-q8 (trida-mlx-convert --bits 8).")
                .font(.caption).foregroundStyle(.secondary)
            LabeledContent("Port") {
                TextField("", value: $state.settings.port, format: .number.grouping(.never))
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 90)
            }
            Picker("Block size N", selection: $state.settings.genBlock) {
                Text("Default (4)").tag(0)
                Text("2").tag(2)
                Text("4").tag(4)
                Text("8").tag(8)
            }
            Text("Tokens drafted per self-spec step (canvas 2N−1). N = 4 is fastest on M-series Macs.")
                .font(.caption).foregroundStyle(.secondary)
            Picker("Context length", selection: $state.settings.contextLength) {
                Text("16K").tag(16384)
                Text("32K").tag(32768)
                Text("64K").tag(65536)
                Text("128K").tag(131072)
            }
            HStack {
                Spacer()
                Button("Rescan Models") { state.refreshModels() }
                Button("Restart Server") { state.startServer() }
                    .keyboardShortcut(.defaultAction)
            }
            Toggle("Log requests and raw model output", isOn: $state.logRequests)
            Text("One JSON file per request in ~/Library/Logs/Zoro-1 Chat/requests (Server → Open Request Logs): the messages sent, the model's raw text before tool-call parsing, and the parsed result.")
                .font(.caption).foregroundStyle(.secondary)
            Text("Port, block size, context length, logging and the server's sampling defaults apply when the server restarts.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var generationTab: some View {
        Form {
            Toggle("Self-speculative decoding", isOn: $state.settings.selfSpec)
            LabeledContent("Temperature") {
                HStack {
                    Slider(value: $state.settings.temperature, in: 0...2, step: 0.05)
                    Text(String(format: "%.2f", state.settings.temperature)).monospacedDigit().frame(width: 40)
                }
            }
            Stepper("Top-k: \(state.settings.topK)", value: $state.settings.topK, in: 1...200)
            LabeledContent("Top-p") {
                HStack {
                    Slider(value: $state.settings.topP, in: 0.05...1, step: 0.01)
                    Text(String(format: "%.2f", state.settings.topP)).monospacedDigit().frame(width: 40)
                }
            }
            Stepper("Max new tokens: \(state.settings.maxTokens)", value: $state.settings.maxTokens,
                    in: 256...65536, step: 1024)
            Text("Thinking and the answer share this budget; the server also caps it at what is left of the context length.")
                .font(.caption).foregroundStyle(.secondary)
            Text("Defaults follow the Zoro-1 card's tool-use setting: temperature 0.6, top-k 20, top-p 0.95 (the card uses 1.0 for plain generation). Temperature 0 = greedy.")
                .font(.caption).foregroundStyle(.secondary)
            LabeledContent("System prompt") {
                TextEditor(text: $state.settings.systemPrompt)
                    .font(.body)
                    .frame(height: 80)
                    .overlay(RoundedRectangle(cornerRadius: 4).stroke(Color.secondary.opacity(0.3)))
            }
        }
    }

    private var agentTab: some View {
        Form {
            Toggle("Agent mode (tools)", isOn: $state.settings.agentMode)
            LabeledContent("Workspace") {
                HStack {
                    Text(state.settings.workspacePath).lineLimit(1).truncationMode(.middle)
                    Spacer()
                    Button("Choose…") { state.chooseWorkspace() }
                }
            }
            Text("Always available: list_dir, read_file, find_files (by name), search_files (inside files), calculator, now — confined to the workspace.")
                .font(.caption).foregroundStyle(.secondary)
            Toggle("Allow write_file (asks every time)", isOn: $state.settings.allowWrite)
            Toggle("Allow run_shell (a terminal for the model)", isOn: $state.settings.allowShell)
            Toggle("Run read-only commands without asking (ls, cat, grep, git status/log/diff, …)",
                   isOn: $state.autoApproveReadOnlyShell)
                .disabled(!state.settings.allowShell)
            Text("Each chat has its own terminal: the working directory and exported variables carry over between commands (it starts in the workspace). Commands run with your user's permissions; anything that is not clearly read-only asks first. Timeout 300 s by default, up to 30 min; long output keeps its start and end.")
                .font(.caption).foregroundStyle(.secondary)
            Divider()
            Text("Image tools (only offered when the model has a vision encoder: \(state.supportsVision ? "yes" : "no") for the current model)")
                .font(.callout)
            Text("view_image: always available, looks at an image file in the workspace.")
                .font(.caption).foregroundStyle(.secondary)
            Toggle("Allow screenshot (on by default; asks every time)", isOn: $state.allowScreenshot)
            Text("Captures your main display. macOS asks once for Screen Recording permission (System Settings → Privacy & Security → Screen Recording); a rebuilt app may need to be allowed again.")
                .font(.caption).foregroundStyle(.secondary)
            Stepper("Max tool steps per message: \(state.settings.maxAgentSteps)",
                    value: $state.settings.maxAgentSteps, in: 1...50)
            Divider()
            Text("Decision tools (offered when the model has a decision head: \(state.supportsDecision ? "yes" : "no")): decide_choice, decide_yes_no, pick_file, and ui_check with the UI agent. The decision model scores the options; below 0.7 it reasons once (self-spec) and decides again. Shift-⌘D opens the Decide window to try decisions by hand.")
                .font(.caption).foregroundStyle(.secondary)
            Toggle("Check replies with the decision model", isOn: $state.reviewReplies)
            Text("After each reply the decision model checks: the language the user asked for, does what was asked, complete, not a repeat of the previous reply (reasoning first when unsure). If one fails, the model rewrites the reply once with that feedback; the draft stays visible, collapsed. Works in chat and agent mode.")
                .font(.caption).foregroundStyle(.secondary)
            Picker("Tool routing", selection: $state.toolRouting) {
                ForEach(ToolRouting.allCases) { Text($0.label).tag($0) }
            }
            Text("Before each message, the decision model picks what the reply needs: answer directly, or one kind of tool (files, screen, time, math, shell, decide, app). It decides directly, without a reasoning pass, so it adds well under a second. Tools it rules out cannot be called: with \"answer\", the server bans <tool_call> for that reply. If it is still unsure (below 0.4), every tool stays available. The result shows above the reply.")
                .font(.caption).foregroundStyle(.secondary)
            Divider()
            Toggle("Allow automation (simulated users, local API)", isOn: $state.automationEnabled)
            Text("Lets a program on this Mac use the app like a person (trida-mlx-simuser: an LLM playing the user) through 127.0.0.1:\(AutomationServer.port), with the token in \(AutomationServer.tokenFile.path). Turns run the normal way, so approvals still ask you.")
                .font(.caption).foregroundStyle(.secondary)
            if let e = state.automationError {
                Text("Could not start: \(e)").font(.caption).foregroundStyle(.red)
            }
            Divider()
            uiAgentSection
        }
        .formStyle(.grouped)   // scrolls: this tab is taller than the window
        .frame(minHeight: 520)
    }

    @ViewBuilder
    private var uiAgentSection: some View {
        Toggle("UI agent (beta): operate other apps", isOn: $state.uiAgentEnabled)
        Text("Adds ui_elements, ui_click, ui_type, ui_key and open_app. The app operated is the one you used last before switching to Zoro-1 Chat (now: \(state.uiTarget)). ui_click is a typed decision by Zoro-1's pointer head over that app's accessibility elements; when the top probability is below 0.7 the model reasons once (self-spec) and decides again. Every action asks first, unless you choose Allow All for that message. ⌘⎋ stops.")
            .font(.caption).foregroundStyle(.secondary)
        LabeledContent("Accessibility") {
            HStack {
                Text(UIAgent.trusted ? "allowed" : "not allowed yet")
                    .foregroundStyle(UIAgent.trusted ? Color.green : Color.orange)
                Spacer()
                if !UIAgent.trusted {
                    Button("Allow…") { UIAgent.requestTrust() }
                }
                Button("Open Settings") { UIAgent.openAccessibilitySettings() }
            }
        }
        LabeledContent("Decision head") {
            HStack {
                if state.supportsDecision {
                    Text("loaded").foregroundStyle(.green)
                } else if state.modelHasDecisionHead {
                    Text("in the model folder; restart the server").foregroundStyle(.orange)
                } else {
                    Text("missing (jev_head.pt)").foregroundStyle(.orange)
                }
                Spacer()
                if !state.supportsDecision && !state.modelHasDecisionHead {
                    Button(state.headDownloading ? "Downloading…" : "Download") { state.downloadDecisionHead() }
                        .disabled(state.headDownloading || state.selectedModel == nil)
                }
            }
        }
        if let m = state.headMessage {
            Text(m).font(.caption).foregroundStyle(.secondary).lineLimit(4)
        }
        Toggle("Send the window's screenshot with each click decision", isOn: $state.uiDecisionScreenshot)
        Text("Off by default: in the screen-agent demo the element list alone did better than the list plus a screenshot; the image temperature is not calibrated yet.")
            .font(.caption).foregroundStyle(.secondary)
    }
}
