import AppKit
import SwiftUI
import TridaChatCore

struct ContentView: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        NavigationSplitView {
            List(selection: $state.selectedChatID) {
                ForEach(state.chats) { chat in
                    Text(chat.title)
                        .lineLimit(1)
                        .tag(chat.id)
                        .contextMenu {
                            Button("Delete Chat", role: .destructive) { state.deleteChat(chat.id) }
                        }
                }
            }
            .navigationSplitViewColumnWidth(min: 180, ideal: 220)
            .toolbar {
                ToolbarItem {
                    Button { state.newChat() } label: { Image(systemName: "square.and.pencil") }
                        .help("New chat (⌘N)")
                }
            }
        } detail: {
            ChatView()
                .toolbar { toolbar }
        }
        .safeAreaInset(edge: .bottom, spacing: 0) {
            StatusBar()
        }
        .sheet(isPresented: $state.showLog) {
            ServerLogView()
        }
        .sheet(isPresented: $state.showModelSetup) {
            ModelSetupView()
        }
        .sheet(item: $state.approval) { req in
            ApprovalSheet(request: req)
        }
        .sheet(item: $state.editTarget) { _ in
            EditReplySheet()
        }
    }

    @ToolbarContentBuilder
    private var toolbar: some ToolbarContent {
        ToolbarItemGroup(placement: .principal) {
            Menu {
                if state.models.isEmpty {
                    Text("No models in \(state.settings.modelsDirPath)")
                }
                ForEach(state.models) { m in
                    Button {
                        state.selectModel(m)
                    } label: {
                        if m.url.path == state.settings.modelPath {
                            Label("\(m.name)  ·  \(m.precision)  ·  \(m.sizeText)", systemImage: "checkmark")
                        } else {
                            Text("\(m.name)  ·  \(m.precision)  ·  \(m.sizeText)")
                        }
                    }
                }
                Divider()
                Button("Get a Model…") { state.showModelSetup = true }
                Button("Rescan Models") { state.refreshModels() }
            } label: {
                Label(state.selectedModel?.name ?? "Choose model", systemImage: "cpu")
            }
            .help("Model (from \(state.settings.modelsDirPath))")

            Picker("Decoding", selection: $state.settings.selfSpec) {
                Text("Self-spec").tag(true)
                Text("AR").tag(false)
            }
            .pickerStyle(.segmented)
            .frame(width: 150)
            .help("Self-speculative decoding (diffusion draft + exact AR verify) or plain autoregressive")
        }
        ToolbarItemGroup(placement: .primaryAction) {

            Toggle(isOn: $state.settings.agentMode) {
                Label("Agent", systemImage: "wrench.and.screwdriver")
            }
            .help("Give the model tools over the workspace folder")

            if state.settings.agentMode {
                Button { state.chooseWorkspace() } label: {
                    Label(AppSettings.url(state.settings.workspacePath).lastPathComponent, systemImage: "folder")
                }
                .help("Workspace: \(state.settings.workspacePath)")
            }
            OpenDecideButton(label: "Decide")
        }
    }
}

struct StatusBar: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(color)
                .frame(width: 8, height: 8)
            Text(state.status.label)
                .lineLimit(1)
                .truncationMode(.middle)
            if let p = state.prefill, p.total > 0 {
                ProgressView(value: Double(p.done), total: Double(p.total))
                    .frame(width: 120)
                Text("prefill \(p.done.formatted())/\(p.total.formatted())")
            }
            Spacer()
            if state.settings.agentMode {
                Text(toolsLabel)
                    .foregroundStyle(.secondary)
            }
            Button("Server Log") { state.showLog = true }
                .buttonStyle(.link)
            switch state.status {
            case .starting, .ready:
                Button("Stop") { state.stopServer() }.buttonStyle(.link)
            default:
                Button("Start") { state.startServer() }.buttonStyle(.link)
            }
        }
        .font(.caption)
        .padding(.horizontal, 12)
        .padding(.vertical, 5)
        .background(.bar)
    }

    private var toolsLabel: String {
        var t = ["read"]
        if state.settings.allowWrite { t.append("write") }
        if state.settings.allowShell { t.append("shell") }
        if state.supportsVision { t.append(state.allowScreenshot ? "image, screenshot" : "image") }
        if state.decisionToolsActive { t.append("decide") }
        if state.uiToolsActive { t.append("UI → \(state.uiTarget) (⌘⎋ stops)") }
        return "tools: " + t.joined(separator: ", ")
    }

    private var color: Color {
        switch state.status {
        case .ready, .external: return .green
        case .starting: return .orange
        case .failed: return .red
        case .stopped: return .gray
        }
    }
}

struct ApprovalSheet: View {
    @EnvironmentObject var state: AppState
    let request: ApprovalRequest

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label("Allow \(request.toolName)?", systemImage: "exclamationmark.shield")
                .font(.headline)
            ScrollView {
                Text(request.detail)
                    .font(.system(.body, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .frame(minHeight: 60, maxHeight: 240)
            .padding(8)
            .background(RoundedRectangle(cornerRadius: 6).fill(Color(nsColor: .textBackgroundColor)))
            Text(request.caption ?? "Workspace: \(state.settings.workspacePath)")
                .font(.caption)
                .foregroundStyle(.secondary)
            HStack {
                Spacer()
                Button("Deny", role: .cancel) { state.resolveApproval(false) }
                    .keyboardShortcut(.cancelAction)
                if request.offerAllowAll {
                    Button("Allow All for This Message") { state.resolveApproval(true, allowAll: true) }
                        .help("Allow the remaining UI actions of this reply without asking (unsure decisions still ask)")
                }
                Button("Allow") { state.resolveApproval(true) }
                    .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(width: 520)
        .interactiveDismissDisabled()
    }
}

struct ServerLogView: View {
    @EnvironmentObject var state: AppState
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Server log").font(.headline)
                Spacer()
                Button("Copy") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(state.serverLog, forType: .string)
                }
                Button("Close") { dismiss() }
                    .keyboardShortcut(.defaultAction)
            }
            ScrollViewReader { proxy in
                ScrollView {
                    Text(state.serverLog.isEmpty ? "(no output yet)" : state.serverLog)
                        .font(.system(size: 11, design: .monospaced))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    Color.clear.frame(height: 1).id("end")
                }
                .onAppear { proxy.scrollTo("end") }
                .onChange(of: state.serverLog) { _, _ in proxy.scrollTo("end") }
            }
        }
        .padding(16)
        .frame(width: 760, height: 480)
    }
}
