import AppKit
import SwiftUI
import TridaChatCore

struct ContentView: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        NavigationSplitView {
            // A List without a selection binding, with our own selected look: List(selection:) draws the
            // selection in the system accent (blue by default), which no SwiftUI modifier changes. Keeping the
            // List (rather than a ScrollView of buttons) keeps the sidebar's layout under the toolbar and its
            // hit testing.
            VStack(spacing: 0) {
                HStack(spacing: 6) {
                    Image(systemName: "magnifyingglass").foregroundStyle(Theme.muted)
                    TextField("Search chats", text: $state.chatSearch)
                        .textFieldStyle(.plain)
                    if !state.chatSearch.isEmpty {
                        Button { state.chatSearch = "" } label: { Image(systemName: "xmark.circle.fill") }
                            .buttonStyle(.plain).foregroundStyle(Theme.muted)
                    }
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 5)
                .background(RoundedRectangle(cornerRadius: 7, style: .continuous).fill(Theme.surface))
                .overlay(RoundedRectangle(cornerRadius: 7, style: .continuous).stroke(Theme.border))
                .padding(.horizontal, 10)
                .padding(.top, 6)
                .padding(.bottom, 4)
                List {
                    ForEach(state.visibleChats) { chat in
                        ChatRow(chat: chat, selected: chat.id == state.selectedChatID)
                            .listRowInsets(EdgeInsets(top: 1, leading: 4, bottom: 1, trailing: 4))
                            .listRowBackground(Color.clear)
                    }
                }
                .listStyle(.sidebar)
                .scrollContentBackground(.hidden)
                if state.showMascot { MascotPanel() }
            }
            .background(Theme.sidebar)
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
                // the toolbar takes the canvas color instead of the system grey band
                .toolbarBackground(Theme.canvas, for: .windowToolbar)
                .toolbarBackground(.visible, for: .windowToolbar)
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
        .sheet(item: $state.renaming) { _ in
            RenameSheet()
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
                    .labelStyle(.titleAndIcon)
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
                        .labelStyle(.titleAndIcon)
                }
                .help("Workspace: \(state.settings.workspacePath)")
            }
            OpenDecideButton(label: "Decide")
        }
    }
}

struct ChatRow: View {
    @EnvironmentObject var state: AppState
    let chat: Chat
    let selected: Bool

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: selected ? "bubble.left.fill" : "bubble.left")
                .foregroundStyle(Theme.accent)
            Text(chat.title)
                .lineLimit(1)
                .fontWeight(selected ? .semibold : .regular)
                .foregroundStyle(Theme.text)
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(RoundedRectangle(cornerRadius: 7, style: .continuous).fill(selected ? Theme.accentSoft : Color.clear))
        .contentShape(Rectangle())          // the whole row, not just the text, takes the click
        .onTapGesture { state.selectedChatID = chat.id }
        .contextMenu {
            Button("Rename…") { state.beginRename(chat.id) }
            Button("Delete Chat", role: .destructive) { state.deleteChat(chat.id) }
        }
        .accessibilityAddTraits(selected ? .isSelected : [])
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
                    .foregroundStyle(Theme.muted)
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .help(toolsLabel)
            }
            Button("Server Log") { state.showLog = true }
                .buttonStyle(.plain).foregroundStyle(Theme.accent)
            switch state.status {
            case .starting, .ready:
                Button("Stop Server") { state.stopServer() }.buttonStyle(.plain).foregroundStyle(Theme.accent)
            case .external:
                // a server the app didn't launch: it can't stop it, and starting one would clash on the port
                EmptyView()
            default:
                Button("Start Server") { state.startServer() }.buttonStyle(.plain).foregroundStyle(Theme.accent)
            }
        }
        .font(.caption)
        .padding(.horizontal, 12)
        .padding(.vertical, 5)
        .background(Theme.sidebar)
        .overlay(alignment: .top) { Rectangle().fill(Theme.border).frame(height: 1) }
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
        case .ready, .external: return Theme.success
        case .starting: return Theme.warning
        case .failed: return Theme.danger
        case .stopped: return Theme.muted
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
            .card(radius: 8)
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
