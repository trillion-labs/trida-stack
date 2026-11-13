import AppKit
import SwiftUI
import TridaKnowledgeUI

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let state = AppState()
    private var dockTimer: Timer?
    private var dockLook: MascotLook?

    func applicationDidFinishLaunching(_ notification: Notification) {
        // A bare `swift run` binary starts as a background process; make it a regular app.
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        state.startServer()
        state.updateAutomation()
        // The Dock icon is the mascot, showing what the model is doing (redrawn only when that changes).
        dockTimer = Timer.scheduledTimer(withTimeInterval: 1, repeats: true) { [weak self] _ in
            MainActor.assumeIsolated { self?.updateDockIcon() }
        }
        updateDockIcon()
    }

    private func updateDockIcon() {
        var look = state.mascotLook
        if case .tool = look.state { look.state = .tool("") }       // one icon per kind of state, not per tool
        guard look != dockLook else { return }
        dockLook = look
        NSApp.applicationIconImage = MascotFrame.icon(look, size: 256)
    }

    func applicationWillTerminate(_ notification: Notification) {
        state.shutdown()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}

@main
struct TridaChatApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate

    var body: some Scene {
        WindowGroup("Zoro-1 Chat") {
            ContentView()
                .environmentObject(delegate.state)
                .tint(Theme.accent)
                .frame(minWidth: 760, minHeight: 520)
        }
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("New Chat") { delegate.state.newChat() }
                    .keyboardShortcut("n")
            }
            CommandMenu("Server") {
                Button("Restart Server") { delegate.state.startServer() }
                    .keyboardShortcut("r", modifiers: [.command, .shift])
                Button("Stop Server") { delegate.state.stopServer() }
                Button("Get a Model…") { delegate.state.showModelSetup = true }
                Button("Rescan Models") { delegate.state.refreshModels() }
                Divider()
                Button("Open Request Logs") { delegate.state.openRequestLogs() }
                Button("Export Training Data…") { delegate.state.exportTrainingData() }
            }
            CommandGroup(after: .sidebar) {
                OpenKnowledgeButton()
                    .keyboardShortcut("k", modifiers: [.command, .shift])
                Button("Show or Hide Mascot") { delegate.state.showMascot.toggle() }
                    .keyboardShortcut("m", modifiers: [.command, .shift])
                Button("Show or Hide Code Viewer") { delegate.state.showViewer.toggle() }
                    .keyboardShortcut("e", modifiers: [.command, .shift])
                Button("Switch Auto / Manual Approvals") {
                    delegate.state.autoMode.toggle()
                }
                .keyboardShortcut("a", modifiers: [.command, .shift])
            }
            CommandGroup(after: .newItem) {
                OpenDecideButton()
                    .keyboardShortcut("d", modifiers: [.command, .shift])
            }
        }

        Window("Knowledge", id: "knowledge") {
            KnowledgeView(model: delegate.state.knowledge)
                .tint(Theme.accent)
        }

        Window("Decide", id: "decide") {
            DecideView(panel: delegate.state.decidePanel)
                .environmentObject(delegate.state)
                .tint(Theme.accent)
        }

        Settings {
            SettingsView()
                .environmentObject(delegate.state)
                .tint(Theme.accent)
        }
    }
}
