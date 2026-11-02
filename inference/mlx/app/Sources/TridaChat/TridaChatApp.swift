import AppKit
import SwiftUI

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let state = AppState()

    func applicationDidFinishLaunching(_ notification: Notification) {
        // A bare `swift run` binary starts as a background process; make it a regular app.
        NSApp.setActivationPolicy(.regular)
        NSApp.activate(ignoringOtherApps: true)
        state.startServer()
        state.updateAutomation()
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
            CommandGroup(after: .newItem) {
                OpenDecideButton()
                    .keyboardShortcut("d", modifiers: [.command, .shift])
            }
        }

        Window("Decide", id: "decide") {
            DecideView(panel: delegate.state.decidePanel)
                .environmentObject(delegate.state)
        }

        Settings {
            SettingsView()
                .environmentObject(delegate.state)
        }
    }
}
