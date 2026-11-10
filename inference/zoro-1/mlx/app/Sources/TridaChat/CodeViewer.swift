import AppKit
import SwiftUI
import TridaChatCore

/// The right-hand code viewer: the files the model wrote in this chat, each as a diff against the file before
/// the write, or as the whole file as it is now on disk.
struct CodeViewer: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        let writes = state.viewerWrites
        let current = writes.last { $0.messageID == state.viewerTarget?.messageID && $0.run.id == state.viewerTarget?.runID }
            ?? writes.last
        VStack(spacing: 0) {
            header(writes, current)
            Divider()
            if let current {
                CodeViewerBody(run: current.run, diff: state.viewerShowsDiff)
                    .id("\(current.run.id)-\(state.viewerShowsDiff)")
            } else {
                VStack(spacing: 8) {
                    Image(systemName: "doc.text.magnifyingglass").font(.largeTitle).foregroundStyle(Theme.muted)
                    Text("Files the model writes in this chat show up here.")
                        .font(.callout).foregroundStyle(Theme.muted).multilineTextAlignment(.center)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding()
            }
        }
        .background(Theme.canvas)
    }

    func header(_ writes: [(messageID: UUID, run: ToolRun)], _ current: (messageID: UUID, run: ToolRun)?) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Menu {
                    ForEach(Array(writes.enumerated().reversed()), id: \.offset) { i, w in
                        Button("\(i + 1). \(Self.name(w.run))") {
                            state.viewerTarget = ViewerTarget(messageID: w.messageID, runID: w.run.id)
                        }
                    }
                } label: {
                    Label(current.map { Self.name($0.run) } ?? "No files", systemImage: "doc.text")
                        .lineLimit(1)
                }
                .menuStyle(.borderlessButton)
                .disabled(writes.isEmpty)
                Spacer()
                Picker("", selection: $state.viewerShowsDiff) {
                    Text("Changes").tag(true)
                    Text("File").tag(false)
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .frame(width: 140)
            }
            if let current, let path = Self.path(current.run) {
                HStack(spacing: 10) {
                    Text(path).font(.caption).foregroundStyle(Theme.muted).lineLimit(1).truncationMode(.head)
                        .textSelection(.enabled)
                    Spacer()
                    Button { NSWorkspace.shared.open(URL(fileURLWithPath: path)) } label: {
                        Image(systemName: "arrow.up.forward.app")
                    }
                    .help("Open in its default app")
                    Button { NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)]) } label: {
                        Image(systemName: "folder")
                    }
                    .help("Show in Finder")
                    if current.run.undo != nil, current.run.undone != true, let chat = state.selectedChatID {
                        Button {
                            state.undoWrite(chatID: chat, messageID: current.messageID, runID: current.run.id)
                        } label: { Image(systemName: "arrow.uturn.backward") }
                        .help("Undo this write")
                    }
                }
                .buttonStyle(.borderless)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
    }

    static func path(_ run: ToolRun) -> String? { run.undo?.path ?? run.call.argumentDict["path"] as? String }
    static func name(_ run: ToolRun) -> String { path(run).map { ($0 as NSString).lastPathComponent } ?? run.call.name }
}

/// The lines of one write: the diff (changed lines with 3 lines of context) or the file now on disk.
struct CodeViewerBody: View {
    let run: ToolRun
    let diff: Bool

    var body: some View {
        let rows = diff ? diffRows() : fileRows()
        if rows.isEmpty {
            Text(diff ? "No changes: the file already had this content." : "The file is empty or no longer exists.")
                .font(.callout).foregroundStyle(Theme.muted)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            VStack(spacing: 0) {
                if diff { summary }
                ScrollView([.vertical, .horizontal]) {
                    LazyVStack(alignment: .leading, spacing: 0) {
                        ForEach(Array(rows.enumerated()), id: \.offset) { _, r in row(r) }
                    }
                    .padding(.vertical, 6)
                    .textSelection(.enabled)
                }
            }
        }
    }

    var before: String {
        guard let b = run.undo?.backup else { return "" }      // nil backup: the write created the file
        return (try? String(contentsOfFile: b, encoding: .utf8)) ?? ""
    }
    var after: String { run.call.argumentDict["content"] as? String ?? "" }

    var summary: some View {
        let d = LineDiff.diff(before, after)
        let added = d.filter { $0.op == .added }.count, removed = d.filter { $0.op == .removed }.count
        return HStack(spacing: 10) {
            Text(run.undo?.backup == nil ? "New file" : "Edited").font(.caption.weight(.semibold))
            Text("+\(added)").font(.caption.monospacedDigit()).foregroundStyle(Theme.accent)
            Text("−\(removed)").font(.caption.monospacedDigit()).foregroundStyle(Theme.danger)
            if run.undone == true { Text("Undone").font(.caption).foregroundStyle(Theme.muted) }
            Spacer()
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(Theme.surfaceRaised)
    }

    func diffRows() -> [LineDiff.Line?] {
        let d = LineDiff.diff(before, after)
        // a new file is all additions: show it whole rather than as hunks
        return run.undo?.backup == nil ? d.map { Optional($0) } : LineDiff.hunks(d)
    }

    func fileRows() -> [LineDiff.Line?] {
        guard let p = CodeViewer.path(run), let t = try? String(contentsOfFile: p, encoding: .utf8) else { return [] }
        return LineDiff.lines(String(t.prefix(2_000_000))).enumerated().map {
            LineDiff.Line(op: .same, text: $1, old: nil, new: $0 + 1)
        }
    }

    @ViewBuilder
    func row(_ r: LineDiff.Line?) -> some View {
        if let r {
            HStack(alignment: .top, spacing: 0) {
                Text(r.old.map(String.init) ?? "").frame(width: diff ? 40 : 0, alignment: .trailing)
                    .opacity(diff ? 1 : 0)
                Text(r.new.map(String.init) ?? "").frame(width: 40, alignment: .trailing)
                Text(r.op == .added ? " + " : r.op == .removed ? " − " : "   ")
                    .foregroundStyle(r.op == .added ? Theme.accent : r.op == .removed ? Theme.danger : Theme.muted)
                Text(r.text.isEmpty ? " " : r.text)
                    .foregroundStyle(Theme.text)
                    .fixedSize(horizontal: true, vertical: false)
                    .padding(.trailing, 12)
            }
            .font(.system(size: 12, design: .monospaced))
            .foregroundStyle(Theme.muted)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(r.op == .added ? Theme.accentSoft.opacity(0.7)
                        : r.op == .removed ? Theme.danger.opacity(0.12) : Color.clear)
        } else {
            Text("⋯").font(.system(size: 12, design: .monospaced)).foregroundStyle(Theme.muted)
                .padding(.leading, 84).padding(.vertical, 2)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Theme.surfaceRaised.opacity(0.5))
        }
    }
}
