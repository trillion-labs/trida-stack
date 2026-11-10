import AppKit
import SwiftUI
import TridaChatCore

extension AppState {
    /// The current chat's scratch folder (see `Scratch`).
    var currentScratch: URL? { selectedChatID.map { Scratch.dir($0) } }

    /// Copies a scratch file into the workspace root under its own name ("name 2.txt" when taken).
    func saveScratchToWorkspace(_ file: URL) {
        let fm = FileManager.default
        let ws = settings.workspace
        let stem = file.deletingPathExtension().lastPathComponent, ext = file.pathExtension
        var dest = ws.appendingPathComponent(file.lastPathComponent)
        var n = 2
        while fm.fileExists(atPath: dest.path) {
            dest = ws.appendingPathComponent(ext.isEmpty ? "\(stem) \(n)" : "\(stem) \(n).\(ext)")
            n += 1
        }
        do {
            try fm.copyItem(at: file, to: dest)
            scratchNote = "Saved \(dest.lastPathComponent) to \(ws.lastPathComponent)"
        } catch {
            scratchNote = "Could not save: \(error.localizedDescription)"
        }
    }
}

/// The toolbar popover: this chat's scratch files, newest first, to open, show in Finder or keep.
struct ScratchView: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        let dir = state.currentScratch
        let files = dir.map(Scratch.files) ?? []
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Scratch files").font(.headline)
                Spacer()
                if let dir, FileManager.default.fileExists(atPath: dir.path) {
                    Button("Show Folder") { NSWorkspace.shared.activateFileViewerSelecting([dir]) }
                        .buttonStyle(.link)
                }
            }
            Text("The model's working files for this chat: drafts, extracted text and long tool outputs. "
                 + "They are deleted with the chat, or after a week untouched.")
                .font(.caption).foregroundStyle(Theme.muted).fixedSize(horizontal: false, vertical: true)
            Divider()
            if files.isEmpty {
                Text("No files yet.").font(.callout).foregroundStyle(Theme.muted)
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 6) {
                        ForEach(files, id: \.self) { f in row(f, dir: dir) }
                    }
                }
                .frame(maxHeight: 320)
            }
            if let note = state.scratchNote {
                Text(note).font(.caption).foregroundStyle(Theme.accent)
            }
        }
        .padding(16)
        .frame(width: 380)
    }

    func row(_ f: URL, dir: URL?) -> some View {
        let v = try? f.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
        let rel = dir.map { f.path.replacingOccurrences(of: $0.path + "/", with: "") } ?? f.lastPathComponent
        return HStack(spacing: 8) {
            Image(systemName: "doc.text").foregroundStyle(Theme.muted)
            VStack(alignment: .leading, spacing: 1) {
                Text(rel).font(.callout).lineLimit(1).truncationMode(.middle)
                Text([v?.fileSize.map { ByteCountFormatter.string(fromByteCount: Int64($0), countStyle: .file) },
                      v?.contentModificationDate.map { $0.formatted(date: .omitted, time: .shortened) }]
                    .compactMap { $0 }.joined(separator: " · "))
                    .font(.caption2).foregroundStyle(Theme.muted)
            }
            Spacer()
            Button { NSWorkspace.shared.open(f) } label: { Image(systemName: "arrow.up.forward.app") }
                .help("Open")
            Button { NSWorkspace.shared.activateFileViewerSelecting([f]) } label: { Image(systemName: "folder") }
                .help("Show in Finder")
            Button { state.saveScratchToWorkspace(f) } label: { Image(systemName: "square.and.arrow.down") }
                .help("Save a copy to the workspace")
        }
        .buttonStyle(.borderless)
    }
}
