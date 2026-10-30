import AppKit
import SwiftUI
import TridaChatCore
import UniformTypeIdentifiers

struct ChatView: View {
    @EnvironmentObject var state: AppState

    private var chat: Chat? {
        state.currentChatIndex.map { state.chats[$0] }
    }

    var body: some View {
        VStack(spacing: 0) {
            if let chat, !chat.messages.isEmpty {
                MessageList(chat: chat)
            } else {
                EmptyChatView()
            }
            Divider()
            Composer()
        }
    }
}

struct EmptyChatView: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        VStack(spacing: 10) {
            Image(systemName: "bubble.left.and.text.bubble.right")
                .font(.system(size: 40))
                .foregroundStyle(.secondary)
            Text(state.selectedModel?.name ?? "No model selected")
                .font(.title2)
            Text(state.settings.selfSpec
                 ? "Self-speculative decoding: diffusion drafts, exact AR verification."
                 : "Plain autoregressive decoding.")
                .foregroundStyle(.secondary)
            if state.models.isEmpty {
                Text("No converted model in \(state.settings.modelsDirPath) yet.")
                    .foregroundStyle(.secondary)
                Button("Get Zoro-1…") {
                    state.setupRepo = ModelRecipe.known[0].repo
                    state.showModelSetup = true
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.large)
            } else if !state.status.isUsable {
                Text(state.status.label)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 480)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }
}

struct MessageList: View {
    @EnvironmentObject var state: AppState
    let chat: Chat

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 14) {
                    ForEach(chat.messages.filter { $0.role != .tool && $0.role != .system }) { m in
                        MessageView(message: m, isLast: m.id == chat.messages.last?.id)
                            .id(m.id)
                    }
                    Color.clear.frame(height: 1).id("bottom")
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 16)
            }
            .onAppear { proxy.scrollTo("bottom", anchor: .bottom) }
            .onChange(of: chat.messages) { _, _ in
                proxy.scrollTo("bottom", anchor: .bottom)
            }
        }
    }
}

struct Composer: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if !slashMatches.isEmpty {
                SlashMenu(matches: slashMatches)
            }
            if !state.draftImages.isEmpty {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                        ForEach(Array(state.draftImages.enumerated()), id: \.offset) { i, data in
                            ZStack(alignment: .topTrailing) {
                                Thumbnail(data: data, side: 64)
                                Button {
                                    state.draftImages.remove(at: i)
                                } label: {
                                    Image(systemName: "xmark.circle.fill")
                                        .symbolRenderingMode(.palette)
                                        .foregroundStyle(.white, .black.opacity(0.6))
                                }
                                .buttonStyle(.plain)
                                .padding(2)
                            }
                        }
                    }
                }
            }
            HStack(alignment: .bottom, spacing: 8) {
                Menu {
                    Button("Choose Image…") { state.importingImage = true }
                    Button("Paste Image") { pasteImage() }
                } label: {
                    Image(systemName: "photo.badge.plus")
                }
                .menuStyle(.borderlessButton)
                .fixedSize()
                .disabled(!state.supportsVision)
                .help(state.supportsVision ? "Attach an image (or drag one in)" : "This model has no vision encoder")

                TextField(placeholder, text: $state.draft, axis: .vertical)
                    .textFieldStyle(.plain)
                    .lineLimit(1...10)
                    .onSubmit(send)
                    .onKeyPress(.upArrow) { moveSlash(-1) }
                    .onKeyPress(.downArrow) { moveSlash(1) }
                    .onKeyPress(.tab) {
                        guard let c = selectedSlash else { return .ignored }
                        state.draft = c.name + (c.hint.isEmpty ? "" : " ")
                        return .handled
                    }
                    .onKeyPress(.escape) {
                        guard !slashMatches.isEmpty else { return .ignored }
                        state.draft = ""
                        return .handled
                    }
                    .onChange(of: state.draft) { _, _ in state.slashSelection = 0 }
                    .padding(8)
                    .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .textBackgroundColor)))
                    .overlay(RoundedRectangle(cornerRadius: 8)
                        .stroke(state.dropTargeted ? Color.accentColor : Color.secondary.opacity(0.3)))

                if state.isGenerating {
                    Button {
                        state.cancelGeneration()
                    } label: {
                        Image(systemName: "stop.circle.fill").font(.title2)
                    }
                    .buttonStyle(.plain)
                    .keyboardShortcut(".", modifiers: .command)
                    .help("Stop (⌘.)")
                } else {
                    Button(action: send) {
                        Image(systemName: "arrow.up.circle.fill").font(.title2)
                    }
                    .buttonStyle(.plain)
                    .disabled(!canSend)
                    .help("Send (Return; Option-Return for a new line)")
                }
            }
        }
        .padding(12)
        .onDrop(of: [.fileURL, .image], isTargeted: $state.dropTargeted) { providers in
            guard state.supportsVision else { return false }
            loadImages(providers)
            return true
        }
        .fileImporter(isPresented: $state.importingImage, allowedContentTypes: [.image], allowsMultipleSelection: true) { result in
            if case let .success(urls) = result {
                for u in urls {
                    if let d = ImageUtil.pngData(contentsOf: u) { state.draftImages.append(d) }
                }
            }
        }
    }

    private var placeholder: String {
        if !state.status.isUsable { return "Waiting for the server…" }
        if let p = state.decidePlaceholder { return p }
        return state.settings.agentMode ? "Ask the agent (workspace: \(AppSettings.url(state.settings.workspacePath).lastPathComponent))…"
                                        : "Message \(state.selectedModel?.name ?? "the model")…"
    }

    private var canSend: Bool {
        state.status.isUsable && !state.isGenerating
            && !(state.draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && state.draftImages.isEmpty)
    }

    private var slashMatches: [SlashCommand] { state.slashMatches }

    private var selectedSlash: SlashCommand? {
        let m = slashMatches
        return m.isEmpty ? nil : m[min(state.slashSelection, m.count - 1)]
    }

    private func moveSlash(_ d: Int) -> KeyPress.Result {
        let n = slashMatches.count
        guard n > 0 else { return .ignored }
        state.slashSelection = (min(state.slashSelection, n - 1) + d + n) % n
        return .handled
    }

    private func send() {
        // The slash menu is open: Return runs the highlighted command ("/de" + Return = /decide).
        if let c = selectedSlash {
            state.runSlash(c.name)
            return
        }
        guard canSend else { return }
        state.send(state.draft, images: state.draftImages)
        state.draft = ""
        state.draftImages = []
    }

    private func pasteImage() {
        let pb = NSPasteboard.general
        if let objs = pb.readObjects(forClasses: [NSImage.self]) as? [NSImage] {
            for img in objs {
                if let d = ImageUtil.pngData(from: img) { state.draftImages.append(d) }
            }
        }
    }

    private func loadImages(_ providers: [NSItemProvider]) {
        let s = state
        for p in providers {
            if p.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
                _ = p.loadObject(ofClass: URL.self) { url, _ in
                    guard let url, let d = ImageUtil.pngData(contentsOf: url) else { return }
                    Task { @MainActor in s.draftImages.append(d) }
                }
            } else if p.canLoadObject(ofClass: NSImage.self) {
                _ = p.loadObject(ofClass: NSImage.self) { obj, _ in
                    guard let img = obj as? NSImage, let d = ImageUtil.pngData(from: img) else { return }
                    Task { @MainActor in s.draftImages.append(d) }
                }
            }
        }
    }
}

struct Thumbnail: View {
    let data: Data
    let side: CGFloat

    var body: some View {
        Group {
            if let img = NSImage(data: data) {
                Image(nsImage: img).resizable().scaledToFill()
            } else {
                Color.secondary
            }
        }
        .frame(width: side, height: side)
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}

enum ImageUtil {
    /// PNG bytes, downscaled so the longest side is at most `maxSide` (the server caps image
    /// tokens anyway; this keeps the request small).
    static func pngData(from image: NSImage, maxSide: CGFloat = 1536) -> Data? {
        guard let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else { return nil }
        let w = CGFloat(cg.width), h = CGFloat(cg.height)
        let scale = min(1, maxSide / max(w, h, 1))
        let nw = max(1, Int(w * scale)), nh = max(1, Int(h * scale))
        guard let ctx = CGContext(data: nil, width: nw, height: nh, bitsPerComponent: 8, bytesPerRow: 0,
                                  space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        ctx.interpolationQuality = .high
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: nw, height: nh))
        guard let out = ctx.makeImage() else { return nil }
        return NSBitmapImageRep(cgImage: out).representation(using: .png, properties: [:])
    }

    static func pngData(contentsOf url: URL) -> Data? {
        guard let img = NSImage(contentsOf: url) else { return nil }
        return pngData(from: img)
    }
}
