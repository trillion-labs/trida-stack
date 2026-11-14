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
            Composer()
                .frame(maxWidth: ChatLayout.width)
                .frame(maxWidth: .infinity)
        }
        .background(Theme.canvas)
        .background(GeometryReader { g in
            Color.clear
                .onAppear { placeMascot(g.size.width) }
                .onChange(of: g.size.width) { _, w in placeMascot(w) }
        })
    }

    /// The mascot stands beside the latest reply when the centered column leaves room for it on the left (with a
    /// little hysteresis so it does not flip back and forth while the window is resized), and in the sidebar
    /// otherwise.
    private func placeMascot(_ width: CGFloat) {
        // it hangs left of the reply's 26 pt avatar column, into the list's 24 pt padding and the margin
        let margin = (width - ChatLayout.width) / 2 + 24 - (ChatLayout.besideWidth - 26)
        let fits = state.mascotInMargin ? margin >= 4 : margin >= 16
        if fits != state.mascotInMargin {
            withAnimation(.easeInOut(duration: 0.2)) { state.mascotInMargin = fits }
        }
    }
}

/// Messages and the composer share one readable column, centered in wide windows.
enum ChatLayout {
    static let width: CGFloat = 860
    /// The mascot's width beside a reply (the 168 pt figure and its caption).
    static let besideWidth: CGFloat = 176
}

struct EmptyChatView: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        VStack(spacing: 12) {
            if state.showMascot && state.mascotInMargin {
                MascotPanel(placement: .margin).frame(width: 240)
            } else {
                ZStack {
                    Circle().fill(Theme.accentSoft).frame(width: 72, height: 72)
                    Image(systemName: "leaf.fill")
                        .font(.system(size: 30, weight: .semibold))
                        .foregroundStyle(Theme.accent)
                }
                .padding(.bottom, 4)
            }
            Text(state.selectedModel?.name ?? "No model selected")
                .font(.system(size: 28, weight: .medium, design: .serif))
                .foregroundStyle(Theme.text)
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
            } else {
                // a few ways in: a click puts the prompt in the composer
                VStack(spacing: 8) {
                    ForEach(suggestions, id: \.self) { s in
                        Button { state.draft = s } label: {
                            HStack(spacing: 8) {
                                Image(systemName: "arrow.turn.down.right").foregroundStyle(Theme.accent)
                                Text(s).foregroundStyle(Theme.text)
                                Spacer(minLength: 0)
                            }
                            .padding(.horizontal, 14)
                            .padding(.vertical, 9)
                            .frame(width: 420)
                            .card(radius: 10)
                            .contentShape(Rectangle())
                        }
                        .buttonStyle(.plain)
                    }
                }
                .padding(.top, 10)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding()
    }

    private var suggestions: [String] {
        state.settings.agentMode
            ? ["What is this folder about?", "Where is the code that saves chats?", "이 폴더에서 가장 큰 파일 5개 알려줘"]
            : ["Explain self-speculative decoding in two sentences", "Write a haiku about a forest at dawn",
               "다음 문장을 영어로 번역해줘: 오늘은 날씨가 정말 좋네요"]
    }
}

struct MessageList: View {
    @EnvironmentObject var state: AppState
    let chat: Chat

    /// The latest reply (not an app notice or a replaced draft): the mascot stands beside it.
    private var mascotHost: UUID? {
        chat.messages.last { $0.role == .assistant && $0.decide == nil && $0.superseded != true }?.id
    }

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 18) {
                    let host = state.showMascot && state.mascotInMargin ? mascotHost : nil
                    ForEach(chat.messages.filter { $0.role != .tool && $0.role != .system }) { m in
                        MessageView(message: m, isLast: m.id == chat.messages.last?.id, hostsMascot: m.id == host)
                            .id(m.id)
                    }
                    SessionRatingCard(chat: chat)
                    Color.clear.frame(height: 1).id("bottom")
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 20)
                .frame(maxWidth: ChatLayout.width)
                .frame(maxWidth: .infinity)
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
            HStack(alignment: .bottom, spacing: 10) {
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
                    .padding(.vertical, 6)

                if state.isGenerating {
                    Button {
                        state.cancelGeneration()
                    } label: {
                        roundIcon("stop.fill", fill: Theme.danger)
                    }
                    .buttonStyle(.plain)
                    .keyboardShortcut(".", modifiers: .command)
                    .help("Stop (⌘.)")
                } else {
                    Button(action: send) {
                        roundIcon("arrow.up", fill: canSend ? Theme.accent : Theme.border)
                    }
                    .buttonStyle(.plain)
                    .disabled(!canSend)
                    .help("Send (Return; Option-Return for a new line)")
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(RoundedRectangle(cornerRadius: 20, style: .continuous).fill(Theme.surface))
            .overlay(RoundedRectangle(cornerRadius: 20, style: .continuous)
                .stroke(state.dropTargeted ? Theme.accent : Theme.border, lineWidth: state.dropTargeted ? 2 : 1))
            HStack {
                Text("Return sends, Option-Return adds a line, / opens commands")
                    .font(.caption2)
                    .foregroundStyle(Theme.muted)
                Spacer()
                ModePill()
                ContextMeter()
            }
            .padding(.horizontal, 6)
        }
        .padding(.horizontal, 24)
        .padding(.top, 8)
        .padding(.bottom, 12)
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

    private func roundIcon(_ name: String, fill: Color) -> some View {
        Image(systemName: name)
            .font(.system(size: 13, weight: .bold))
            .foregroundStyle(Theme.onAccent)
            .frame(width: 28, height: 28)
            .background(Circle().fill(fill))
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

/// Manual or auto approvals, next to the context meter; click to switch (⇧⌘A).
struct ModePill: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        Button { state.autoMode.toggle() } label: {
            HStack(spacing: 4) {
                Image(systemName: state.autoMode ? "bolt.fill" : "hand.raised")
                Text(state.autoMode ? "Auto" : "Manual")
            }
            .font(.caption2.weight(.semibold))
            .padding(.horizontal, 7)
            .padding(.vertical, 2)
            .foregroundStyle(state.autoMode ? Theme.onAccent : Theme.muted)
            .background(Capsule().fill(state.autoMode ? Theme.accent : Theme.surfaceRaised))
        }
        .buttonStyle(.plain)
        .help(state.autoMode
              ? "Auto: writes and commands run without asking unless risky (Zoro-1 judges the rest). Click for manual."
              : "Manual: every write, command and screenshot asks first. Click for auto.")
    }
}
