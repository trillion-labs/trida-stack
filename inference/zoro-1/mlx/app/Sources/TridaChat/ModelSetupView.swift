import AppKit
import SwiftUI
import TridaChatCore

/// "Get a Model": download a checkpoint from Hugging Face and convert it to MLX, then load it.
struct ModelSetupView: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Get a Model").font(.title2).bold()
            Text("Downloads the checkpoint from Hugging Face into \(state.settings.modelsDirPath), converts it to MLX, and loads it.")
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)

            Form {
                Picker("Model", selection: $state.setupRepo) {
                    ForEach(ModelRecipe.known) { r in
                        Text("\(r.name) — \(r.note)").tag(r.repo)
                    }
                    if !ModelRecipe.known.contains(where: { $0.repo == state.setupRepo }) {
                        Text(state.setupRepo).tag(state.setupRepo)
                    }
                }
                TextField("Hugging Face repo", text: $state.setupRepo)
                    .textFieldStyle(.roundedBorder)
                Picker("Precision", selection: $state.setupBits) {
                    Text("8-bit (recommended, ~5 GB)").tag(8)
                    Text("6-bit (~4 GB)").tag(6)
                    Text("4-bit (~3 GB, lower quality)").tag(4)
                    Text("bf16, no conversion (~10 GB)").tag(16)
                }
                if !ProjectTool.huggingFaceLoggedIn() {
                    SecureField("Hugging Face access token (hf_…)", text: $state.setupToken)
                        .textFieldStyle(.roundedBorder)
                    HStack(spacing: 12) {
                        Link("Create a token", destination: URL(string: "https://huggingface.co/settings/tokens")!)
                        if let page = URL(string: "https://huggingface.co/\(state.setupRepo)") {
                            Link("Accept the model's terms", destination: page)
                        }
                    }
                    .font(.caption)
                    Text("Gated models need a token from an account that accepted the terms. It is used for this download only and not saved.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                if state.setupBits < 16 {
                    Toggle("Delete the original download after converting", isOn: $state.setupDeleteRaw)
                }
                LabeledContent("Saved as") {
                    Text("\(state.settings.modelsDirPath)/\(state.setupOutputFolder)")
                        .font(.system(.body, design: .monospaced))
                        .lineLimit(1).truncationMode(.middle)
                }
                if let free = ProjectTool.freeSpace(at: state.settings.modelsDir) {
                    LabeledContent("Free disk space") {
                        Text(ByteCountFormatter.string(fromByteCount: free, countStyle: .file))
                            .foregroundStyle(free < 16_000_000_000 ? Color.red : Color.primary)
                    }
                }
            }
            .disabled(state.setupRunning)

            if state.setupAlreadyConverted && !state.setupRunning {
                Label("\(state.setupOutputFolder) already exists; running again re-downloads missing files and re-converts.",
                      systemImage: "info.circle")
                    .font(.caption).foregroundStyle(.secondary)
            }

            if state.setupRunning || !state.setupLog.current.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: 8) {
                        if state.setupRunning { ProgressView().controlSize(.small) }
                        Text(state.setupPhase).bold()
                    }
                    Text(state.setupLog.current)
                        .font(.system(size: 11, design: .monospaced))
                        .lineLimit(2)
                        .foregroundStyle(.secondary)
                    ScrollViewReader { proxy in
                        ScrollView {
                            Text(state.setupLog.text)
                                .font(.system(size: 10, design: .monospaced))
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                            Color.clear.frame(height: 1).id("end")
                        }
                        .frame(height: 140)
                        .background(RoundedRectangle(cornerRadius: 6).fill(Color(nsColor: .textBackgroundColor)))
                        .onChange(of: state.setupLog) { _, _ in proxy.scrollTo("end") }
                    }
                }
            }

            if let err = state.setupError {
                Label(err, systemImage: "exclamationmark.triangle.fill")
                    .foregroundStyle(.red)
                    .fixedSize(horizontal: false, vertical: true)
            }

            HStack {
                Spacer()
                if state.setupRunning {
                    Button("Cancel") { state.cancelPrepare() }
                } else {
                    Button("Close") { state.showModelSetup = false }
                        .keyboardShortcut(.cancelAction)
                    Button("Download & Convert") { state.prepareModel() }
                        .keyboardShortcut(.defaultAction)
                }
            }
        }
        .padding(20)
        .frame(width: 600)
        .interactiveDismissDisabled(state.setupRunning)
    }
}
