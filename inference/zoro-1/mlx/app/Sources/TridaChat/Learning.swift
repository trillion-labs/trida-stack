import AppKit
import SwiftUI
import TridaChatCore

/// Online RL from inside the app (Settings → Learning): score the chats (trida-mlx-rewards), train a LoRA adapter
/// on those rewards (trida-mlx-rl-train), and choose which adapter the server loads (server --adapter).
struct LearningStatus: Equatable {
    enum Phase: Equatable { case idle, scoring, training }
    var phase: Phase = .idle
    /// 0...1 while training (steps done / steps planned)
    var progress: Double = 0
    var message: String = ""
    /// The last scoring run's summary line.
    var agreement: String?
    /// The last training run's summary line and report.
    var trained: String?
    var report: URL?
    /// The trainer's sanity check: the new adapter loops where the base model does not.
    var sanityFailed = false
    var log: String = ""
}

/// Splits streamed output into whole lines (a chunk can end mid-line).
final class LineSplitter: @unchecked Sendable {
    private let lock = NSLock()
    private var buf = ""
    func feed(_ s: String) -> [String] {
        lock.lock(); defer { lock.unlock() }
        buf += s
        var lines = buf.components(separatedBy: "\n")
        buf = lines.removeLast()
        return lines
    }
}

extension AppState {
    static let learningDir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("Zoro-1 Chat/learning", isDirectory: true)
    static var rewardsDir: URL { learningDir.appendingPathComponent("rewards", isDirectory: true) }
    static var adaptersDir: URL { learningDir.appendingPathComponent("adapters", isDirectory: true) }
    static var rewardsFile: URL { rewardsDir.appendingPathComponent("turn_rewards.jsonl") }

    /// Adapter folders (each with adapters.safetensors), oldest first.
    var adapters: [String] {
        let fm = FileManager.default
        let names = (try? fm.contentsOfDirectory(atPath: Self.adaptersDir.path)) ?? []
        return names.filter {
            fm.fileExists(atPath: Self.adaptersDir.appendingPathComponent("\($0)/adapters.safetensors").path)
        }.sorted()
    }

    /// The adapter the server should load: the chosen one, if it still exists.
    var activeAdapterURL: URL? {
        guard let name = adapterName, adapters.contains(name) else { return nil }
        return Self.adaptersDir.appendingPathComponent(name, isDirectory: true)
    }

    func selectAdapter(_ name: String?) {
        guard name != adapterName else { return }
        adapterName = name
        if launchedOrStarting { startServer() }
    }

    private var launchedOrStarting: Bool {
        switch status {
        case .ready, .starting, .failed: return true
        default: return false
        }
    }

    /// Runs trida-mlx-rewards against this app's server (the decision head is the judge).
    func scoreChats() {
        guard learning.phase == .idle else { return }
        guard status.isUsable, supportsDecision else {
            learning.message = "Start a model with a decision head first: the judge runs on it."
            return
        }
        guard activeAdapterURL == nil else {
            learning.message = "Switch Weights to the base model to score: the judge must not be the model being trained."
            return
        }
        chatStore.save(chats, cards: decisionCards)
        learning.phase = .scoring
        learning.message = "Scoring the chats with the decision model…"
        learning.log = ""
        let rt = runtime
        let (exe, args) = rt.rewards(["--out", Self.rewardsDir.path, "--chats", Self.chatsDir.path,
                                      "--judge-url", "http://127.0.0.1:\(settings.port)"])
        let lines = LineSplitter()
        Task {
            let code = (try? await learningRunner.run(exe, args, cwd: rt.workingDirectory, extraEnv: rt.environment,
                                                      onOutput: { t in
                for line in lines.feed(t) { Task { @MainActor in self.learningLine(line) } }
            })) ?? -1
            learning.phase = .idle
            if code == 0, let summary = learning.agreement {
                learning.message = summary
            } else if code != 0 {
                learning.message = "Scoring failed (exit \(code)). See the log below."
            }
        }
    }

    /// Stops the chat server (the model can't be in memory twice), trains a new adapter on the latest rewards,
    /// then restarts the server with the adapter that was chosen before.
    func trainAdapter() {
        guard learning.phase == .idle, let model = selectedModel else { return }
        guard FileManager.default.fileExists(atPath: Self.rewardsFile.path) else {
            learning.message = "Score the chats first."
            return
        }
        let next = String(format: "%04d", (adapters.compactMap { Int($0) }.max() ?? 0) + 1)
        let out = Self.adaptersDir.appendingPathComponent(next, isDirectory: true)
        learning.phase = .training
        learning.progress = 0
        learning.message = "Stopping the chat model and loading it for training…"
        learning.log = ""
        learning.trained = nil
        learning.sanityFailed = false
        trainingSteps = (0, 0)
        stopServer()
        let rt = runtime
        let (exe, args) = rt.rlTrain(["--model", model.url.path, "--rewards", Self.rewardsFile.path,
                                      "--out", out.path])
        let lines = LineSplitter()
        Task {
            try? await Task.sleep(nanoseconds: 1_500_000_000)   // let the server process exit and free its memory
            let code = (try? await learningRunner.run(exe, args, cwd: rt.workingDirectory, extraEnv: rt.environment,
                                                      onOutput: { t in
                for line in lines.feed(t) { Task { @MainActor in self.learningLine(line) } }
            })) ?? -1
            learning.phase = .idle
            if code == 0 {
                learning.progress = 1
                learning.report = out.appendingPathComponent("report.json")
                learning.message = learning.sanityFailed
                    ? "Adapter \(next) failed the sanity check: its replies loop where the base model's don't. Stay on the base model."
                    : "Adapter \(next) is ready. Choose it under Weights to chat with it."
            } else {
                learning.message = code == 15 || code == -1 ? "Training stopped." : "Training failed (exit \(code)). See the log below."
            }
            startServer()
        }
    }

    func cancelLearning() {
        learningRunner.cancel()
    }

    /// One line of the scorer's or trainer's output: progress, summaries, and the log.
    private func learningLine(_ line: String) {
        learning.log += line + "\n"
        if learning.log.count > 20_000 { learning.log = String(learning.log.suffix(16_000)) }
        guard line.hasPrefix("{"), let data = line.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
        if let i = obj["base"] as? Int, let n = obj["of"] as? Int {
            learning.message = "Preparing: the base model's log-probs, \(i) of \(n)"
        } else if let plan = obj["plan"] as? [String: Any], let steps = plan["steps"] as? Int {
            trainingSteps = (0, steps)
            learning.message = "Training: \(plan["samples"] as? Int ?? 0) samples, \(steps) steps"
        } else if obj["step"] != nil, let loss = obj["loss"] as? Double {
            trainingSteps.done += 1
            let (done, total) = trainingSteps
            if total > 0 { learning.progress = Double(done) / Double(total) }
            let secs = obj["seconds"] as? Double ?? 0
            let left = total > done ? Int(Double(total - done) * secs / 60) : 0
            learning.message = "Step \(done) of \(total) · loss \(String(format: "%.3f", loss))"
                + (left > 0 ? " · about \(left) min left" : "")
        } else if let stop = obj["early_stop"] as? String, obj["mean_delta_logp"] == nil {
            learning.message = "Stopped early to stay close to the base model (\(stop))"
        } else if let d = obj["mean_delta_logp"] as? [String: Any] {
            func f(_ k: String) -> String { (d[k] as? Double).map { String(format: "%+.3f", $0) } ?? "–" }
            learning.sanityFailed = (obj["sanity"] as? String ?? "").hasPrefix("failed")
            learning.trained = "Log-prob change: rewarded \(f("rewarded")) · penalised \(f("penalised"))"
                + " · held out \(f("held_out")) · sanity \(learning.sanityFailed ? "failed" : "passed")"
        } else if obj["judged"] != nil {
            let turns = obj["turns"] as? Int ?? 0, judged = obj["judged"] as? Int ?? 0
            let rated = obj["rated_turns"] as? Int ?? 0
            let auc = (obj["judge_auc_thumbs"] as? [String: Any])?["auc"] as? Double
            learning.agreement = "\(judged) of \(turns) turns judged · \(rated) covered by a rating · "
                + "👍/👎 AUC \(auc.map { String(format: "%.2f", $0) } ?? "– (needs both)")"
        }
    }
}

struct LearningTab: View {
    @EnvironmentObject var state: AppState

    var body: some View {
        Form {
            Text("Zoro-1 can learn from how you use it, on this Mac only: your ratings, 👍 / 👎 and edits, plus the decision model judging each past reply in hindsight, become rewards for a small LoRA adapter on top of the model. Experimental: train when you don't need the chat, then compare the adapter with the base model.")
                .font(.caption).foregroundStyle(.secondary)
            LabeledContent("1. Score chats") {
                HStack {
                    Text(state.learning.agreement ?? "Not scored yet").font(.callout).lineLimit(2)
                    Spacer()
                    Button("Score") { state.scoreChats() }
                        .disabled(state.learning.phase != .idle)
                }
            }
            LabeledContent("2. Train") {
                HStack {
                    if state.learning.phase == .training {
                        ProgressView(value: state.learning.progress).frame(maxWidth: 200)
                        Button("Stop") { state.cancelLearning() }
                    } else {
                        Text(state.learning.trained ?? "").font(.callout).lineLimit(2)
                        Spacer()
                        Button("Train adapter") { state.trainAdapter() }
                            .disabled(state.learning.phase != .idle || state.selectedModel == nil)
                    }
                }
            }
            Text("Training stops the chat model while it runs (the model can't be in memory twice): about 20 seconds per reply on an 18 GB M3 Pro, then the chat model starts again.")
                .font(.caption).foregroundStyle(.secondary)
            LabeledContent("3. Weights") {
                HStack {
                    Picker("", selection: Binding(get: { state.adapterName ?? "" },
                                                  set: { state.selectAdapter($0.isEmpty ? nil : $0) })) {
                        Text("Base model").tag("")
                        ForEach(state.adapters, id: \.self) { Text("Adapter \($0)").tag($0) }
                    }
                    .labelsHidden()
                    .disabled(state.learning.phase != .idle)
                    if let report = state.learning.report {
                        Button("Report") { NSWorkspace.shared.open(report) }
                    }
                    Button("Folder") {
                        try? FileManager.default.createDirectory(at: AppState.adaptersDir,
                                                                 withIntermediateDirectories: true)
                        NSWorkspace.shared.open(AppState.learningDir)
                    }
                }
            }
            if !state.learning.message.isEmpty {
                Text(state.learning.message).font(.callout)
            }
            if !state.learning.log.isEmpty {
                ScrollView {
                    Text(state.learning.log).font(.system(.caption, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading).textSelection(.enabled)
                }
                .frame(height: 140)
            }
        }
    }
}
