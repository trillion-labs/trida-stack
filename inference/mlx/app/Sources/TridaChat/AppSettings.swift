import Foundation
import TridaChatCore

/// Everything the user can change, persisted as one JSON blob in UserDefaults.
struct AppSettings: Codable, Equatable {
    var mlxProjectPath: String = "~/Desktop/trillionlabs/trida-stack/inference/mlx"
    var modelsDirPath: String = "~/models"
    var modelPath: String = ""
    var port: Int = 8080
    /// 0 = the server's default (N = 4, fastest on Apple Silicon).
    var genBlock: Int = 0
    var contextLength: Int = 131072

    var selfSpec: Bool = true
    /// Kept so older saved settings still decode; thinking is always on (see `generation`).
    var thinking: Bool = true
    var temperature: Double = 0.6   // Zoro-1 card, reasoned-action / tool-use setting
    var topK: Int = 20
    var topP: Double = 0.95
    var maxTokens: Int = 65536

    var agentMode: Bool = false
    var workspacePath: String = "~/Desktop"
    var allowShell: Bool = true
    var allowWrite: Bool = false
    var maxAgentSteps: Int = 12

    var systemPrompt: String = ""

    static let key = "TridaChat.settings.v1"

    static func load() -> AppSettings {
        guard let data = UserDefaults.standard.data(forKey: key),
              let s = try? JSONDecoder().decode(AppSettings.self, from: data) else { return AppSettings() }
        return s
    }

    func save() {
        if let data = try? JSONEncoder().encode(self) {
            UserDefaults.standard.set(data, forKey: Self.key)
        }
    }

    static func url(_ path: String) -> URL {
        URL(fileURLWithPath: (path as NSString).expandingTildeInPath)
    }

    var mlxProject: URL { Self.url(mlxProjectPath) }
    var modelsDir: URL { Self.url(modelsDirPath) }
    var workspace: URL { Self.url(workspacePath) }
    var baseURL: URL { URL(string: "http://127.0.0.1:\(port)/v1")! }

    var generation: GenerationOptions {
        var o = GenerationOptions()
        o.selfSpec = selfSpec
        o.thinking = true   // always think; the reasoning shows collapsed above each reply
        o.temperature = temperature
        o.topK = topK
        o.topP = topP
        o.maxTokens = maxTokens
        return o
    }
}
