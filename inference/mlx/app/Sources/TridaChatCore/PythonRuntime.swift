import Foundation

/// Where the Python side (trida_mlx: server, convert, Hugging Face download) runs from.
///
/// * `.bundled` — a self-contained CPython + site-packages inside the app
///   (`Contents/Resources/python`, made by `build_app.sh --bundle`). Nothing else to install.
/// * `.project` — a trida-stack `inference/mlx` checkout with its `uv sync`ed `.venv`
///   (development builds).
public enum PythonRuntime: Equatable, Sendable {
    case bundled(python: URL)
    case project(URL)

    /// `<App>/Contents/Resources/python/bin/python3`, if this build carries one.
    public static func bundledPython(in bundle: Bundle = .main) -> URL? {
        guard let res = bundle.resourceURL else { return nil }
        let py = res.appendingPathComponent("python/bin/python3")
        return FileManager.default.isExecutableFile(atPath: py.path) ? py : nil
    }

    public static func resolve(project: URL, preferProject: Bool) -> PythonRuntime {
        if !preferProject, let py = bundledPython() { return .bundled(python: py) }
        return .project(project)
    }

    public var isBundled: Bool {
        if case .bundled = self { return true }
        return false
    }

    public var label: String {
        switch self {
        case .bundled: return "Built-in Python runtime (inside the app)"
        case let .project(dir): return "MLX project at \(dir.path)"
        }
    }

    public var workingDirectory: URL {
        switch self {
        case .bundled: return FileManager.default.homeDirectoryForCurrentUser
        case let .project(dir): return dir
        }
    }

    /// The bundled interpreter must never write into the (signed) app bundle or pick up the
    /// user's own site-packages.
    public var environment: [String: String] {
        switch self {
        case .bundled:
            return ["PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"]
        case .project:
            return ["PYTHONUNBUFFERED": "1"]
        }
    }

    public enum RuntimeError: LocalizedError {
        case projectMissing(String)

        public var errorDescription: String? {
            switch self {
            case let .projectMissing(p):
                return "No MLX project at \(p). Set the trida-stack/inference/mlx folder in Settings, or use a build made with build_app.sh --bundle."
            }
        }
    }

    public func validate() throws {
        if case let .project(dir) = self,
           !FileManager.default.fileExists(atPath: dir.appendingPathComponent("pyproject.toml").path) {
            throw RuntimeError.projectMissing(dir.path)
        }
    }

    func module(_ module: String, tool: String, _ args: [String]) -> (executable: URL, arguments: [String]) {
        switch self {
        case let .bundled(py): return (py, ["-m", module] + args)
        case let .project(dir): return ProjectTool.command(tool, project: dir, args: args)
        }
    }

    public func server(_ args: [String]) -> (executable: URL, arguments: [String]) {
        module("trida_mlx.server", tool: "trida-mlx-server", args)
    }

    public func convert(_ args: [String]) -> (executable: URL, arguments: [String]) {
        module("trida_mlx.convert", tool: "trida-mlx-convert", args)
    }

    /// trida_mlx.datamine: chats + request logs -> candidate training data (JSONL, scrubbed).
    public func datamine(_ args: [String]) -> (executable: URL, arguments: [String]) {
        module("trida_mlx.datamine", tool: "trida-mlx-datamine", args)
    }

    static let downloadScript = """
        import sys
        from huggingface_hub import snapshot_download
        path = snapshot_download(sys.argv[1], local_dir=sys.argv[2])
        print("downloaded to", path)
        """

    static let downloadFileScript = """
        import sys
        from huggingface_hub import hf_hub_download
        print("downloaded", hf_hub_download(sys.argv[1], sys.argv[2], local_dir=sys.argv[3]))
        """

    /// One file of a Hugging Face repo into `dir` (e.g. the decision head, jev_head.pt).
    public func downloadFile(repo: String, file: String, to dir: URL) -> (executable: URL, arguments: [String]) {
        switch self {
        case let .bundled(py): return (py, ["-c", Self.downloadFileScript, repo, file, dir.path])
        case let .project(d): return ProjectTool.command("hf", project: d, args: ["download", repo, file, "--local-dir", dir.path])
        }
    }

    public func download(repo: String, to dir: URL) -> (executable: URL, arguments: [String]) {
        switch self {
        case let .bundled(py): return (py, ["-c", Self.downloadScript, repo, dir.path])
        case let .project(d): return ProjectTool.command("hf", project: d, args: ["download", repo, "--local-dir", dir.path])
        }
    }
}
