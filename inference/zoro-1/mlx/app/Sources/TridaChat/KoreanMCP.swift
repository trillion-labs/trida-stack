import AppKit
import Security
import SwiftUI
import TridaChatCore

/// Korean MCP servers for the agent (Settings → Korean MCP), a starter set picked from awesome-mcp-korea
/// (https://github.com/darjeeling/awesome-mcp-korea). Each is third-party code that runs on this Mac with the user's
/// permissions, so nothing starts until it is switched on; versions are pinned, and only a few of each server's tools
/// are offered to the model (a 4B model reads every tool schema in its prompt).
struct MCPCatalogEntry: Identifiable {
    struct Key {
        let env: String
        let label: String
        let signup: String
        var optional = false
    }

    /// Also the prefix of its tools' names (law_search_law).
    let id: String
    let name: String
    let summary: String
    let repo: String
    /// The shell command line that starts it (stdio).
    let command: String
    /// What has to be installed: "npx" (Node.js) or "uv".
    let needs: String
    var keys: [Key] = []
    /// The server's tools offered to the model (nil = all of them).
    var tools: [String]? = nil
    /// What the tool router offers this group for (ToolRouter).
    let router: String
    /// Ask before every call (tools that read files by path).
    var ask = false
    /// Seconds to wait for the first start: npx / uvx download the package, ko-stdict builds its dictionary.
    var startTimeout: Double = 180
    /// Appended to each of its tools' descriptions (what a small model gets wrong about the arguments).
    var hint: String? = nil
    /// Repairs arguments a small model reliably gets wrong; returns the note to show with the result.
    var fix: (@Sendable ([String: Any]) -> ([String: Any], String?))? = nil

    /// Coordinates in Korea: east longitude 124–132, latitude 33–39. A west (negative) longitude or swapped
    /// lat / lon is read the way it must have been meant, and the result says so.
    static let koreanCoordinates: @Sendable ([String: Any]) -> ([String: Any], String?) = { args in
        func num(_ v: Any?) -> Double? { (v as? NSNumber)?.doubleValue ?? (v as? String).flatMap(Double.init) }
        guard var lon = num(args["lon"]), var lat = num(args["lat"]) else { return (args, nil) }
        var notes: [String] = []
        if (124...132).contains(lat), (33...39).contains(lon) {
            swap(&lat, &lon)
            notes.append("lat and lon were swapped")
        }
        if (-132 ... -124).contains(lon) {
            lon = -lon
            notes.append("lon \(-lon) was read as \(lon) (Korea is east of Greenwich)")
        }
        guard !notes.isEmpty else { return (args, nil) }
        var out = args
        out["lon"] = lon
        out["lat"] = lat
        return (out, "(Note: " + notes.joined(separator: "; ") + ".)")
    }

    static let all: [MCPCatalogEntry] = [
        MCPCatalogEntry(
            id: "dict", name: "표준국어대사전",
            summary: "Looks up Korean words in the Standard Korean Dictionary, offline. No key. The first start downloads the dictionary and builds a local database, which takes a few minutes.",
            repo: "https://github.com/dahlia/ko-stdict-mcp",
            // the package root (mod.ts) is the library and exits at once; ./main is the stdio server
            command: "npx -y deno@2 run -A jsr:@hongminhee/ko-stdict-mcp@0.2.0/main", needs: "npx",
            tools: ["search_entries", "get_entry"],
            router: "the meaning, spelling, hanja or usage of a Korean word (표준국어대사전)", startTimeout: 900),
        MCPCatalogEntry(
            id: "spell", name: "맞춤법 검사",
            summary: "Corrects Korean spelling, spacing and grammar (uses Naver's spell checker). No key.",
            repo: "https://github.com/winterjung/mcp-korean-spell",
            command: "npx -y @winterjung/mcp-korean-spell@1.0.1", needs: "npx",
            tools: ["fix_korean_spell"],
            router: "checking or correcting Korean spelling, spacing or grammar (맞춤법)"),
        MCPCatalogEntry(
            id: "doc", name: "한글·PDF 문서 (kordoc)",
            summary: "Reads HWP, HWPX, PDF, XLSX and DOCX files as Markdown, extracts tables and compares two documents. No key. Asks before every call, since it opens files by path.",
            repo: "https://github.com/chrisryugj/kordoc",
            command: "npx -y -p kordoc@4.24.80 kordoc-mcp", needs: "npx",
            tools: ["parse_document", "parse_table", "compare_documents"],
            router: "reading or comparing a Korean document file (HWP, HWPX, PDF, XLSX, DOCX)", ask: true),
        MCPCatalogEntry(
            id: "law", name: "법령·판례 (국가법령정보)",
            summary: "Searches Korean laws, ordinances, administrative rules and court decisions through the 국가법령정보 Open API.",
            repo: "https://github.com/chrisryugj/korean-law-mcp",
            command: "npx -y korean-law-mcp@4.15.23", needs: "npx",
            keys: [Key(env: "LAW_OC", label: "국가법령정보 OC (your open.law.go.kr ID)", signup: "https://open.law.go.kr/LSO/openApi/guideList.do")],
            tools: ["search_law", "get_law_text", "get_annexes", "search_decisions", "get_decision_text"],
            router: "Korean laws, regulations, ordinances or court decisions (법령, 판례)"),
        MCPCatalogEntry(
            id: "biz", name: "사업자등록 상태조회 (국세청)",
            summary: "Checks whether a Korean business registration number is valid and active (data.go.kr).",
            repo: "https://github.com/Koomook/data-go-mcp-servers",
            command: "uvx --with 'mcp<2' data-go-mcp.nts-business-verification==0.2.0", needs: "uv",
            keys: [Key(env: "API_KEY", label: "공공데이터포털 인증키 (국세청 사업자등록정보 진위확인 및 상태조회 서비스)", signup: "https://www.data.go.kr/data/15081808/openapi.do")],
            tools: ["check_business_status", "validate_business"],
            router: "whether a Korean business registration number (사업자등록번호) is valid or active"),
        MCPCatalogEntry(
            id: "weather", name: "날씨 (기상청 단기예보)",
            summary: "Current weather and short-term forecasts from the Korea Meteorological Administration, by coordinates (Kakao Map can find them).",
            repo: "https://github.com/ohhan777/korea_weather",
            command: "uv run --no-project --with 'git+https://github.com/ohhan777/korea_weather@d8d6030f4d050b8378188ac0266f6179ddb4d3f9' --with 'mcp<2' python -m korea_weather",
            needs: "uv",
            keys: [Key(env: "KOREA_WEATHER_API_KEY", label: "공공데이터포털 인증키 (기상청 단기예보 조회서비스)", signup: "https://www.data.go.kr/data/15084084/openapi.do")],
            router: "the current weather or a weather forecast in Korea (기상청)",
            hint: "lat/lon in decimal degrees; Korea is lat 33–39, lon 124–132 (east, positive). Seoul: lat 37.5665, "
                + "lon 126.978; Busan: lat 35.1796, lon 129.0756. For another place, get its coordinates first.",
            fix: MCPCatalogEntry.koreanCoordinates),
        MCPCatalogEntry(
            id: "kakao", name: "카카오맵 장소·주소",
            summary: "Finds places, addresses and coordinates in Korea with the Kakao Local API.",
            repo: "https://github.com/cgoinglove/mcp-server-kakao-map",
            command: "npx -y mcp-server-kakao-map@0.3.2", needs: "npx",
            keys: [Key(env: "KAKAO_API_KEY", label: "카카오 REST API 키", signup: "https://developers.kakao.com/console/app")],
            tools: ["kakao_local_search_keyword", "kakao_local_search_address", "kakao_local_geo_coord2address"],
            router: "finding a place, an address or a location's coordinates in Korea (카카오맵)"),
        MCPCatalogEntry(
            id: "krx", name: "코스피·주가 (KRX)",
            summary: "KOSPI / KOSDAQ index levels and stock prices by company name, daily, from KRX through pykrx. Our own two-tool server (KRXServerScript.swift): pykrx-mcp has no index tool and wants tickers. KRX's data service needs a free login.",
            repo: "https://github.com/sharebook-kr/pykrx",
            command: "uv run --no-project --with 'mcp<2' --with 'pykrx==1.2.9' python {krx_server}", needs: "uv",
            keys: [Key(env: "KRX_ID", label: "KRX 정보데이터시스템 ID", signup: "https://data.krx.co.kr"),
                   Key(env: "KRX_PW", label: "KRX 정보데이터시스템 비밀번호", signup: "https://data.krx.co.kr")],
            router: "the KOSPI or KOSDAQ index, or a Korean stock's price (코스피, 코스닥, 주가)"),
        MCPCatalogEntry(
            id: "dart", name: "공시·재무제표 (DART)",
            summary: "Company disclosures and financial statements from OpenDART, plus KRX prices.",
            repo: "https://github.com/jjlabsio/korea-stock-mcp",
            command: "npx -y korea-stock-mcp@1.4.1", needs: "npx",
            keys: [Key(env: "DART_API_KEY", label: "OpenDART 인증키", signup: "https://opendart.fss.or.kr/uss/umt/EgovMberInsertView.do"),
                   Key(env: "KRX_API_KEY", label: "KRX Open API 인증키 (optional)", signup: "https://openapi.krx.co.kr", optional: true)],
            router: "Korean company disclosures or financial statements (DART 공시, 재무제표)"),
    ]
}

/// API keys for the MCP servers, in the login Keychain (never in the chats, settings or logs).
enum MCPKeychain {
    static let service = "Zoro-1 Chat MCP"

    static func get(_ account: String) -> String? {
        let q: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
                                kSecAttrAccount as String: account, kSecReturnData as String: true,
                                kSecMatchLimit as String: kSecMatchLimitOne]
        var out: CFTypeRef?
        guard SecItemCopyMatching(q as CFDictionary, &out) == errSecSuccess, let d = out as? Data else { return nil }
        return String(data: d, encoding: .utf8)
    }

    static func set(_ account: String, _ value: String) {
        let q: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service,
                                kSecAttrAccount as String: account]
        SecItemDelete(q as CFDictionary)
        guard !value.isEmpty else { return }
        var add = q
        add[kSecValueData as String] = Data(value.utf8)
        SecItemAdd(add as CFDictionary, nil)
    }
}

/// Runs the enabled servers and turns their tools into the agent's tools.
@MainActor
final class MCPHub: ObservableObject {
    enum Status: Equatable {
        case off, starting, ready(Int), failed(String)
    }

    @Published private(set) var status: [String: Status] = [:]
    @Published private(set) var enabled: Set<String> =
        Set(UserDefaults.standard.stringArray(forKey: "TridaChat.mcpEnabled") ?? []) {
        didSet { UserDefaults.standard.set(Array(enabled).sorted(), forKey: "TridaChat.mcpEnabled") }
    }
    private var clients: [String: MCPClient] = [:]
    private var tools: [String: [MCPClient.Tool]] = [:]
    private var tasks: [String: Task<Void, Never>] = [:]

    static func entry(_ id: String) -> MCPCatalogEntry? { MCPCatalogEntry.all.first { $0.id == id } }

    func key(_ entry: MCPCatalogEntry, _ k: MCPCatalogEntry.Key) -> String {
        MCPKeychain.get("\(entry.id).\(k.env)") ?? ""
    }

    func setKey(_ entry: MCPCatalogEntry, _ k: MCPCatalogEntry.Key, _ value: String) {
        MCPKeychain.set("\(entry.id).\(k.env)", value.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    func setEnabled(_ id: String, _ on: Bool) {
        if on { enabled.insert(id) } else { enabled.remove(id) }
        if on { start(id) } else { stop(id) }
    }

    func startEnabled() {
        for id in enabled { start(id) }
    }

    func start(_ id: String) {
        guard let entry = Self.entry(id) else { return }
        stop(id)
        if let missing = entry.keys.first(where: { !$0.optional && key(entry, $0).isEmpty }) {
            status[id] = .failed("Enter \(missing.label) first.")
            return
        }
        var env: [String: String] = [:]
        for k in entry.keys {
            let v = key(entry, k)
            if !v.isEmpty { env[k.env] = v }
        }
        var command = entry.command
        if command.contains("{krx_server}") {   // our own server: write the script out first
            let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("Zoro-1 Chat/mcp", isDirectory: true)
            let file = dir.appendingPathComponent("krx_server.py")
            do {
                try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
                try KRXServerScript.source.write(to: file, atomically: true, encoding: .utf8)
            } catch {
                status[id] = .failed("Could not write \(file.path): \(error.localizedDescription)")
                return
            }
            let quoted = "'" + file.path.replacingOccurrences(of: "'", with: "'\\''") + "'"
            command = command.replacingOccurrences(of: "{krx_server}", with: quoted)
        }
        let client = MCPClient()
        clients[id] = client
        status[id] = .starting
        tasks[id] = Task { [weak self] in
            do {
                try client.start(command: command, environment: env,
                                 cwd: FileManager.default.homeDirectoryForCurrentUser)
                _ = try await client.initialize(timeout: entry.startTimeout)
                let all = try await client.listTools()
                let offered = entry.tools.map { names in all.filter { names.contains($0.name) } } ?? all
                guard !Task.isCancelled else { return }
                self?.tools[id] = offered
                self?.status[id] = offered.isEmpty ? .failed("The server lists none of the expected tools.")
                    : .ready(offered.count)
            } catch {
                guard !Task.isCancelled else { return }
                let tail = client.stderrTail.split(separator: "\n").suffix(3).joined(separator: "\n")
                var msg = error.localizedDescription
                if tail.contains("command not found") || tail.contains("not found: \(entry.needs)") {
                    msg = entry.needs == "uv" ? "uv is not installed (brew install uv)."
                        : "Node.js is not installed (brew install node)."
                } else if !tail.isEmpty {
                    msg += "\n" + tail
                }
                self?.status[id] = .failed(msg)
                client.stop()
            }
        }
    }

    func stop(_ id: String) {
        tasks[id]?.cancel()
        tasks[id] = nil
        clients[id]?.stop()
        clients[id] = nil
        tools[id] = nil
        status[id] = .off
    }

    func stopAll() {
        for c in clients.values { c.stop() }
    }

    /// The model's name for a server's tool: the server's id as a prefix, unless the tool already has it.
    static func exposedName(_ entry: MCPCatalogEntry, _ tool: String) -> String {
        tool.hasPrefix(entry.id + "_") ? tool : "\(entry.id)_\(tool)"
    }

    /// OpenAI tool specs for every ready server's offered tools.
    var specs: [[String: Any]] {
        MCPCatalogEntry.all.flatMap { entry -> [[String: Any]] in
            guard case .ready = status[entry.id] else { return [] }
            return (tools[entry.id] ?? []).map { t in
                var desc = t.description
                if desc.count > 500 { desc = String(desc.prefix(500)) + "…" }
                if let hint = entry.hint { desc += " " + hint }
                return ["type": "function",
                        "function": ["name": Self.exposedName(entry, t.name),
                                     "description": "[\(entry.name)] " + desc,
                                     "parameters": t.schemaObject] as [String: Any]]
            }
        }
    }

    /// One router group per ready server.
    var routerGroups: [ToolRouter.Group] {
        MCPCatalogEntry.all.compactMap { entry in
            guard case .ready = status[entry.id] else { return nil }
            // `now` comes along: "tomorrow's weather", "last week's price" need today's date
            return ToolRouter.Group(key: "mcp_" + entry.id, description: entry.router,
                                    tools: (tools[entry.id] ?? []).map { Self.exposedName(entry, $0.name) } + ["now"])
        }
    }

    var readyNames: [String] {
        MCPCatalogEntry.all.filter { if case .ready = status[$0.id] { return true }; return false }.map(\.name)
    }

    /// The server and tool behind a name the model called, if it is one of ours.
    func lookup(_ name: String) -> (entry: MCPCatalogEntry, tool: String)? {
        for entry in MCPCatalogEntry.all {
            guard case .ready = status[entry.id] else { continue }
            if let t = tools[entry.id]?.first(where: { Self.exposedName(entry, $0.name) == name }) {
                return (entry, t.name)
            }
        }
        return nil
    }

    func call(_ entry: MCPCatalogEntry, _ tool: String, arguments: [String: Any]) async -> String {
        guard let client = clients[entry.id], client.isRunning else {
            return "error: the \(entry.name) server is not running"
        }
        var arguments = arguments
        var note: String? = nil
        if let fix = entry.fix { (arguments, note) = fix(arguments) }
        do {
            let r = try await client.callTool(tool, arguments: arguments, timeout: 120)
            var text = r.text.isEmpty ? "(no result)" : r.text
            if let note { text = note + "\n" + text }
            if text.count > 8000 { text = String(text.prefix(8000)) + "\n… (cut at 8,000 characters)" }
            return r.isError ? "error: " + text : text
        } catch {
            return "error: " + error.localizedDescription
        }
    }
}

struct KoreanMCPTab: View {
    @EnvironmentObject var state: AppState
    @ObservedObject var hub: MCPHub

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Text("Korean services for the agent, from awesome-mcp-korea. Each is a third-party program that runs on this Mac with your permissions and its own network access; it starts only when you switch it on. Versions are pinned. Needs Node.js (npx) or uv, as listed. Keys stay in your Keychain.")
                    .font(.caption).foregroundStyle(.secondary)
                ForEach(MCPCatalogEntry.all) { entry in
                    MCPEntryRow(entry: entry, hub: hub)
                    Divider()
                }
            }
            .padding(.vertical, 4)
        }
        .frame(height: 460)
    }
}

private struct MCPEntryRow: View {
    let entry: MCPCatalogEntry
    @ObservedObject var hub: MCPHub
    @State private var keys: [String: String] = [:]

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Toggle(isOn: Binding(get: { hub.enabled.contains(entry.id) },
                                     set: { hub.setEnabled(entry.id, $0) })) {
                    Text(entry.name).fontWeight(.semibold)
                }
                Spacer()
                statusView
                Link("GitHub", destination: URL(string: entry.repo)!).font(.caption)
            }
            Text(entry.summary + " Needs \(entry.needs == "uv" ? "uv" : "Node.js").")
                .font(.caption).foregroundStyle(.secondary)
            ForEach(entry.keys, id: \.env) { k in
                HStack {
                    SecureField(k.label, text: Binding(get: { keys[k.env] ?? hub.key(entry, k) },
                                                      set: { keys[k.env] = $0 }))
                        .textFieldStyle(.roundedBorder)
                        .onSubmit { save(k) }
                    Button("Save") { save(k) }
                    Link("Get a key", destination: URL(string: k.signup)!).font(.caption)
                }
            }
            if case let .failed(msg) = hub.status[entry.id] {
                Text(msg).font(.caption).foregroundStyle(.orange).lineLimit(4).textSelection(.enabled)
            }
        }
    }

    @ViewBuilder private var statusView: some View {
        switch hub.status[entry.id] ?? .off {
        case .off: EmptyView()
        case .starting: HStack(spacing: 4) { ProgressView().controlSize(.small); Text("starting").font(.caption) }
        case let .ready(n): Text("ready · \(n) tool\(n == 1 ? "" : "s")").font(.caption).foregroundStyle(.green)
        case .failed:
            Button("Retry") { hub.start(entry.id) }.font(.caption)
        }
    }

    private func save(_ k: MCPCatalogEntry.Key) {
        hub.setKey(entry, k, keys[k.env] ?? hub.key(entry, k))
        keys[k.env] = nil
        if hub.enabled.contains(entry.id) { hub.start(entry.id) }
    }
}
