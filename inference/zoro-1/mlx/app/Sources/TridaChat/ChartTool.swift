import AppKit
import Charts
import SwiftUI
import TridaChatCore
import UniformTypeIdentifiers

/// `plot_chart`: the model gives the numbers, the app draws the chart (Swift Charts) and shows it in the chat.
/// A 4B model writing matplotlib code fails in ways a schema can't (missing packages, wrong APIs, bad indexing);
/// filling in a few arrays it gets right.
enum ChartTool {
    static let names: Set<String> = ["plot_chart"]

    static let spec: [String: Any] = [
        "type": "function",
        "function": [
            "name": "plot_chart",
            "description": "Draw a line or bar chart of numbers already in the conversation and show it to the user. "
                + "Give the x labels (dates, names) and one or more series with one value per x label. Use this "
                + "instead of writing plotting code.",
            "parameters": [
                "type": "object",
                "properties": [
                    "title": ["type": "string", "description": "the chart's title, in the user's language"],
                    "kind": ["type": "string", "enum": ["line", "bar"],
                             "description": "line for values over time, bar to compare items"],
                    "x": ["type": "array", "items": ["type": "string"],
                          "description": "x-axis labels in order, e.g. [\"11-11\", \"11-12\", \"11-13\"]"],
                    "series": [
                        "type": "array",
                        "items": ["type": "object",
                                  "properties": ["name": ["type": "string"],
                                                 "values": ["type": "array", "items": ["type": "number"]]],
                                  "required": ["name", "values"]],
                        "description": "one or more series, each with one value per x label",
                    ],
                    "y_label": ["type": "string", "description": "what the values are, with units"],
                ] as [String: Any],
                "required": ["kind", "x", "series"],
            ] as [String: Any],
        ] as [String: Any],
    ]

    struct Series: Identifiable {
        let name: String
        let values: [Double]
        var id: String { name }
    }

    struct Spec {
        let title: String
        let kind: String
        let x: [String]
        let series: [Series]
        let yLabel: String
    }

    /// The arguments as a chart, or a message saying exactly what to fix.
    static func parse(_ a: [String: Any]) -> Result<Spec, Error> {
        struct Bad: LocalizedError { let errorDescription: String? }
        func num(_ v: Any) -> Double? { (v as? NSNumber)?.doubleValue ?? (v as? String).flatMap { Double($0.replacingOccurrences(of: ",", with: "")) } }
        let x = (a["x"] as? [Any] ?? []).map { "\($0)" }
        guard !x.isEmpty, x.count <= 400 else { return .failure(Bad(errorDescription: "x needs 1 to 400 labels")) }
        var series: [Series] = []
        for (i, s) in (a["series"] as? [[String: Any]] ?? []).prefix(8).enumerated() {
            let raw = s["values"] as? [Any] ?? []
            let values = raw.compactMap(num)
            let name = (s["name"] as? String).flatMap { $0.isEmpty ? nil : $0 } ?? "series \(i + 1)"
            guard values.count == raw.count else {
                return .failure(Bad(errorDescription: "series \"\(name)\" has values that are not numbers"))
            }
            guard values.count == x.count else {
                return .failure(Bad(errorDescription: "series \"\(name)\" has \(values.count) values but x has "
                                    + "\(x.count) labels: give one value per x label"))
            }
            series.append(Series(name: name, values: values))
        }
        guard !series.isEmpty else { return .failure(Bad(errorDescription: "give at least one series")) }
        let kind = (a["kind"] as? String) == "bar" ? "bar" : "line"
        return .success(Spec(title: a["title"] as? String ?? "", kind: kind, x: x, series: series,
                             yLabel: a["y_label"] as? String ?? ""))
    }

    /// PNG of the chart (light, 2x).
    @MainActor
    static func render(_ spec: Spec) -> Data? {
        let r = ImageRenderer(content: ChartImage(spec: spec).environment(\.colorScheme, .light))
        r.scale = 2
        guard let image = r.nsImage, let tiff = image.tiffRepresentation,
              let rep = NSBitmapImageRep(data: tiff) else { return nil }
        return rep.representation(using: .png, properties: [:])
    }

    /// The chart itself: the PNG for the model and for saving, and the live card in the chat (hover for values).
    struct ChartBody: View {
        let spec: Spec
        var interactive = false
        @State private var hovered: String?

        private var yName: String { spec.yLabel.isEmpty ? "value" : spec.yLabel }

        var body: some View {
            VStack(alignment: .leading, spacing: 10) {
                if !spec.title.isEmpty { Text(spec.title).font(.system(size: 15, weight: .semibold)) }
                Chart {
                    ForEach(spec.series) { s in
                        ForEach(Array(zip(spec.x, s.values).enumerated()), id: \.offset) { _, p in
                            if spec.kind == "bar" {
                                BarMark(x: .value("x", p.0), y: .value(yName, p.1))
                                    .foregroundStyle(by: .value("series", s.name))
                                    .position(by: .value("series", s.name))
                            } else {
                                LineMark(x: .value("x", p.0), y: .value(yName, p.1))
                                    .foregroundStyle(by: .value("series", s.name))
                                    .symbol(by: .value("series", s.name))
                            }
                        }
                    }
                    if interactive, let hovered, let i = spec.x.firstIndex(of: hovered) {
                        RuleMark(x: .value("x", hovered))
                            .foregroundStyle(.secondary.opacity(0.4))
                            .annotation(position: .top, overflowResolution: .init(x: .fit, y: .disabled)) {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(hovered).font(.caption.bold())
                                    ForEach(spec.series) { s in
                                        Text((spec.series.count > 1 ? s.name + ": " : "")
                                             + s.values[i].formatted(.number.precision(.fractionLength(0...2))))
                                            .font(.caption.monospacedDigit())
                                    }
                                }
                                .padding(6)
                                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 6))
                            }
                    }
                }
                .chartYScale(domain: .automatic(includesZero: spec.kind == "bar"))
                .chartYAxisLabel(spec.yLabel)
                .chartLegend(spec.series.count > 1 ? .visible : .hidden)
                .chartXSelection(value: $hovered)
            }
        }
    }

    struct ChartImage: View {
        let spec: Spec

        var body: some View {
            ChartBody(spec: spec)
                .padding(20)
                .frame(width: 760, height: 440)
                .background(Color.white)
        }
    }
}

extension AppState {
    /// Runs plot_chart: draws it, saves the PNG in the chat's scratch folder, and returns the text and image.
    func runChartTool(_ call: ToolCall, chatID: Chat.ID) -> ToolOutput {
        switch ChartTool.parse(call.argumentDict) {
        case let .failure(e):
            return ToolOutput(text: "error: " + e.localizedDescription, images: [])
        case let .success(spec):
            guard let png = ChartTool.render(spec) else {
                return ToolOutput(text: "error: the chart could not be drawn", images: [])
            }
            let dir = Scratch.dir(chatID)
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let file = dir.appendingPathComponent("chart-\(Int(Date().timeIntervalSince1970)).png")
            try? png.write(to: file)
            let n = spec.series.map(\.values.count).reduce(0, +)
            return ToolOutput(text: "Drew a \(spec.kind) chart (\(spec.series.count) series, \(n) points); the user "
                              + "sees it under this tool call. Saved as \(file.path). Don't describe how it was "
                              + "drawn; tell the user what the chart shows.", images: [png])
        }
    }
}

/// A plot_chart result in the chat: the chart drawn live (hover a point for its values), with Copy, Save and Open.
struct ChartCard: View {
    let run: ToolRun

    var body: some View {
        if case let .success(spec) = ChartTool.parse(run.call.argumentDict) {
            VStack(alignment: .leading, spacing: 8) {
                ChartTool.ChartBody(spec: spec, interactive: true)
                    .frame(height: 300)
                if let png = run.images.first {
                    HStack(spacing: 14) {
                        Spacer()
                        Button { copy(png) } label: { Label("Copy", systemImage: "doc.on.doc") }
                        Button { save(png, spec.title) } label: { Label("Save…", systemImage: "square.and.arrow.down") }
                        Button { open(png) } label: { Label("Open", systemImage: "arrow.up.forward.app") }
                    }
                    .buttonStyle(.borderless)
                    .font(.caption)
                }
            }
            .padding(14)
            .card(radius: 10, fill: Theme.surfaceRaised)
        }
    }

    private func copy(_ png: Data) {
        guard let img = NSImage(data: png) else { return }
        NSPasteboard.general.clearContents()
        NSPasteboard.general.writeObjects([img])
    }

    private func save(_ png: Data, _ title: String) {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.png]
        let name = title.isEmpty ? "chart" : title.replacingOccurrences(of: "/", with: "-")
        panel.nameFieldStringValue = name + ".png"
        if panel.runModal() == .OK, let url = panel.url { try? png.write(to: url) }
    }

    private func open(_ png: Data) {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("chart-\(run.id).png")
        try? png.write(to: url)
        NSWorkspace.shared.open(url)
    }
}
