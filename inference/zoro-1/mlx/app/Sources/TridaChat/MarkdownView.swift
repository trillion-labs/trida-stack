import AppKit
import SwiftUI
import TridaChatCore

/// A reply rendered from Markdown (Markdown.parse in TridaChatCore): headings, paragraphs, nested and task
/// lists, quotes, code blocks with their language and a copy button, tables, rules and display math (shown as
/// monospaced LaTeX). Inline bold, italics, code and links come from AttributedString. Each block is
/// selectable; the copy button under a reply copies the raw Markdown.
struct MarkdownText: View {
    let blocks: [MDBlock]

    init(_ source: String) { blocks = Markdown.parse(source) }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(Array(blocks.enumerated()), id: \.offset) { _, b in
                block(b)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private func block(_ b: MDBlock) -> some View {
        switch b {
        case let .heading(level, text):
            Text(Self.inline(text))
                .font(.system(size: level == 1 ? 22 : level == 2 ? 18 : 15,
                              weight: level <= 2 ? .semibold : .bold, design: level <= 2 ? .serif : .default))
                .foregroundStyle(Theme.text)
                .padding(.top, level <= 2 ? 6 : 2)
                .textSelection(.enabled)
        case let .paragraph(text):
            Text(Self.inline(text))
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        case let .list(items):
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                    HStack(alignment: .firstTextBaseline, spacing: 6) {
                        if let c = item.checked {
                            Image(systemName: c ? "checkmark.square" : "square").foregroundStyle(.secondary)
                        }
                        if !item.marker.isEmpty {
                            Text(item.marker).foregroundStyle(.secondary).monospacedDigit()
                        }
                        Text(Self.inline(item.text))
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .padding(.leading, CGFloat(item.depth) * 18)
                }
            }
        case let .quote(text):
            HStack(alignment: .top, spacing: 10) {
                RoundedRectangle(cornerRadius: 1.5).fill(Theme.accent.opacity(0.7)).frame(width: 3)
                Text(Self.inline(text))
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .fixedSize(horizontal: false, vertical: true)
            }
        case let .code(language, text):
            CodeBlockView(language: language, code: text)
        case let .table(header, alignments, rows):
            TableBlockView(header: header, alignments: alignments, rows: rows)
        case .rule:
            Divider().padding(.vertical, 2)
        case let .math(tex):
            ScrollView(.horizontal) {
                Text(tex)
                    .font(.system(.body, design: .monospaced))
                    .textSelection(.enabled)
                    .padding(.vertical, 6)
                    .padding(.horizontal, 12)
            }
            .frame(maxWidth: .infinity)
            .card(radius: 8)
            .help("LaTeX (shown as source)")
        }
    }

    static func inline(_ s: String) -> AttributedString {
        let opts = AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        return (try? AttributedString(markdown: s, options: opts)) ?? AttributedString(s)
    }
}

/// (No "Copied" confirmation: that needs @State, a macro whose plugin only Xcode ships; see AppState.)
struct CodeBlockView: View {
    let language: String
    let code: String

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                Text(language.isEmpty ? "code" : language).font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(code, forType: .string)
                } label: {
                    Label("Copy", systemImage: "doc.on.doc").font(.caption)
                }
                .buttonStyle(.borderless)
                .help("Copy this code")
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 5)
            .background(Theme.surfaceRaised)
            ScrollView(.horizontal) {
                Text(code)
                    .font(.system(size: 12, design: .monospaced))
                    .textSelection(.enabled)
                    .padding(10)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Theme.surface)
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).stroke(Theme.border))
    }
}

struct TableBlockView: View {
    let header: [String]
    let alignments: [MDAlignment]
    let rows: [[String]]

    private func align(_ c: Int) -> HorizontalAlignment {
        switch c < alignments.count ? alignments[c] : .leading {
        case .leading: return .leading
        case .center: return .center
        case .trailing: return .trailing
        }
    }

    var body: some View {
        ScrollView(.horizontal) {
            Grid(alignment: .leading, horizontalSpacing: 0, verticalSpacing: 0) {
                // (a GridRow takes no modifiers of its own, so the shading goes on each cell)
                GridRow {
                    ForEach(Array(header.enumerated()), id: \.offset) { c, h in
                        cell(h, column: c, fill: Theme.surfaceRaised, bold: true)
                    }
                }
                ForEach(Array(rows.enumerated()), id: \.offset) { r, row in
                    Divider().gridCellUnsizedAxes(.horizontal)
                    GridRow {
                        ForEach(Array(row.enumerated()), id: \.offset) { c, v in
                            cell(v, column: c, fill: r % 2 == 1 ? Theme.canvas : Theme.surface, bold: false)
                        }
                    }
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 8, style: .continuous).stroke(Theme.border))
        }
    }

    private func cell(_ s: String, column c: Int, fill: Color, bold: Bool) -> some View {
        Text(MarkdownText.inline(s))
            .fontWeight(bold ? .semibold : nil)
            .textSelection(.enabled)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .frame(minWidth: 40, maxWidth: 360, alignment: Alignment(horizontal: align(c), vertical: .center))
            .gridColumnAlignment(align(c))
            .fixedSize(horizontal: false, vertical: true)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: Alignment(horizontal: align(c), vertical: .center))
            .background(fill)
    }
}
