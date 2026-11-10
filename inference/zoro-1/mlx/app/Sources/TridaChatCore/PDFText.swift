import Foundation
#if canImport(PDFKit)
import AppKit
import PDFKit
#endif

/// The text of a PDF's pages (PDFKit), for read_pdf and read_file on a .pdf. Pages with almost no text
/// (scans, figures) are rendered as images instead when the model can see them.
public enum PDFText {
    public struct Result: Sendable {
        public var text: String
        public var images: [Data]
    }

    /// "3", "1-5", "2,4-6" -> 1-based page numbers within 1...count (at most 50).
    public static func pageNumbers(_ spec: String?, count: Int) -> [Int] {
        guard count > 0 else { return [] }
        guard let spec, !spec.trimmingCharacters(in: .whitespaces).isEmpty else { return Array(1...min(5, count)) }
        var out: [Int] = []
        for part in spec.split(separator: ",") {
            let b = part.split(separator: "-").map { Int($0.trimmingCharacters(in: .whitespaces)) }
            if b.count == 1, let n = b[0] { out.append(n) }
            if b.count == 2, let x = b[0], let y = b[1], x <= y { out += Array(x...y) }
        }
        return Array(out.filter { (1...count).contains($0) }.prefix(50))
    }

    public static func read(_ url: URL, pages spec: String?, maxChars: Int, renderScans: Bool = false) -> Result {
        #if canImport(PDFKit)
        guard let doc = PDFDocument(url: url) else { return Result(text: "error: not a PDF macOS can read", images: []) }
        if doc.isLocked { return Result(text: "error: the PDF is password-protected", images: []) }
        let numbers = pageNumbers(spec, count: doc.pageCount)
        guard !numbers.isEmpty else {
            return Result(text: "error: no such pages (the PDF has \(doc.pageCount))", images: [])
        }
        var text = "\(url.lastPathComponent): \(doc.pageCount) pages\n"
        var images: [Data] = []
        for n in numbers {
            guard let page = doc.page(at: n - 1) else { continue }
            let t = (page.string ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            if t.count < 40, renderScans, images.count < 4, let png = render(page) {
                images.append(png)
                text += "\n--- page \(n) (no text layer: shown as an image) ---\n"
            } else {
                text += "\n--- page \(n) ---\n" + (t.isEmpty ? "(no text)" : t) + "\n"
            }
            if text.count > maxChars {
                text = String(text.prefix(maxChars)) + "\n… (cut at \(maxChars) characters; ask for fewer pages)"
                break
            }
        }
        return Result(text: text, images: images)
        #else
        return Result(text: "error: PDF reading needs macOS", images: [])
        #endif
    }

    #if canImport(PDFKit)
    /// A page as PNG, about 1500 px on its longest side.
    static func render(_ page: PDFPage) -> Data? {
        let box = page.bounds(for: .mediaBox)
        let scale = 1500 / max(box.width, box.height, 1)
        let img = page.thumbnail(of: CGSize(width: box.width * scale, height: box.height * scale), for: .mediaBox)
        guard let tiff = img.tiffRepresentation, let rep = NSBitmapImageRep(data: tiff) else { return nil }
        return rep.representation(using: .png, properties: [:])
    }
    #endif
}
