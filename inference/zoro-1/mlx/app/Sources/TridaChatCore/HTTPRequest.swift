import Foundation

/// One HTTP/1.1 request as the app's local automation API reads it (Automation.swift). It is parsed before the
/// token is checked, so anything a local process sends must fail cleanly here, never trap.
public struct HTTPRequest {
    public var method: String
    public var path: String
    public var headers: [String: String]
    public var body: Data

    /// Bodies are small JSON objects; anything larger is refused.
    public static let maxBody = 4 << 20

    public enum Parsed {
        case needMore
        case bad
        case ok(HTTPRequest)
    }

    /// A complete request, `.needMore` while bytes are still missing, `.bad` for a request that can never be
    /// valid (malformed request line, negative or oversized Content-Length).
    public static func parse(_ d: Data) -> Parsed {
        guard let end = d.range(of: Data("\r\n\r\n".utf8)) else { return .needMore }
        let head = String(decoding: d[d.startIndex..<end.lowerBound], as: UTF8.self)
            .split(separator: "\r\n").map(String.init)
        guard let first = head.first else { return .bad }
        let parts = first.split(separator: " ").map(String.init)
        guard parts.count >= 2 else { return .bad }
        var h: [String: String] = [:]
        for line in head.dropFirst() {
            if let i = line.firstIndex(of: ":") {
                h[line[..<i].lowercased()] = line[line.index(after: i)...].trimmingCharacters(in: .whitespaces)
            }
        }
        guard let n = Int(h["content-length"] ?? "0"), (0...maxBody).contains(n) else { return .bad }
        let start = end.upperBound
        guard d.endIndex - start >= n else { return .needMore }
        return .ok(HTTPRequest(method: parts[0], path: parts[1], headers: h, body: Data(d[start..<(start + n)])))
    }

    public var json: [String: Any] { (try? JSONSerialization.jsonObject(with: body) as? [String: Any]) ?? [:] }

    /// Whether the Authorization header is `Bearer <token>`, compared in constant time.
    public func hasBearer(_ token: String) -> Bool {
        let got = Array((headers["authorization"] ?? "").utf8), want = Array("Bearer \(token)".utf8)
        var diff = UInt8(got.count == want.count ? 0 : 1)
        for i in 0..<want.count { diff |= want[i] ^ (i < got.count ? got[i] : 0) }
        return diff == 0
    }
}
