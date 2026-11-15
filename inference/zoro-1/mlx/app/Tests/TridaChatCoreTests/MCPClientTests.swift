import XCTest
@testable import TridaChatCore

/// MCPClient against a tiny stdio MCP server written in Python (macOS ships /usr/bin/python3): the handshake, paged
/// tools/list, tools/call (text, isError), a server-side ping, junk on stdout, and a server that exits.
final class MCPClientTests: XCTestCase {
    static let server = #"""
import json, sys
print("profile noise on stdout", flush=True)
def send(o): print(json.dumps(o), flush=True)
for line in sys.stdin:
    m = json.loads(line)
    i, meth = m.get("id"), m.get("method")
    if meth == "initialize":
        send({"jsonrpc": "2.0", "id": 999, "method": "ping"})
        send({"jsonrpc": "2.0", "id": i, "result": {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
              "serverInfo": {"name": "fake", "version": "1.0"}}})
    elif meth == "tools/list":
        page = (m.get("params") or {}).get("cursor")
        tools = [{"name": "echo", "description": " Echo back. ", "inputSchema": {"type": "object",
                  "properties": {"text": {"type": "string"}}, "required": ["text"]}}]
        if page:
            send({"jsonrpc": "2.0", "id": i, "result": {"tools": [{"name": "fail", "inputSchema": {"type": "object"}}]}})
        else:
            send({"jsonrpc": "2.0", "id": i, "result": {"tools": tools, "nextCursor": "p2"}})
    elif meth == "tools/call":
        p = m["params"]
        if p["name"] == "echo":
            send({"jsonrpc": "2.0", "id": i, "result": {"content": [{"type": "text", "text": "echo: " + p["arguments"]["text"]}]}})
        elif p["name"] == "quit":
            sys.exit(3)
        else:
            send({"jsonrpc": "2.0", "id": i, "result": {"content": [{"type": "text", "text": "bad"}], "isError": True}})
    elif i is not None and meth is None:
        pass   # our answer to the server's ping
"""#

    func testHandshakeToolsAndCalls() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("mcp-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let script = dir.appendingPathComponent("server.py")
        try Self.server.write(to: script, atomically: true, encoding: .utf8)
        let client = MCPClient()
        try client.start(command: "/usr/bin/python3 -u '\(script.path)'", environment: [:], cwd: dir)
        defer { client.stop() }

        let info = try await client.initialize(timeout: 30)
        XCTAssertEqual(info, "fake 1.0")
        let tools = try await client.listTools(timeout: 10)
        XCTAssertEqual(tools.map(\.name), ["echo", "fail"])          // both pages
        XCTAssertEqual(tools.first?.description, "Echo back.")
        XCTAssertEqual(tools.first?.schemaObject["required"] as? [String], ["text"])

        let ok = try await client.callTool("echo", arguments: ["text": "안녕"], timeout: 10)
        XCTAssertEqual(ok.text, "echo: 안녕")
        XCTAssertFalse(ok.isError)
        let bad = try await client.callTool("fail", arguments: [:], timeout: 10)
        XCTAssertTrue(bad.isError)

        do {
            _ = try await client.callTool("quit", arguments: [:], timeout: 10)
            XCTFail("a server that exits must fail the call")
        } catch {
            XCTAssertTrue("\(error.localizedDescription)".contains("exited"))
        }
        XCTAssertFalse(client.isRunning)
    }
}
