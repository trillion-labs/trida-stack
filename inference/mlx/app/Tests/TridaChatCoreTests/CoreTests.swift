import XCTest
@testable import TridaChatCore

final class SSEParserTests: XCTestCase {
    func testStreamFromServer() {
        // Lines exactly as trida_mlx/server.py writes them.
        let lines = [
            ": prefill 512/14000",
            #"data: {"id":"c","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}"#,
            "",
            #"data: {"choices":[{"index":0,"delta":{"reasoning_content":"hmm"},"finish_reason":null}]}"#,
            #"data: {"choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}"#,
            #"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\": \"a.txt\"}"},"index":0}]},"finish_reason":null}]}"#,
            #"data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
            #"data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5,"trida":{"mode":"self-spec(N=4)","prompt_tokens":10,"reused_tokens":4,"new_tokens":5,"forwards":2,"tokens_per_forward":2.5,"prefill_s":0.1,"decode_s":0.1,"decode_tok_s":50.0,"finish_reason":"stop"}}}"#,
            "data: [DONE]",
        ]
        var p = SSEParser()
        var events: [StreamEvent] = []
        var ended = false
        for l in lines {
            guard let e = p.parse(line: l) else { ended = true; break }
            events += e
        }
        XCTAssertTrue(ended)
        XCTAssertEqual(events.first, .prefill(done: 512, total: 14000))
        XCTAssertTrue(events.contains(.reasoning("hmm")))
        XCTAssertTrue(events.contains(.content("Hi")))
        XCTAssertTrue(events.contains(.toolCalls([ToolCall(id: "call_1", name: "read_file", arguments: #"{"path": "a.txt"}"#)])))
        XCTAssertTrue(events.contains(.finish("tool_calls")))
        guard case let .usage(st)? = events.last else { return XCTFail("no usage") }
        XCTAssertEqual(st.mode, "self-spec(N=4)")
        XCTAssertEqual(st.cachedTokens, 4)
        XCTAssertEqual(st.tokensPerForward, 2.5)
        XCTAssertEqual(st.decodeTokensPerSecond, 50)
    }

    func testRequestBytesAreStableAcrossRequests() throws {
        // Same tools + messages must give byte-identical JSON every time (prompt-cache hits).
        let tools = AgentTools(root: URL(fileURLWithPath: "/tmp"), allowShell: true, allowWrite: true, vision: true)
        let msgs = [ChatMessage(role: .user, content: "hi").openAIMessage()]
        let first = try ChatClient.encode(ChatClient.requestBody(messages: msgs, tools: tools.specs,
                                                                 options: GenerationOptions(), model: "m"))
        for _ in 0..<20 {
            let again = try ChatClient.encode(ChatClient.requestBody(messages: msgs, tools: tools.specs,
                                                                     options: GenerationOptions(), model: "m"))
            XCTAssertEqual(first, again)
        }
        let text = String(decoding: first, as: UTF8.self)
        XCTAssertTrue(text.contains(#"{"function":{"description":"#), text.prefix(300).description)
    }

    func testRequestBodyIsJSON() throws {
        var m = ChatMessage(role: .user, content: "look", images: [Data([0x89, 0x50])])
        m.reasoning = ""
        var o = GenerationOptions()
        o.selfSpec = false
        let body = ChatClient.requestBody(messages: [m.openAIMessage()], tools: nil, options: o, model: "zoro-1-mlx-q8")
        XCTAssertTrue(JSONSerialization.isValidJSONObject(body))
        XCTAssertEqual(body["trida_mode"] as? String, "causal")
        let parts = (body["messages"] as? [[String: Any]])?.first?["content"] as? [[String: Any]]
        XCTAssertEqual(parts?.count, 2)
        XCTAssertEqual(parts?.first?["type"] as? String, "image_url")
    }
}

final class ToolTests: XCTestCase {
    func testCalculator() throws {
        XCTAssertEqual(try Calculator.evaluate("17*24+3"), 411)
        XCTAssertEqual(try Calculator.evaluate("2^3^2"), 512)
        XCTAssertEqual(try Calculator.evaluate("2**10 - (3 - 5) * -1"), 1022)
        XCTAssertEqual(try Calculator.evaluate("7 % 3"), 1)
        XCTAssertThrowsError(try Calculator.evaluate("import os"))
        XCTAssertEqual(Calculator.format(4.0), "4")
    }

    func testWorkspaceConfinement() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("trida-tools-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try "hello needle\n".write(to: root.appendingPathComponent("a.txt"), atomically: true, encoding: .utf8)
        let t = AgentTools(root: root, allowShell: false, allowWrite: true)

        let read = await t.call(ToolCall(id: "1", name: "read_file", arguments: #"{"path":"a.txt"}"#))
        XCTAssertEqual(read, "hello needle\n")
        let esc = await t.call(ToolCall(id: "2", name: "read_file", arguments: #"{"path":"../../etc/passwd"}"#))
        XCTAssertTrue(esc.hasPrefix("error:"), esc)
        let hit = await t.call(ToolCall(id: "3", name: "search_files", arguments: #"{"query":"needle"}"#))
        XCTAssertEqual(hit, "a.txt:1: hello needle")
        let shell = await t.call(ToolCall(id: "4", name: "run_shell", arguments: #"{"command":"echo hi"}"#))
        XCTAssertTrue(shell.contains("not enabled"), shell)
        let w = await t.call(ToolCall(id: "5", name: "write_file", arguments: #"{"path":"sub/b.txt","content":"x"}"#))
        XCTAssertTrue(w.hasPrefix("wrote"), w)
        XCTAssertEqual(t.specs.count, 7)

        // find_files by name; absolute paths inside the workspace work, outside ones are refused
        let found = await t.call(ToolCall(id: "6", name: "find_files", arguments: #"{"pattern":"*.txt"}"#))
        XCTAssertTrue(found.contains("a.txt") && found.contains("sub/b.txt"), found)
        let plain = await t.call(ToolCall(id: "7", name: "find_files", arguments: #"{"pattern":"B"}"#))
        XCTAssertTrue(plain.contains("sub/b.txt"), plain)
        let abs = await t.call(ToolCall(id: "8", name: "read_file", arguments: "{\"path\":\"\(t.root.path)/a.txt\"}"))
        XCTAssertEqual(abs, "hello needle\n")
        let noSlash = await t.call(ToolCall(id: "9", name: "list_dir", arguments: "{\"path\":\"\(t.root.path.dropFirst())\"}"))
        XCTAssertTrue(noSlash.contains("a.txt"), noSlash)
        let outside = await t.call(ToolCall(id: "10", name: "read_file", arguments: #"{"path":"/home/user/x.txt"}"#))
        XCTAssertTrue(outside.hasPrefix("error:"), outside)
        let missingDir = await t.call(ToolCall(id: "11", name: "search_files", arguments: #"{"query":"x","path":"nope"}"#))
        XCTAssertTrue(missingDir.contains("no such directory"), missingDir)
    }
}

final class SetupTests: XCTestCase {
    func testTerminalLogHandlesProgressBars() {
        var log = TerminalLog()
        log.append("Fetching 17 files:   0%|          | 0/17\rFetching 17 files:  47%|####      | 8/17")
        log.append("\rFetching 17 files: 100%|##########| 17/17\n")
        log.append("done\r\n")
        XCTAssertEqual(log.lines, ["Fetching 17 files: 100%|##########| 17/17", "done", ""])
        XCTAssertEqual(log.current, "done")
    }

    func testRecipeFolders() {
        let r = ModelRecipe(repo: "trillionlabs/Zoro-1", note: "")
        XCTAssertEqual(r.rawFolder(), "Zoro-1")
        XCTAssertEqual(r.convertedFolder(bits: 8), "Zoro-1-mlx-q8")
        XCTAssertEqual(r.convertedFolder(bits: 16), "Zoro-1")
    }
}

final class RuntimeTests: XCTestCase {
    func testBundledCommandsUseModules() {
        let py = URL(fileURLWithPath: "/Applications/Zoro-1 Chat.app/Contents/Resources/python/bin/python3")
        let rt = PythonRuntime.bundled(python: py)
        let srv = rt.server(["--port", "8080"])
        XCTAssertEqual(srv.executable, py)
        XCTAssertEqual(Array(srv.arguments.prefix(2)), ["-m", "trida_mlx.server"])
        let dl = rt.download(repo: "trillionlabs/Zoro-1", to: URL(fileURLWithPath: "/tmp/Zoro-1"))
        XCTAssertEqual(dl.arguments.first, "-c")
        XCTAssertEqual(Array(dl.arguments.suffix(2)), ["trillionlabs/Zoro-1", "/tmp/Zoro-1"])
        XCTAssertEqual(rt.environment["PYTHONDONTWRITEBYTECODE"], "1")
    }

    func testProjectRuntimeNeedsPyproject() {
        let rt = PythonRuntime.project(URL(fileURLWithPath: "/nonexistent-\(UUID().uuidString)"))
        XCTAssertThrowsError(try rt.validate())
    }
}

#if canImport(ImageIO)
import CoreGraphics
import ImageIO

final class ImageToolTests: XCTestCase {
    func testViewImageReturnsPNGAndToolMessageCarriesIt() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("trida-img-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        // a 3000x2000 red image -> must come back downscaled to <= maxImageSide
        let ctx = CGContext(data: nil, width: 3000, height: 2000, bitsPerComponent: 8, bytesPerRow: 0,
                            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(red: 1, green: 0, blue: 0, alpha: 1)
        ctx.fill(CGRect(x: 0, y: 0, width: 3000, height: 2000))
        let url = root.appendingPathComponent("big.png")
        let dst = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)!
        CGImageDestinationAddImage(dst, ctx.makeImage()!, nil)
        XCTAssertTrue(CGImageDestinationFinalize(dst))

        let tools = AgentTools(root: root, allowShell: false, allowWrite: false, vision: true)
        XCTAssertTrue(tools.specs.contains { ($0["function"] as? [String: Any])?["name"] as? String == "view_image" })
        let out = await tools.run(ToolCall(id: "1", name: "view_image", arguments: #"{"path":"big.png"}"#))
        XCTAssertEqual(out.images.count, 1)
        let src = CGImageSourceCreateWithData(out.images[0] as CFData, nil)!
        let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as! [CFString: Any]
        XCTAssertEqual(props[kCGImagePropertyPixelWidth] as? Int, AgentTools.maxImageSide)

        var msg = ChatMessage(role: .tool, content: out.text, images: out.images)
        msg.toolCallID = "1"
        let parts = msg.openAIMessage()["content"] as? [[String: Any]]
        XCTAssertEqual(parts?.last?["type"] as? String, "image_url")

        let missing = await tools.run(ToolCall(id: "2", name: "view_image", arguments: #"{"path":"nope.png"}"#))
        XCTAssertTrue(missing.text.hasPrefix("error:") && missing.images.isEmpty)
        let noVision = AgentTools(root: root, allowShell: false, allowWrite: false)
        XCTAssertFalse(noVision.specs.contains { ($0["function"] as? [String: Any])?["name"] as? String == "view_image" })
    }
}
#endif

final class DecisionTests: XCTestCase {
    /// Object keys keep their order (the decision state is rendered in key order; options are criteria in order).
    func testOrderedJSONKeepsKeyOrder() throws {
        let body = OrderedJSON.object([
            ("state", .object([("app", .string("Mail")), ("goal", .string("say \"hi\"\n")),
                               ("actions so far", .strings(["row 'A'"]))])),
            ("questions", .object([("click", .object([("type", .string("choice")),
                                                       ("criteria", .object([("2", .string("b")), ("1", .string("a"))]))]))])),
            ("threshold", .number(0.7)), ("max_reasoning_tokens", .number(1024)), ("stream", .bool(false)),
        ])
        let text = String(decoding: body.data, as: UTF8.self)
        XCTAssertEqual(text, #"{"state":{"app":"Mail","goal":"say \"hi\"\n","actions so far":["row 'A'"]},"#
            + #""questions":{"click":{"type":"choice","criteria":{"2":"b","1":"a"}}},"#
            + #""threshold":0.7,"max_reasoning_tokens":1024,"stream":false}"#)
        XCTAssertNotNil(try JSONSerialization.jsonObject(with: body.data))
    }

    func testChoiceDecisionFromResponse() throws {
        let r: [String: Any] = [
            "answers": ["click": ["choice": "2", "probabilities": ["1": 0.3, "2": 0.6, "3": 0.1]]],
            "escalated": true, "reasoning": "the send button finishes it", "direct": ["timing_ms": 40.0],
            "reasoned": ["timing_ms": 60.0],
        ]
        let d = try XCTUnwrap(ChoiceDecision(response: r, question: "click"))
        XCTAssertEqual(d.choice, "2")
        XCTAssertEqual(d.ranked.map(\.0), ["2", "1", "3"])
        XCTAssertEqual(d.timingMs, 100, accuracy: 1e-9)
        XCTAssertTrue(d.summary.hasPrefix("p 0.60 after reasoning"))
    }
}

final class DecisionRequestTests: XCTestCase {
    func testBodyAndAnswersKeepOptionOrder() throws {
        let qs = [
            DecisionQuestion.choice("team", "Which team?", DecisionQuestion.parseOptions("it: IT support\nfacilities", type: .choice)),
            DecisionQuestion.yesNo("urgent", "Urgent?"),
            DecisionQuestion.score("sev", "How severe?", levels: ["low", "medium", "high"]),
        ]
        XCTAssertEqual(qs[0].options.map(\.0), ["it", "facilities"])
        let text = String(decoding: Decision.body(state: .string("s"), questions: qs, settings: DecisionSettings(),
                                                  stream: true).data, as: UTF8.self)
        XCTAssertTrue(text.hasPrefix(#"{"state":"s","questions":{"team":{"type":"choice","instructions":"Which team?","criteria":{"it":"IT support","facilities":""}},"urgent":{"type":"noul","instructions":"Urgent?"},"sev":{"type":"score","instructions":"How severe?","criteria":["low","medium","high"]}}"#), text)
        XCTAssertTrue(text.hasSuffix(#""trida_mode":"self-spec","stream":true}"#), text)
        let r: [String: Any] = [
            "answers": ["team": ["type": "choice", "choice": "facilities", "probabilities": ["it": 0.2, "facilities": 0.8]],
                        "urgent": ["type": "noul", "noul": 0.3],
                        "sev": ["type": "score", "score": 1.6, "probabilities": ["0": 0.1, "1": 0.2, "2": 0.7]]],
            "escalated": false,
        ]
        let d = try XCTUnwrap(DecisionResult(r, questions: qs))
        XCTAssertEqual(d.answers["team"]?.options.map(\.key), ["it", "facilities"])
        XCTAssertEqual(d.answers["urgent"]?.pick, "no")
        XCTAssertEqual(d.answers["sev"]?.pick, "high")
        XCTAssertEqual(d.provenance, "direct decision")
    }
}

final class PersistenceTests: XCTestCase {
    func testMessageAndDecisionRoundTrip() throws {
        var m = ChatMessage(role: .assistant, content: "hi", images: [Data([1, 2, 3])])
        m.toolRuns = [ToolRun(call: ToolCall(id: "c1", name: "now", arguments: "{}"))]
        m.decide = .card(UUID())
        m.stats = GenStats(usage: ["trida": ["mode": "self-spec(N=4)", "new_tokens": 5]])
        let back = try JSONDecoder().decode(ChatMessage.self, from: JSONEncoder().encode(m))
        XCTAssertEqual(back, m)

        let q = DecisionQuestion.choice("q1", "Which?", [("a", "first"), ("b", "")])
        let q2 = try JSONDecoder().decode(DecisionQuestion.self, from: JSONEncoder().encode(q))
        XCTAssertEqual(q2.options.map(\.0), ["a", "b"])
        XCTAssertEqual(q2.options.map(\.1), ["first", ""])
        let r = try XCTUnwrap(DecisionResult(["answers": ["q1": ["type": "choice", "choice": "b",
                                                                 "probabilities": ["a": 0.25, "b": 0.75]]]],
                                             questions: [q2]))
        let r2 = try XCTUnwrap(DecisionResult(raw: r.raw, questions: [q2]))
        XCTAssertEqual(r2.answers["q1"]?.pick, "b")
    }
}
