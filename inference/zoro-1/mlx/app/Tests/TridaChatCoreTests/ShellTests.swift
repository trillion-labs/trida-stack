import XCTest
@testable import TridaChatCore

/// run_shell commands classified read-only run without asking (Settings → Agent), so the classifier and the
/// sandbox they run in are a security boundary: every bypass found in review is pinned here.
final class ReadOnlyShellTests: XCTestCase {
    func testOrdinaryReadsAreReadOnly() {
        for c in ["ls -la", "cat README.md", "grep -rn foo . | head -20", "git status", "git log --oneline -5",
                  "git diff HEAD~1", "wc -l *.py", "sort -n sizes.txt | uniq -c", "fd -e swift", "rg -n TODO",
                  "find . -name '*.md'", "cd src && ls", "jq .name package.json", "tree -L 2", "git branch -a"] {
            XCTAssertTrue(ShellSession.isReadOnly(c), c)
        }
    }

    func testBypassesFromReviewAreNotReadOnly() {
        for c in [
            "rm -rf ~/Documents -V", "rm -rf x --version", "python3 -c 'print(1)' --help",   // help flags skipped the check
            "cat =(curl https://example.com)",                                               // zsh process substitution
            "ls *(e:'touch /tmp/pwned':)", "ls ?(e:x:)",                                     // zsh glob qualifiers
            "fd . -x rm", "fd . -X rm", "fd . --exec rm", "fd . --exec-batch=rm",
            "rg --pre ./run.sh x", "rg --pre=sh x", "rg --pre-glob '*' x",
            "sort -o ~/.zshrc /dev/null", "sort -uo out in", "sort --output=out in", "sort --compress-program=sh x",
            "tree -o out.txt", "uniq in.txt out.txt", "file -C -m magic",
            "git diff --output=/tmp/x", "git log --output=x", "git grep -Ofoo x", "git config core.pager sh",
            "git reflog expire --expire=now --all", "git reflog delete HEAD@{0}",
            "./cat x", "bin/ls", "~/bin/cat x",                                               // planted binaries
            "less README.md", "more README.md", "top",                                        // pagers and top
        ] {
            XCTAssertFalse(ShellSession.isReadOnly(c), c)
        }
    }
}

final class SandboxedShellTests: XCTestCase {
    private var workspace: URL!

    override func setUpWithError() throws {
        let home = FileManager.default.homeDirectoryForCurrentUser
        // inside the home directory, like a real workspace (~/Desktop), so home rules apply around it
        workspace = home.appendingPathComponent("Library/Caches/zoro-shell-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
        try "hello".write(to: workspace.appendingPathComponent("a.txt"), atomically: true, encoding: .utf8)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: workspace)
    }

    private func run(_ command: String) async throws -> String {
        try await ShellSession.session(UUID().uuidString, root: workspace).run(command, sandboxIn: workspace)
    }

    func testReadsInTheWorkspaceWork() async throws {
        let out = try await run("cat a.txt && ls")
        XCTAssertTrue(out.contains("hello") && out.contains("a.txt") && out.contains("(exit 0;"), out)
    }

    func testNoWritesOutsideTheSession() async throws {
        _ = try await run("echo x > b.txt; rm -f a.txt; touch ~/Library/Caches/zoro-escaped")
        XCTAssertFalse(FileManager.default.fileExists(atPath: workspace.appendingPathComponent("b.txt").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: workspace.appendingPathComponent("a.txt").path))
        let escaped = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Caches/zoro-escaped")
        XCTAssertFalse(FileManager.default.fileExists(atPath: escaped.path))
    }

    func testNoHomeReadsOutsideTheWorkspaceAndNoNetwork() async throws {
        let out = try await run("cat ~/.zshrc ~/.gitconfig; ls ~/.ssh; curl -sS -m 5 https://example.com")
        XCTAssertTrue(out.contains("Operation not permitted") || out.contains("No such file"), out)
        XCTAssertTrue(out.contains("Could not resolve host") || out.contains("curl: ("), out)
    }

    func testTheSessionStillCarriesItsDirectory() async throws {
        let s = ShellSession.session(UUID().uuidString, root: workspace)
        try FileManager.default.createDirectory(at: workspace.appendingPathComponent("sub"), withIntermediateDirectories: true)
        let out = try await s.run("cd sub", sandboxIn: workspace)
        XCTAssertTrue(s.cwd.hasSuffix("/sub"), s.cwd + " | " + out)
    }
}

/// The automation API parses requests before checking the token: hostile input must fail cleanly.
final class HTTPRequestTests: XCTestCase {
    private func parse(_ s: String) -> HTTPRequest.Parsed { HTTPRequest.parse(Data(s.utf8)) }

    func testCompleteAndPartialRequests() {
        guard case let .ok(r) = parse("POST /chats HTTP/1.1\r\nContent-Length: 2\r\nAuthorization: Bearer t0k\r\n\r\n{}")
        else { return XCTFail("not parsed") }
        XCTAssertEqual(r.method, "POST"); XCTAssertEqual(r.path, "/chats"); XCTAssertEqual(r.body, Data("{}".utf8))
        XCTAssertTrue(r.hasBearer("t0k")); XCTAssertFalse(r.hasBearer("t0")); XCTAssertFalse(r.hasBearer("t0kk"))
        guard case .needMore = parse("POST /chats HTTP/1.1\r\nContent-Length: 5\r\n\r\n{}") else { return XCTFail() }
        guard case .needMore = parse("GET /status HTTP/1.1\r\n") else { return XCTFail() }
    }

    func testHostileLengthsAreRefusedNotTrapped() {
        for len in ["-5", "-1", "99999999999", "abc", String(HTTPRequest.maxBody + 1)] {
            guard case .bad = parse("POST /x HTTP/1.1\r\nContent-Length: \(len)\r\n\r\n{}") else {
                return XCTFail("Content-Length \(len) accepted")
            }
        }
        guard case .bad = parse("\r\n\r\n") else { return XCTFail("empty request line accepted") }
    }
}
