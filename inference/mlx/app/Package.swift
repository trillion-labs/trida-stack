// swift-tools-version:5.9
// Zoro-1 Chat — a native macOS chat app for the on-device MLX server (../trida_mlx/server.py).
import PackageDescription

let package = Package(
    name: "TridaChat",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "TridaChat", targets: ["TridaChat"]),
    ],
    // The personal knowledge graph (Knowledge/INTEGRATION.md).
    dependencies: [.package(path: "Knowledge")],
    targets: [
        // Foundation only: OpenAI streaming client, server process, model catalog, agent tools.
        .target(name: "TridaChatCore", path: "Sources/TridaChatCore"),
        // SwiftUI app.
        .executableTarget(name: "TridaChat",
                          dependencies: ["TridaChatCore",
                                         .product(name: "TridaKnowledge", package: "Knowledge"),
                                         .product(name: "TridaKnowledgeUI", package: "Knowledge")],
                          path: "Sources/TridaChat"),
        .testTarget(name: "TridaChatCoreTests", dependencies: ["TridaChatCore"], path: "Tests/TridaChatCoreTests"),
    ]
)
