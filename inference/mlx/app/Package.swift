// swift-tools-version:5.9
// Zoro-1 Chat — a native macOS chat app for the on-device MLX server (../trida_mlx/server.py).
import PackageDescription

let package = Package(
    name: "TridaChat",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "TridaChat", targets: ["TridaChat"]),
    ],
    targets: [
        // Foundation only: OpenAI streaming client, server process, model catalog, agent tools.
        .target(name: "TridaChatCore", path: "Sources/TridaChatCore"),
        // SwiftUI app.
        .executableTarget(name: "TridaChat", dependencies: ["TridaChatCore"], path: "Sources/TridaChat"),
        .testTarget(name: "TridaChatCoreTests", dependencies: ["TridaChatCore"], path: "Tests/TridaChatCoreTests"),
    ]
)
