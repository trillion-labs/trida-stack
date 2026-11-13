// swift-tools-version:5.9
// The personal knowledge graph for Zoro-1 Chat, as a standalone package: the graph and the learning pipeline
// (TridaKnowledge, Foundation only), the Knowledge window and the "Learned" chip (TridaKnowledgeUI), and a
// self-test that runs without XCTest (`swift run kg-selftest`). See INTEGRATION.md for wiring it into the app.
import PackageDescription

let package = Package(
    name: "TridaKnowledge",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "TridaKnowledge", targets: ["TridaKnowledge"]),
        .library(name: "TridaKnowledgeUI", targets: ["TridaKnowledgeUI"]),
    ],
    targets: [
        .target(name: "TridaKnowledge"),
        .target(name: "TridaKnowledgeUI", dependencies: ["TridaKnowledge"]),
        .executableTarget(name: "kg-selftest", dependencies: ["TridaKnowledge"]),
    ]
)
