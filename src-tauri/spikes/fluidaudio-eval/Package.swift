// swift-tools-version: 6.0
// Spike: FluidAudio (CoreML Parakeet) over the same JSONL manifests as
// parakeet-eval / asr_eval. Build: FLUIDAUDIO_PATH=<checkout> swift build -c release
import PackageDescription
import Foundation

let local = ProcessInfo.processInfo.environment["FLUIDAUDIO_PATH"]

let package = Package(
    name: "fluidaudio-eval",
    platforms: [.macOS(.v14)],
    dependencies: [
        local.map { .package(path: $0) }
            ?? .package(url: "https://github.com/FluidInference/FluidAudio", revision: "e0c6103")
    ],
    targets: [
        .executableTarget(
            name: "fluidaudio-eval",
            dependencies: [.product(name: "FluidAudio", package: "FluidAudio")]
        )
    ]
)
