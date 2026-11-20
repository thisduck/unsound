// fluidaudio-eval <v2|v3> <models_dir> <ane|gpu> eval <manifest.jsonl> <out.jsonl>
// Same output format as parakeet-eval (first line meta, then one line per clip).
import CoreML
import FluidAudio
import Foundation

func readWav(_ path: String) -> [Float] {
    let d = try! Data(contentsOf: URL(fileURLWithPath: path))
    let b = [UInt8](d)
    func u16(_ i: Int) -> Int { Int(b[i]) | Int(b[i + 1]) << 8 }
    func u32(_ i: Int) -> Int { u16(i) | u16(i + 2) << 16 }
    var i = 12
    var bits = 16
    while i + 8 <= b.count {
        let id = String(bytes: b[i..<i + 4], encoding: .ascii)!
        let len = u32(i + 4)
        if id == "fmt " { bits = u16(i + 8 + 14) }
        if id == "data" {
            precondition(bits == 16, "need PCM16")
            var out = [Float](repeating: 0, count: len / 2)
            for k in 0..<len / 2 { out[k] = Float(Int16(bitPattern: UInt16(u16(i + 8 + 2 * k)))) / 32768 }
            return out
        }
        i += 8 + len + (len & 1)
    }
    fatalError("no data chunk: \(path)")
}

func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e6 }

let a = CommandLine.arguments
let version: AsrModelVersion = a[1] == "v2" ? .v2 : .v3
let modelsDir = URL(fileURLWithPath: a[2], isDirectory: true)
let units: MLComputeUnits? = a[3] == "gpu" ? .cpuAndGPU : nil
let manifest = a[5], outPath = a[6]

let t0 = now()
let repo = try await AsrModels.download(to: modelsDir, version: version)
let tDl = now()
let models = try await AsrModels.load(from: repo, version: version, encoderComputeUnits: units)
let asr = AsrManager(config: .default)
try await asr.loadModels(models)
let loadMs = now() - tDl
FileHandle.standardError.write("download/check \(Int(tDl - t0)) ms, load \(Int(loadMs)) ms\n".data(using: .utf8)!)

let lines = try! String(contentsOfFile: manifest, encoding: .utf8).split(separator: "\n")
let items = lines.map { try! JSONSerialization.jsonObject(with: Data($0.utf8)) as! [String: Any] }
FileManager.default.createFile(atPath: outPath, contents: nil)
let out = FileHandle(forWritingAtPath: outPath)!
func emit(_ o: [String: Any]) {
    out.write(try! JSONSerialization.data(withJSONObject: o, options: [.sortedKeys]))
    out.write("\n".data(using: .utf8)!)
}

func run(_ pcm: [Float]) async -> ASRResult? {
    var st = TdtDecoderState.make(decoderLayers: await asr.decoderLayerCount)
    do { return try await asr.transcribe(pcm, decoderState: &st) } catch {
        FileHandle.standardError.write("transcribe failed: \(error)\n".data(using: .utf8)!)
        return nil
    }
}

let first = readWav(items[0]["path"] as! String)
var t = now()
_ = await run(first)
emit(["meta": true, "engine": "fluidaudio-\(a[1])-\(a[3])", "load_ms": loadMs, "first_run_ms": now() - t,
      "first_clip_s": Double(first.count) / 16000])
for (i, it) in items.enumerated() {
    let pcm = readWav(it["path"] as! String)
    t = now()
    let r = await run(pcm)
    let ms = now() - t
    // Group subword tokens into words (SentencePiece "▁" starts a word).
    var spans: [[Any]] = []
    for tk in r?.tokenTimings ?? [] {
        let piece = tk.token
        if piece.hasPrefix("▁") || piece.hasPrefix(" ") || spans.isEmpty {
            spans.append([tk.startTime, tk.endTime, piece.replacingOccurrences(of: "▁", with: "").trimmingCharacters(in: .whitespaces)])
        } else {
            spans[spans.count - 1][1] = tk.endTime
            spans[spans.count - 1][2] = (spans[spans.count - 1][2] as! String) + piece
        }
    }
    emit(["id": it["id"]!, "dur": Double(pcm.count) / 16000, "ms": ms, "hyp": r?.text ?? "", "spans": spans])
    if i % 20 == 0 { FileHandle.standardError.write("[fa] \(i)/\(items.count) \(Int(ms))ms \((r?.text ?? "").prefix(60))\n".data(using: .utf8)!) }
}
