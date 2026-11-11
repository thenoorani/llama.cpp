import Foundation
import UIKit
import llama

// On-device llama.cpp benchmark and correctness gate, driven by run.sh through devicectl.
//
// MODE=bench (default): pp512 on real text (upper quartile ranks, like scripts/artemis/bench.sh
//   on the Mac), tg128 from an empty context, and tg128 after DEPTH tokens of context. Before
//   each timed sample it waits for the phone to cool to THERMAL_MAX.
// MODE=ref: scores fixed chunks of text.txt and saves each position's top-K next-token
//   log-probs to Documents/ref.bin. Run once on the original code.
// MODE=gate: scores the same chunks and compares them with the bundled ref.bin (KL divergence,
//   top-1 agreement); it fails when they drift past GATE_MAX_MEAN_KL or GATE_MIN_TOP1.
//
// Every result is an "ARTEMIS_METRIC name=value" line. The app exits 0 on success, 1 otherwise.

@main
final class AppDelegate: UIResponder, UIApplicationDelegate {
    var window: UIWindow?

    func application(_ application: UIApplication,
                     didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?) -> Bool {
        application.isIdleTimerDisabled = true // keep the phone from locking mid-run
        UIDevice.current.isBatteryMonitoringEnabled = true
        if env["MODE"] == "session" && envInt("UI", 1) != 0 {
            window = UILoad.install(iters: envInt("UI_LOAD", 16))
        }
        let battery = UIDevice.current.batteryState
        DispatchQueue.global(qos: .userInitiated).async {
            let ok = Runner(battery: battery).run()
            fflush(stdout)
            exit(ok ? 0 : 1)
        }
        return true
    }
}

let env = ProcessInfo.processInfo.environment
func envInt(_ key: String, _ def: Int) -> Int { Int(env[key] ?? "") ?? def }
func envDouble(_ key: String, _ def: Double) -> Double { Double(env[key] ?? "") ?? def }

func metric(_ name: String, _ value: Double) {
    print(String(format: "ARTEMIS_METRIC %@=%.6g", name, value))
}

func info(_ msg: String) { print("ARTEMIS_INFO \(msg)") }

struct BenchError: Error, CustomStringConvertible {
    let description: String
    init(_ d: String) { description = d }
}

// nominal=0, fair=1, serious=2, critical=3
func thermalLevel() -> Int { ProcessInfo.processInfo.thermalState.rawValue }

func quantile(_ xs: [Double], _ q: Double) -> Double {
    let s = xs.sorted()
    let pos = q * Double(s.count - 1)
    let lo = Int(pos.rounded(.down)), hi = Int(pos.rounded(.up))
    return s[lo] + (s[hi] - s[lo]) * (pos - Double(lo))
}

func cvPct(_ xs: [Double]) -> Double {
    let mean = xs.reduce(0, +) / Double(xs.count)
    let sd = sqrt(xs.map { ($0 - mean) * ($0 - mean) }.reduce(0, +) / Double(max(1, xs.count - 1)))
    return 100 * sd / mean
}

final class Session {
    let model: OpaquePointer
    let vocab: OpaquePointer
    let nVocab: Int
    var ctx: OpaquePointer?
    var batch: llama_batch
    var batchCap = 0

    init(path: String) throws {
        llama_backend_init()
        var mp = llama_model_default_params()
        if let ngl = env["N_GPU_LAYERS"].flatMap(Int32.init) { mp.n_gpu_layers = ngl } // calibration only
        guard let m = llama_model_load_from_file(path, mp) else {
            throw BenchError("could not load model")
        }
        model = m
        vocab = llama_model_get_vocab(m)
        nVocab = Int(llama_vocab_n_tokens(vocab))
        batch = llama_batch_init(1, 0, 1)
    }

    deinit {
        llama_batch_free(batch)
        if let ctx { llama_free(ctx) }
        llama_model_free(model)
        llama_backend_free()
    }

    func newContext(nCtx: Int, nBatch: Int, nUbatch: Int? = nil) throws {
        if let ctx { llama_free(ctx) }
        var p = llama_context_default_params()
        p.n_ctx = UInt32(nCtx)
        p.n_batch = UInt32(nBatch)
        p.n_ubatch = UInt32(nUbatch ?? nBatch)
        guard let c = llama_init_from_model(model, p) else { throw BenchError("could not create context") }
        ctx = c
        llama_batch_free(batch)
        batch = llama_batch_init(Int32(nBatch), 0, 1)
        batchCap = nBatch
    }

    func clear() { llama_memory_clear(llama_get_memory(ctx), false) }

    // Decodes tokens at positions startPos..., in batch-sized pieces. Logits for every position
    // when allLogits, otherwise only for the last one.
    func decode(_ tokens: ArraySlice<llama_token>, startPos: Int, allLogits: Bool = false) throws {
        var pos = startPos
        var rest = tokens[...]
        while !rest.isEmpty {
            let piece = rest.prefix(batchCap)
            rest = rest.dropFirst(piece.count)
            batch.n_tokens = 0
            for t in piece {
                let n = Int(batch.n_tokens)
                batch.token[n] = t
                batch.pos[n] = Int32(pos)
                batch.n_seq_id[n] = 1
                batch.seq_id[n]![0] = 0
                batch.logits[n] = allLogits ? 1 : 0
                batch.n_tokens += 1
                pos += 1
            }
            batch.logits[Int(batch.n_tokens) - 1] = 1
            guard llama_decode(ctx, batch) == 0 else { throw BenchError("llama_decode failed") }
        }
        llama_synchronize(ctx)
    }

    func logits(_ i: Int) -> UnsafeBufferPointer<Float> {
        UnsafeBufferPointer(start: llama_get_logits_ith(ctx, Int32(i)), count: nVocab)
    }

    func tokenize(_ text: String) -> [llama_token] {
        let addBos = llama_vocab_get_add_bos(vocab)
        let cap = text.utf8.count + 2
        var out = [llama_token](repeating: 0, count: cap)
        let n = llama_tokenize(vocab, text, Int32(text.utf8.count), &out, Int32(cap), addBos, false)
        return Array(out.prefix(Int(max(0, n))))
    }
}

struct Runner {
    let battery: UIDevice.BatteryState

    func run() -> Bool {
        let mode = env["MODE"] ?? "bench"
        let lowPower = ProcessInfo.processInfo.isLowPowerModeEnabled
        info("mode=\(mode) battery_state=\(battery.rawValue) low_power=\(lowPower) thermal=\(thermalLevel())")
        do {
            guard let modelPath = Bundle.main.path(forResource: "model", ofType: "gguf"),
                  let textPath = Bundle.main.path(forResource: "text", ofType: "txt") else {
                throw BenchError("model.gguf or text.txt missing from app bundle")
            }
            let s = try Session(path: modelPath)
            let tokens = s.tokenize(try String(contentsOfFile: textPath, encoding: .utf8))
            info("text_tokens=\(tokens.count) n_vocab=\(s.nVocab)")
            switch mode {
            case "bench":
                guard battery == .charging || battery == .full else { throw BenchError("refusing to benchmark on battery") }
                guard !lowPower else { throw BenchError("refusing to benchmark in Low Power Mode") }
                try bench(s, tokens)
            case "session":
                guard battery == .charging || battery == .full else { throw BenchError("refusing to benchmark on battery") }
                guard !lowPower else { throw BenchError("refusing to benchmark in Low Power Mode") }
                try session(s, tokens)
            case "ref", "gate":
                try score(s, tokens, makeRef: mode == "ref")
            default:
                throw BenchError("unknown MODE \(mode)")
            }
            return true
        } catch {
            print("ARTEMIS_ERROR \(error)")
            return false
        }
    }

    // MARK: benchmark

    func bench(_ s: Session, _ tokens: [llama_token]) throws {
        let pp = 512, tg = 128
        let depth = envInt("DEPTH", 4096)
        let ppReps = envInt("PP_REPS", 10), tgReps = envInt("TG_REPS", 3), tgdReps = envInt("TGD_REPS", 3)
        let thermalMax = envInt("THERMAL_MAX", 1)
        let coolTimeout = envDouble("COOL_TIMEOUT_S", 600)
        guard tokens.count >= depth + tg else { throw BenchError("text too short for DEPTH=\(depth)") }

        try s.newContext(nCtx: depth + tg + 64, nBatch: pp)

        var waited = 0.0
        var hottest = thermalLevel()
        func waitCool() throws {
            let t0 = Date()
            while thermalLevel() > thermalMax {
                if Date().timeIntervalSince(t0) > coolTimeout {
                    throw BenchError("phone did not cool to thermal level \(thermalMax) within \(Int(coolTimeout))s")
                }
                Thread.sleep(forTimeInterval: 5)
            }
            waited += Date().timeIntervalSince(t0)
        }
        func timed(_ body: () throws -> Void) rethrows -> Double {
            let t0 = DispatchTime.now().uptimeNanoseconds
            try body()
            hottest = max(hottest, thermalLevel())
            return Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
        }
        func generate(from start: Int) throws {
            for i in start..<(start + tg) {
                try s.decode(tokens[i...i], startPos: i)
            }
        }

        s.clear()
        try s.decode(tokens[0..<pp], startPos: 0) // warmup: shaders, weight paging

        var ppS: [Double] = [], tgS: [Double] = [], tgdS: [Double] = []
        var thermals: [Int] = []
        for _ in 0..<ppReps {
            try waitCool()
            s.clear()
            thermals.append(thermalLevel())
            ppS.append(Double(pp) / (try timed { try s.decode(tokens[0..<pp], startPos: 0) }))
        }
        for _ in 0..<tgReps {
            try waitCool()
            s.clear()
            thermals.append(thermalLevel())
            tgS.append(Double(tg) / (try timed { try generate(from: 0) }))
        }
        for _ in 0..<tgdReps {
            s.clear()
            try s.decode(tokens[0..<depth], startPos: 0) // untimed context fill
            try waitCool()
            thermals.append(thermalLevel())
            tgdS.append(Double(tg) / (try timed { try generate(from: depth) }))
        }

        info("pp512_samples=\(ppS.map { String(format: "%.1f", $0) }.joined(separator: ","))")
        info("tg128_samples=\(tgS.map { String(format: "%.2f", $0) }.joined(separator: ","))")
        info("tg128_d\(depth)_samples=\(tgdS.map { String(format: "%.2f", $0) }.joined(separator: ","))")
        info("thermal_at_sample_start=\(thermals.map(String.init).joined(separator: ",")) thermal_end=\(thermalLevel())")
        metric("pp512_tps", quantile(ppS, 0.75))
        metric("pp512_median_tps", quantile(ppS, 0.5))
        metric("pp512_cv_pct", cvPct(ppS))
        metric("tg128_tps", quantile(tgS, 0.5))
        metric("tg128_cv_pct", cvPct(tgS))
        metric("tg128_d\(depth)_tps", quantile(tgdS, 0.5))
        metric("tg128_d\(depth)_cv_pct", cvPct(tgdS))
        metric("cool_wait_s", waited)
        metric("thermal_max", Double(hottest))
    }

    // MARK: multi-turn session

    // Back-to-back conversations of TURNS turns, until SESSION_MIN minutes have passed (at least
    // one). Each turn prefills TURN_IN tokens of text as the user message, then generates TURN_OUT
    // tokens greedily and feeds them back, so the context grows to TURNS*(TURN_IN+TURN_OUT).
    // Starts once the phone is at THERMAL_START. One "ARTEMIS_TURN {json}" line per turn; run.sh
    // turns those into metrics and a chart.
    func session(_ s: Session, _ tokens: [llama_token]) throws {
        let turns = envInt("TURNS", 15), tin = envInt("TURN_IN", 250), tout = envInt("TURN_OUT", 150)
        let minutes = envDouble("SESSION_MIN", 0)
        let thermalStart = envInt("THERMAL_START", 0), coolTimeout = envDouble("COOL_TIMEOUT_S", 900)
        guard tokens.count >= turns * tin else { throw BenchError("text too short for \(turns) turns") }

        try s.newContext(nCtx: turns * (tin + tout) + 64, nBatch: 512)
        let smpl = llama_sampler_init_greedy()
        defer { llama_sampler_free(smpl) }
        func now() -> Double { Double(DispatchTime.now().uptimeNanoseconds) / 1e9 }

        // warmup outside the session: shader compilation for the batch and single-token paths
        s.clear()
        try s.decode(tokens[0..<32], startPos: 0)
        try s.decode(tokens[32...32], startPos: 32)
        s.clear()

        let w0 = now()
        while thermalLevel() > thermalStart {
            if now() - w0 > coolTimeout { throw BenchError("phone did not cool to thermal level \(thermalStart) within \(Int(coolTimeout))s") }
            Thread.sleep(forTimeInterval: 5)
        }
        info(String(format: "cool_wait_s=%.0f", now() - w0))
        if let ui = UILoad.shared {
            Thread.sleep(forTimeInterval: 3) // let launch and warmup frames settle
            _ = ui.drain()
            Thread.sleep(forTimeInterval: 10)
            let idle = ui.drain()
            info(String(format: "ui_idle frames=%d dropped=%d p99_ms=%.1f", idle.frames, idle.dropped, idle.p99ms))
        }

        let t0 = now()
        var conv = 0
        repeat {
            s.clear()
            llama_sampler_reset(smpl)
            var pos = 0
            for t in 0..<turns {
                _ = UILoad.shared?.drain()
                let start = now(), depth = pos
                try s.decode(tokens[(t * tin)..<((t + 1) * tin)], startPos: pos)
                pos += tin
                var tok = llama_sampler_sample(smpl, s.ctx, -1)
                let ttft = now() - start
                let g0 = now()
                for _ in 1..<tout {
                    try s.decode([tok][...], startPos: pos)
                    pos += 1
                    tok = llama_sampler_sample(smpl, s.ctx, -1)
                }
                try s.decode([tok][...], startPos: pos) // the reply's last token joins the context
                pos += 1
                let gen = now() - g0
                let ui = UILoad.shared?.drain()
                let battery = DispatchQueue.main.sync { UIDevice.current.batteryLevel }
                let row: [String: Double] = [
                    "t_s": start - t0, "conv": Double(conv), "turn": Double(t), "depth": Double(depth),
                    "ttft_ms": ttft * 1000, "prefill_tps": Double(tin) / ttft, "tg_tps": Double(tout) / gen,
                    "thermal": Double(thermalLevel()), "mem_mb": footprintMB(), "battery": Double(battery),
                    "ui_frames": Double(ui?.frames ?? 0), "ui_dropped": Double(ui?.dropped ?? 0),
                    "ui_p99_ms": ui?.p99ms ?? 0,
                ]
                let json = try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys])
                print("ARTEMIS_TURN " + String(decoding: json, as: UTF8.self))
                fflush(stdout)
            }
            conv += 1
        } while now() - t0 < minutes * 60
        info(String(format: "session_s=%.1f conversations=%d", now() - t0, conv))
    }

    // MARK: reference and gate

    static let refMagic: UInt32 = 0x4C42_5246 // "LBRF"

    func score(_ s: Session, _ tokens: [llama_token], makeRef: Bool) throws {
        let chunk = envInt("GATE_CHUNK", 256), nChunks = envInt("GATE_CHUNKS", 4), k = 32
        guard tokens.count >= chunk * nChunks else { throw BenchError("text too short for gate") }
        let used = Array(tokens[0..<(chunk * nChunks)])
        // GATE_UBATCH < chunk takes different kernel paths for the same math (calibration only)
        try s.newContext(nCtx: chunk, nBatch: chunk, nUbatch: env["GATE_UBATCH"].flatMap(Int.init))

        // ref layout (little endian): magic, n_vocab, chunk, n_chunks, k, used token ids,
        // then per scored position k int32 ids and k float32 log-probs.
        var refIds: [Int32] = [], refLps: [Float] = []
        var out = Data()
        if makeRef {
            for v in [Self.refMagic, UInt32(s.nVocab), UInt32(chunk), UInt32(nChunks), UInt32(k)] {
                withUnsafeBytes(of: v.littleEndian) { out.append(contentsOf: $0) }
            }
            used.withUnsafeBytes { out.append(contentsOf: $0) }
        } else {
            guard let refPath = Bundle.main.path(forResource: "ref", ofType: "bin") else {
                throw BenchError("ref.bin missing from app bundle; create it with MODE=ref on the original code")
            }
            let d = try Data(contentsOf: URL(fileURLWithPath: refPath))
            let header = d.withUnsafeBytes { Array($0.bindMemory(to: UInt32.self).prefix(5)) }
            guard header == [Self.refMagic, UInt32(s.nVocab), UInt32(chunk), UInt32(nChunks), UInt32(k)] else {
                throw BenchError("ref.bin was made with a different model or gate settings")
            }
            var off = 20
            let refTokens = d.subdata(in: off..<(off + 4 * used.count)).withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }
            guard refTokens == used else { throw BenchError("ref.bin was made from different text or tokenizer") }
            off += 4 * used.count
            let nPos = nChunks * (chunk - 1)
            d.withUnsafeBytes { raw in
                for p in 0..<nPos {
                    let base = off + p * k * 8
                    refIds.append(contentsOf: UnsafeRawBufferPointer(rebasing: raw[base..<(base + 4 * k)]).bindMemory(to: Int32.self))
                    refLps.append(contentsOf: UnsafeRawBufferPointer(rebasing: raw[(base + 4 * k)..<(base + 8 * k)]).bindMemory(to: Float.self))
                }
            }
        }

        var nll = 0.0, klSum = 0.0, klMax = 0.0, top1 = 0, n = 0
        var kls: [Double] = []
        // per kernel path: 0 = batched (prompt processing), 1 = one token at a time (generation)
        var pathKL = [0.0, 0.0], pathTop1 = [0, 0], pathN = [0, 0]
        var topIds = [Int32](repeating: 0, count: k), topVals = [Float](repeating: 0, count: k)
        for c in 0..<nChunks {
            let toks = used[(c * chunk)..<((c + 1) * chunk)]
            // First half of the chunks in one batch (prompt-processing kernels), second half one
            // token at a time (the generation kernels: mat-vec, single-step recurrent updates).
            let stepwise = c >= nChunks / 2
            s.clear()
            if !stepwise { try s.decode(toks, startPos: 0, allLogits: true) }
            for i in 0..<(chunk - 1) {
                if stepwise { try s.decode(toks[(toks.startIndex + i)...(toks.startIndex + i)], startPos: i) }
                let lg = s.logits(stepwise ? 0 : i)
                var mx = -Float.infinity
                for v in lg where v > mx { mx = v }
                var sum = 0.0
                for v in lg { sum += Double(expf(v - mx)) }
                let lse = Double(mx) + log(sum)
                let next = Int(toks[toks.startIndex + i + 1])
                nll += lse - Double(lg[next])

                if makeRef {
                    // top-k by logit, kept sorted descending
                    for j in 0..<k { topIds[j] = -1; topVals[j] = -.infinity }
                    for (id, v) in lg.enumerated() where v > topVals[k - 1] {
                        var j = k - 1
                        while j > 0 && topVals[j - 1] < v { topVals[j] = topVals[j - 1]; topIds[j] = topIds[j - 1]; j -= 1 }
                        topVals[j] = v; topIds[j] = Int32(id)
                    }
                    topIds.withUnsafeBytes { out.append(contentsOf: $0) }
                    let lps = topVals.map { Float(Double($0) - lse) }
                    lps.withUnsafeBytes { out.append(contentsOf: $0) }
                } else {
                    // KL(ref || new) over ref's top-k plus one bucket for the remaining mass
                    let base = n * k
                    var kl = 0.0, pSum = 0.0, qSum = 0.0
                    var argmax = 0
                    for j in 0..<k {
                        let lp = Double(refLps[base + j]), lq = Double(lg[Int(refIds[base + j])]) - lse
                        let p = exp(lp)
                        kl += p * (lp - lq); pSum += p; qSum += exp(lq)
                    }
                    let rP = max(0, 1 - pSum), rQ = max(1e-12, 1 - qSum)
                    if rP > 1e-12 { kl += rP * log(rP / rQ) }
                    for (id, v) in lg.enumerated() where v > lg[argmax] { argmax = id }
                    let path = stepwise ? 1 : 0
                    if Int32(argmax) == refIds[base] { top1 += 1; pathTop1[path] += 1 }
                    kl = max(0, kl)
                    kls.append(kl); klSum += kl; klMax = max(klMax, kl)
                    pathKL[path] += kl; pathN[path] += 1
                }
                n += 1
            }
        }

        let ppl = exp(nll / Double(n))
        metric("gate_ppl", ppl)
        if makeRef {
            let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("ref.bin")
            try out.write(to: url, options: .atomic)
            info("wrote Documents/ref.bin positions=\(n) bytes=\(out.count)")
            return
        }
        let meanKL = klSum / Double(n), top1Rate = Double(top1) / Double(n)
        metric("gate_mean_kl", meanKL)
        metric("gate_p99_kl", quantile(kls, 0.99))
        metric("gate_max_kl", klMax)
        metric("gate_top1_agree", top1Rate)
        // Thresholds calibrated on Qwen3.5-2B (2026-11-04): reordering the same GPU math gives
        // mean KL ~4e-7; a planted off-by-one in the q6_K mat-vec kernel gives ~1.5e-3 on its path.
        let maxMeanKL = envDouble("GATE_MAX_MEAN_KL", 1e-4), minTop1 = envDouble("GATE_MIN_TOP1", 0.99)
        var failures: [String] = []
        for (path, name) in ["batch", "step"].enumerated() {
            let mk = pathKL[path] / Double(pathN[path]), t1 = Double(pathTop1[path]) / Double(pathN[path])
            metric("gate_\(name)_mean_kl", mk)
            metric("gate_\(name)_top1_agree", t1)
            if mk > maxMeanKL || t1 < minTop1 {
                failures.append(String(format: "%@ path mean KL %.2e (max %.0e), top-1 %.4f (min %.2f)",
                                       name, mk, maxMeanKL, t1, minTop1))
            }
        }
        guard failures.isEmpty else { throw BenchError("gate failed: " + failures.joined(separator: "; ")) }
        info("gate passed")
    }
}
