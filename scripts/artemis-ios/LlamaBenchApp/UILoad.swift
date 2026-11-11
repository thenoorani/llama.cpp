import Foundation
import MetalKit
import UIKit

// Stand-in for a busy foreground UI while the model runs: a full-screen Metal scene with a fixed
// per-pixel cost (UI_LOAD loop iterations), drawn at 60 Hz, the iPhone 16's refresh rate. It
// records when each frame actually reached the screen, so jank caused by the model competing
// for the GPU and memory bus shows up as dropped frames.
final class UILoad: NSObject, MTKViewDelegate {
    static var shared: UILoad?
    static let period = 1.0 / 60

    private let queue: MTLCommandQueue
    private let pipeline: MTLRenderPipelineState
    private var iters: Int32
    private let t0 = CACurrentMediaTime()

    private let lock = NSLock()
    private var intervals: [Double] = []
    private var lastPresented = 0.0
    private var missed = 0

    static func install(iters: Int) -> UIWindow? {
        guard let device = MTLCreateSystemDefaultDevice() else { return nil }
        let view = MTKView(frame: UIScreen.main.bounds, device: device)
        view.preferredFramesPerSecond = 60
        view.colorPixelFormat = .bgra8Unorm
        guard let load = UILoad(view: view, iters: iters) else { return nil }
        view.delegate = load
        shared = load
        let vc = UIViewController()
        vc.view = view
        let window = UIWindow(frame: UIScreen.main.bounds)
        window.rootViewController = vc
        window.makeKeyAndVisible()
        return window
    }

    private init?(view: MTKView, iters: Int) {
        guard let device = view.device, let queue = device.makeCommandQueue() else { return nil }
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        struct VOut { float4 pos [[position]]; float2 uv; };
        struct P { float t; int iters; };
        vertex VOut vert(uint vid [[vertex_id]]) {
            float2 uv = float2((vid << 1) & 2, vid & 2);
            VOut o; o.pos = float4(uv * 2.0 - 1.0, 0, 1); o.uv = uv; return o;
        }
        fragment float4 frag(VOut in [[stage_in]], constant P& p [[buffer(0)]]) {
            float2 uv = in.uv + float2(0, p.t * 0.25); // scrolls
            float3 c = 0;
            for (int i = 0; i < p.iters; i++) {
                c += 0.5 + 0.5 * cos(p.t + uv.xyx * (3.0 + float(i) * 0.1) + float3(0, 2, 4));
                uv = fract(uv * 1.01 + 0.013);
            }
            return float4(c / float(max(p.iters, 1)), 1);
        }
        """
        do {
            let lib = try device.makeLibrary(source: src, options: nil)
            let desc = MTLRenderPipelineDescriptor()
            desc.vertexFunction = lib.makeFunction(name: "vert")
            desc.fragmentFunction = lib.makeFunction(name: "frag")
            desc.colorAttachments[0].pixelFormat = view.colorPixelFormat
            pipeline = try device.makeRenderPipelineState(descriptor: desc)
        } catch {
            print("ARTEMIS_INFO ui_load_unavailable \(error)")
            return nil
        }
        self.queue = queue
        self.iters = Int32(iters)
        super.init()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let pass = view.currentRenderPassDescriptor, let drawable = view.currentDrawable,
              let cb = queue.makeCommandBuffer(), let enc = cb.makeRenderCommandEncoder(descriptor: pass) else { return }
        struct P { var t: Float; var iters: Int32 }
        var p = P(t: Float(CACurrentMediaTime() - t0), iters: iters)
        enc.setRenderPipelineState(pipeline)
        enc.setFragmentBytes(&p, length: MemoryLayout<P>.stride, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
        drawable.addPresentedHandler { [weak self] d in self?.presented(d.presentedTime) }
        cb.present(drawable)
        cb.commit()
    }

    private func presented(_ t: CFTimeInterval) {
        lock.lock(); defer { lock.unlock() }
        if t == 0 { missed += 1; return } // never shown
        if lastPresented > 0 { intervals.append(t - lastPresented) }
        lastPresented = t
    }

    // Frame stats since the previous drain. An interval of k refresh periods means k-1 dropped frames.
    func drain() -> (frames: Int, dropped: Int, p99ms: Double) {
        lock.lock()
        let xs = intervals, miss = missed
        intervals.removeAll(keepingCapacity: true)
        missed = 0
        lock.unlock()
        guard !xs.isEmpty else { return (0, miss, 0) }
        let dropped = xs.reduce(miss) { $0 + max(0, Int(($1 / Self.period).rounded()) - 1) }
        return (xs.count, dropped, quantile(xs, 0.99) * 1000)
    }
}

func footprintMB() -> Double {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
    let kr = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    return kr == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : -1
}
