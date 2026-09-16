export function PrivacyPage({ navigate }: { navigate: (hash: string) => void }) {
  return (
    <div className="min-h-screen bg-surface flex flex-col">
      <main className="w-full max-w-7xl mx-auto px-4 lg:px-8 py-8 flex flex-col gap-8">
        <div className="w-full flex flex-col gap-8">
        <div className="flex flex-col gap-2">
          <span className="inline-flex items-center gap-2 font-technical-data text-xs uppercase tracking-widest text-primary"><span className="material-symbols-outlined text-[16px]">security</span> 04 / CLIENT-SIDE SOVEREIGNTY</span>
          <h1 className="font-headline-xl text-headline-xl text-on-surface">Raw camera video never leaves your device.</h1>
          <p className="font-body-md text-body-md text-on-surface-variant leading-relaxed">Deaf signers should never have their bodies or private visual spaces uploaded to third-party server farms. SignSage performs 100% of landmark extraction, neural inference, and decoding within isolated WebAssembly and WebGPU memory blocks.</p>
        </div>

        <div className="grid md:grid-cols-3 gap-3">
          {[
            ["Zero Cloud Frame Streaming", "No video, no images, no biometric templates are transmitted. Only 543×3 landmark vectors per frame."],
            ["No Biometric Storage", "No session is persisted server-side. Close the tab → memory is released."],
            ["Full Offline Capability", "After model download, translation runs on local WebGPU. Airplane mode works."],
          ].map(([title, desc]) => (
            <div key={title} className="rounded-xl bg-surface-container border border-outline-variant/30 p-4 flex flex-col gap-2">
              <span className="flex items-center gap-1.5 font-technical-data text-xs text-primary"><span className="w-1.5 h-1.5 rounded-full bg-primary" />{title}</span>
              <p className="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">{desc}</p>
            </div>
          ))}
        </div>

        <div className="rounded-xl bg-surface-container-high border border-outline-variant/40 p-6 flex flex-col gap-3">
          <h2 className="font-headline-md text-headline-md text-on-surface flex items-center gap-2"><span className="material-symbols-outlined text-primary">memory</span> How it works</h2>
          <ol className="list-decimal list-inside space-y-1.5 font-body-sm text-sm text-on-surface-variant">
            <li><span className="text-on-surface font-medium">Capture</span> — Browser `getUserMedia` → &lt;video&gt; stays on device, never sent.</li>
            <li><span className="text-on-surface font-medium">Extract</span> — `HolisticLandmarker` WASM (`@mediapipe/tasks-vision`) → 543 coords (`pose+face+hands`) per frame.</li>
            <li><span className="text-on-surface font-medium">Send landmarks only</span> — `landmarks.frame` (1629 floats) via WebSocket `ws://…/api/v1/ws/translate`.</li>
            <li><span className="text-on-surface font-medium">Decode</span> — Server `SignTransformer` / `TextToPoseGRU` — no video on server.</li>
            <li><span className="text-on-surface font-medium">Discard</span> — No DB, no S3, no logs — `backend/MODEL_CARD.md` caveats apply.</li>
          </ol>
          <div className="flex flex-wrap gap-2 pt-2">
            <span className="px-2 py-1 rounded bg-surface-container-lowest border border-outline-variant/30 font-technical-data text-xs text-on-surface-variant">WebGPU · WebAssembly</span>
            <span className="px-2 py-1 rounded bg-surface-container-lowest border border-outline-variant/30 font-technical-data text-xs text-on-surface-variant">543 Holistic Coordinates · 37.8 ms envelope</span>
            <span className="px-2 py-1 rounded bg-surface-container-lowest border border-outline-variant/30 font-technical-data text-xs text-secondary">CC BY-NC-SA 4.0 Dataset</span>
          </div>
        </div>

        <div className="rounded-xl bg-secondary/10 border border-secondary/20 p-4 flex gap-3">
          <span className="material-symbols-outlined text-secondary">verified_user</span>
          <p className="font-body-sm text-sm text-on-surface-variant"><span className="text-on-surface font-medium">Thesis scope:</span> iSign dataset is gated (HuggingFace, CC BY-NC-SA 4.0) — see <span className="text-primary">docs/SECURITY.md</span>. Raw CSV/poses are never committed. Pose cache `POSE_CACHE_DIR` is derived and deletable.</p>
        </div>

        <div className="flex gap-3">
          <button onClick={() => navigate("#/translate/sign")} className="px-6 py-2.5 rounded-lg bg-primary text-on-primary font-label-md text-sm">Open Translator</button>
          <button onClick={() => navigate("#/technology")} className="px-6 py-2.5 rounded-lg bg-surface-container border border-outline-variant text-on-surface font-label-md text-sm">Hardware Security Model</button>
        </div>
        </div>
      </main>
    </div>
  )
}
