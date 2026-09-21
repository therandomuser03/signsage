export const stitchLandingHtml = `<header class="fixed top-0 left-0 w-full z-50 bg-surface/85 backdrop-blur-md border-b border-outline-variant/30"><div class="h-16 w-full max-w-7xl mx-auto px-margin lg:px-margin-desktop flex items-center justify-between gap-space-md"><div class="flex items-center gap-space-md"><a class="flex items-center gap-space-sm" data-path="landing" href="#"><img alt="SignSage logo" class="h-8 w-auto object-contain" src="/signsage-logo.svg"><span class="font-headline-md text-headline-md tracking-tight text-on-surface">SignSage</span></a><span class="hidden sm:inline-flex items-center px-space-sm py-0.5 rounded border border-outline-variant/40 bg-surface-container-low text-on-surface-variant font-technical-data text-technical-data">M.Tech Research Prototype</span></div><nav class="hidden md:flex items-center gap-space-sm" data-active-classes="bg-surface-container-high text-primary font-medium rounded-lg"><a class="px-space-sm py-1.5 rounded text-on-surface-variant hover:bg-surface-container hover:text-on-surface transition-colors font-label-md text-label-md" data-path="translator" href="#">Translator</a><a class="px-space-sm py-1.5 rounded text-on-surface-variant hover:bg-surface-container hover:text-on-surface transition-colors font-label-md text-label-md" data-path="research-benchmarks" href="#">Research &amp; Benchmarks</a><a class="px-space-sm py-1.5 rounded text-on-surface-variant hover:bg-surface-container hover:text-on-surface transition-colors font-label-md text-label-md" data-path="technology" href="#">Technology</a><a class="px-space-sm py-1.5 rounded text-on-surface-variant hover:bg-surface-container hover:text-on-surface transition-colors font-label-md text-label-md" data-path="privacy" href="#">Privacy</a></nav><div class="flex items-center gap-space-md"><a class="hidden sm:inline-flex items-center justify-center px-space-md py-2 rounded-lg bg-primary text-on-primary font-label-md text-label-md hover:bg-primary-fixed transition-colors shadow-[0_1px_8px_rgba(0,0,0,0.2)]" data-path="translator" href="#">Open Translator</a><div class="w-8 h-8 rounded-full bg-primary flex items-center justify-center"><span class="material-symbols-outlined text-on-primary text-[18px]">person</span></div></div></div></header><main class="w-full pt-16 bg-surface flex-1"><div class="flex flex-col w-full">
<!-- Top Technical Status Bar -->
<section class="w-full bg-surface-container-lowest border-b border-outline-variant/30 py-space-xs px-margin lg:px-margin-desktop">
<div class="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-space-sm font-technical-data text-technical-data text-on-surface-variant">
<div class="flex items-center gap-space-md">
<span class="inline-flex items-center gap-1.5 text-primary">
<span class="w-1.5 h-1.5 rounded-full bg-primary animate-pulse"></span>
<span>Browser: MediaPipe (WebAssembly)</span>
</span>
<span class="hidden md:inline text-outline/50">/</span>
<span class="hidden md:inline text-on-surface">Server: FastAPI + PyTorch over WebSocket</span>
</div>
<div class="flex items-center gap-space-md">
<span class="text-secondary font-medium">Training vocabulary: 17,659 tokens</span>
<span class="inline-flex items-center gap-1 px-space-xs py-0.5 rounded bg-surface-container text-outline text-[11px]">
<span class="material-symbols-outlined text-[14px]">lock</span> Video stays on-device
        </span>
</div>
</div>
</section>
<!-- 1. Hero Section -->
<section class="w-full bg-surface py-space-xl lg:py-24 px-margin lg:px-margin-desktop relative overflow-hidden">
<div class="max-w-7xl mx-auto flex flex-col gap-space-xl">
<!-- Tagline & Core Statement -->
<div class="flex flex-col items-start gap-space-md max-w-4xl">
<div class="inline-flex items-center gap-2 px-space-sm py-1 rounded bg-surface-container-high border border-outline-variant/40">
<span class="w-2 h-2 rounded-full bg-primary"></span>
<span class="font-technical-data text-technical-data tracking-wider uppercase text-primary">M.Tech Research Initiative • Bidirectional ISL</span>
</div>
<h1 class="font-headline-xl text-headline-xl text-on-surface tracking-tight leading-[1.08]">
          ISL ↔ English, as a research prototype.<br>
<span class="text-on-surface-variant">A bounded, honest demo — not a production translator.</span>
</h1>
<p class="font-body-lg text-body-lg text-on-surface-variant max-w-3xl leading-relaxed">
          SignSage is an M.Tech research project exploring bidirectional translation between Indian Sign Language (ISL) and English: MediaPipe Holistic tracks hand/body/face landmarks in your browser, and a small PyTorch backend attempts sign→text and text→pose translation. The underlying research models are Phase 0 checkpoints with real, disclosed limitations (see Research &amp; Benchmarks) — a fixed set of ~10 phrases has a reliable recorded fallback, but general open-vocabulary translation is not solved by this project.
        </p>
<!-- CTA Cluster -->
<div class="flex flex-wrap items-center gap-space-md pt-space-sm">
<a class="inline-flex items-center gap-2 px-space-lg py-3 rounded-lg bg-primary text-on-primary font-label-md text-label-md hover:bg-primary-fixed transition-colors shadow-sm" data-path="translator" href="#">
<span>Open Live Translator</span>
<span class="material-symbols-outlined text-[18px]">arrow_forward</span>
</a>
<a class="inline-flex items-center gap-2 px-space-lg py-3 rounded-lg bg-surface-container-high border border-outline-variant/50 text-on-surface hover:border-secondary hover:text-secondary-fixed transition-colors font-label-md text-label-md" href="#benchmarks">
<span class="material-symbols-outlined text-[18px]">article</span>
<span>Read Research Results</span>
</a>
<span class="font-technical-data text-technical-data text-outline pl-2 hidden sm:inline">
            Requires camera + microphone permission
          </span>
</div>
</div>
<!-- Facts Micro-Row (static, not live telemetry) -->
<div class="grid grid-cols-2 md:grid-cols-4 gap-space-sm pt-space-xs font-technical-data text-technical-data">
<div class="bg-surface-container-low p-space-sm rounded border border-outline-variant/20 flex flex-col">
<span class="text-outline text-[11px] uppercase tracking-wider">Landmark Extraction</span>
<span class="text-on-surface font-semibold flex items-center gap-1.5 pt-0.5">
<span class="w-1.5 h-1.5 rounded-full bg-primary"></span> Browser (MediaPipe)
          </span>
</div>
<div class="bg-surface-container-low p-space-sm rounded border border-outline-variant/20 flex flex-col">
<span class="text-outline text-[11px] uppercase tracking-wider">Translation Inference</span>
<span class="text-primary font-semibold pt-0.5">Server (PyTorch, WebSocket)</span>
</div>
<div class="bg-surface-container-low p-space-sm rounded border border-outline-variant/20 flex flex-col">
<span class="text-outline text-[11px] uppercase tracking-wider">Spatial Landmarks</span>
<span class="text-on-surface font-semibold pt-0.5">543 Holistic Coordinates</span>
</div>
<div class="bg-surface-container-low p-space-sm rounded border border-outline-variant/20 flex flex-col">
<span class="text-outline text-[11px] uppercase tracking-wider">Video Transmitted</span>
<span class="text-secondary font-semibold pt-0.5">Never — landmarks only</span>
</div>
</div>
<!-- Workbench Preview -->
<div class="w-full bg-surface-container-lowest rounded-xl border border-outline-variant/40 shadow-xl overflow-hidden mt-space-md">
<div class="px-space-md py-2.5 bg-surface-container-low border-b border-outline-variant/30 flex items-center justify-between">
<div class="flex items-center gap-2">
<div class="w-2.5 h-2.5 rounded-full bg-outline-variant/70"></div>
<div class="w-2.5 h-2.5 rounded-full bg-outline-variant/70"></div>
<div class="w-2.5 h-2.5 rounded-full bg-outline-variant/70"></div>
<span class="ml-2 font-technical-data text-[12px] text-outline">SIGNSAGE_WORKBENCH // TRANSLATOR</span>
</div>
</div>
<div class="p-space-lg flex flex-col gap-space-sm">
<p class="font-body-sm text-body-sm text-on-surface-variant">Open the translator to sign into your camera (ISL → English) or type a supported phrase to see a recorded ISL clip play back (English → ISL). Both directions are bounded demos over a fixed phrase set, clearly labelled as such in the app.</p>
</div>
</div>
</div>
</section>
<!-- 2. Pipeline Steps -->
<section class="w-full bg-surface-container-lowest py-space-xl lg:py-24 px-margin lg:px-margin-desktop border-t border-outline-variant/30">
<div class="max-w-7xl mx-auto flex flex-col gap-space-xl">
<div class="flex flex-col gap-space-xs max-w-3xl">
<div class="font-technical-data text-technical-data text-primary uppercase tracking-wider">
          01 / HOW IT WORKS
        </div>
<h2 class="font-headline-lg text-headline-lg text-on-surface">
          Four steps, each with a stated limitation.
        </h2>
</div>
<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-gutter-desktop">
<!-- 01 Capture -->
<div class="bg-surface-container p-space-lg rounded-xl border border-outline-variant/30 flex flex-col justify-between hover:border-outline transition-colors">
<div class="flex flex-col gap-space-sm">
<div class="flex items-center justify-between">
<span class="font-technical-metric text-technical-metric text-primary">01</span>
<span class="material-symbols-outlined text-outline">videocam</span>
</div>
<h3 class="font-headline-md text-headline-md text-on-surface">Capture</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
              Your browser's camera feed is read locally by MediaPipe Holistic. The video frame is never uploaded — only the 543 extracted landmark coordinates per frame leave the browser.
            </p>
</div>
<div class="pt-space-md mt-space-md border-t border-outline-variant/20 font-technical-data text-[11px] text-outline">
            543 landmarks × 3 coords = 1629 values/frame
          </div>
</div>
<!-- 02 Normalize -->
<div class="bg-surface-container p-space-lg rounded-xl border border-outline-variant/30 flex flex-col justify-between hover:border-outline transition-colors">
<div class="flex flex-col gap-space-sm">
<div class="flex items-center justify-between">
<span class="font-technical-metric text-technical-metric text-secondary">02</span>
<span class="material-symbols-outlined text-outline">tune</span>
</div>
<h3 class="font-headline-md text-headline-md text-on-surface">Normalize &amp; Buffer</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
              Landmarks are hip-centered and scale-normalized, then buffered over a sliding window until a sentence boundary (a pause) is detected.
            </p>
</div>
<div class="pt-space-md mt-space-md border-t border-outline-variant/20 font-technical-data text-[11px] text-outline">
            Up to 160 frames per sentence window
          </div>
</div>
<!-- 03 Translate -->
<div class="bg-surface-container p-space-lg rounded-xl border border-outline-variant/30 flex flex-col justify-between hover:border-outline transition-colors">
<div class="flex flex-col gap-space-sm">
<div class="flex items-center justify-between">
<span class="font-technical-metric text-technical-metric text-primary">03</span>
<span class="material-symbols-outlined text-outline">translate</span>
</div>
<h3 class="font-headline-md text-headline-md text-on-surface">Translate</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
              A small Transformer attempts sign→text decoding. Its own evaluation (see Research &amp; Benchmarks) shows a mode-collapse problem on the current dataset, so a distance-based match against ~10 recorded reference phrases is used as an honestly-labelled fallback when the model's output looks unreliable.
            </p>
</div>
<div class="pt-space-md mt-space-md border-t border-outline-variant/20 font-technical-data text-[11px] text-outline">
            Status: NO-GO (research finding, not hidden)
          </div>
</div>
<!-- 04 Respond -->
<div class="bg-surface-container p-space-lg rounded-xl border border-outline-variant/30 flex flex-col justify-between hover:border-outline transition-colors">
<div class="flex flex-col gap-space-sm">
<div class="flex items-center justify-between">
<span class="font-technical-metric text-technical-metric text-secondary">04</span>
<span class="material-symbols-outlined text-outline">record_voice_over</span>
</div>
<h3 class="font-headline-md text-headline-md text-on-surface">Respond</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
              English text is mapped to an ISL gloss and, for the same bounded phrase set, a real recorded hand-motion clip plays on a skeleton canvas. Outside that set, a generative pose model runs instead — it is currently near-static, and the UI says so.
            </p>
</div>
<div class="pt-space-md mt-space-md border-t border-outline-variant/20 font-technical-data text-[11px] text-outline">
            Canvas 2D render • no avatar/3D model
          </div>
</div>
</div>
</div>
</section>
<!-- 3. Bidirectional Section -->
<section class="w-full bg-surface py-space-xl lg:py-24 px-margin lg:px-margin-desktop">
<div class="max-w-7xl mx-auto flex flex-col gap-space-xl">
<div class="flex flex-col gap-space-xs max-w-3xl">
<div class="font-technical-data text-technical-data text-secondary uppercase tracking-wider">
          02 / BOTH DIRECTIONS, BOTH BOUNDED
        </div>
<h2 class="font-headline-lg text-headline-lg text-on-surface">
          Sign-to-text and text-to-sign share the same honesty rules.
        </h2>
<p class="font-body-md text-body-md text-on-surface-variant">
          Neither direction claims open-vocabulary fluency. Each has a labelled bounded/demo path for a fixed phrase set, and a labelled research-model path for everything else.
        </p>
</div>
<div class="grid grid-cols-1 lg:grid-cols-2 gap-gutter-desktop">
<!-- Direction A: ISL -> English -->
<div class="bg-surface-container p-space-xl rounded-xl border border-outline-variant/30 flex flex-col gap-space-md relative overflow-hidden">
<div class="flex items-center justify-between border-b border-outline-variant/20 pb-space-sm">
<div class="flex items-center gap-2">
<span class="px-2 py-0.5 rounded bg-primary text-on-primary font-technical-data text-[11px] font-semibold">DIRECTION A</span>
<span class="font-headline-md text-headline-md text-on-surface">ISL → English</span>
</div>
<span class="material-symbols-outlined text-primary text-[22px]">sign_language</span>
</div>
<p class="font-body-md text-body-md text-on-surface-variant leading-relaxed">
            Sign in front of your camera. Landmarks stream to the backend over WebSocket; when you pause, the buffered sentence is sent for translation and a caption is returned.
          </p>
<div class="bg-surface-container-low p-space-md rounded-lg border border-outline-variant/20 flex flex-col gap-space-sm font-technical-data text-technical-data">
<div class="flex items-center justify-between text-outline text-[11px]">
<span>SUPPORTED DEMO PHRASES</span>
<span>BOUNDED FALLBACK</span>
</div>
<div class="flex items-center justify-between bg-surface-container-lowest p-space-sm rounded">
<span class="text-primary font-semibold">Hello, Thank you, Yes, No, Help…</span>
<span class="text-on-surface text-right font-body-sm text-body-sm font-normal">matched via DTW distance</span>
</div>
<div class="flex items-center justify-between text-[11px] text-outline">
<span>Outside the phrase set: research Transformer</span>
<span>Labelled "model" vs "demo" in the UI</span>
</div>
</div>
<ul class="flex flex-col gap-2 font-body-sm text-body-sm text-on-surface-variant pt-space-xs">
<li class="flex items-start gap-2">
<span class="material-symbols-outlined text-primary text-[18px]">check</span>
<span>Ten recorded reference phrases give a reliable bounded demo.</span>
</li>
<li class="flex items-start gap-2">
<span class="material-symbols-outlined text-tertiary text-[18px]">warning</span>
<span>Open-vocabulary continuous signing is not reliable — the research checkpoint mode-collapsed in evaluation.</span>
</li>
</ul>
</div>
<!-- Direction B: English -> ISL -->
<div class="bg-surface-container p-space-xl rounded-xl border border-outline-variant/30 flex flex-col gap-space-md relative overflow-hidden">
<div class="flex items-center justify-between border-b border-outline-variant/20 pb-space-sm">
<div class="flex items-center gap-2">
<span class="px-2 py-0.5 rounded bg-secondary text-on-secondary font-technical-data text-[11px] font-semibold">DIRECTION B</span>
<span class="font-headline-md text-headline-md text-on-surface">English → ISL</span>
</div>
<span class="material-symbols-outlined text-secondary text-[22px]">smart_toy</span>
</div>
<p class="font-body-md text-body-md text-on-surface-variant leading-relaxed">
            Type English text. If it matches one of ten supported phrases (exactly, or a close rephrasing), a real recorded hand-motion clip for that phrase plays back on the pose canvas.
          </p>
<div class="bg-surface-container-low p-space-md rounded-lg border border-outline-variant/20 flex flex-col gap-space-sm font-technical-data text-technical-data">
<div class="flex items-center justify-between text-outline text-[11px]">
<span>TEXT INPUT</span>
<span>RENDERED OUTPUT</span>
</div>
<div class="flex items-center justify-between bg-surface-container-lowest p-space-sm rounded">
<span class="text-on-surface font-body-sm text-body-sm">"Thank you"</span>
<span class="text-secondary font-semibold text-right">Recorded clip → hands-only skeleton</span>
</div>
<div class="flex items-center justify-between text-[11px] text-outline">
<span>Unsupported phrase</span>
<span>Falls back to generative model (near-static)</span>
</div>
</div>
<ul class="flex flex-col gap-2 font-body-sm text-body-sm text-on-surface-variant pt-space-xs">
<li class="flex items-start gap-2">
<span class="material-symbols-outlined text-secondary text-[18px]">check</span>
<span>Bounded phrase lookup, not general English→ISL generation.</span>
</li>
<li class="flex items-start gap-2">
<span class="material-symbols-outlined text-tertiary text-[18px]">warning</span>
<span>Only hand motion was reliably captured for the recorded clips; the skeleton view shows hands, not a full body pose.</span>
</li>
</ul>
</div>
</div>
</div>
</section>
<!-- 4. Research & Benchmarks -->
<section class="w-full bg-surface-container-lowest py-space-xl lg:py-24 px-margin lg:px-margin-desktop border-t border-outline-variant/30" id="benchmarks">
<div class="max-w-7xl mx-auto flex flex-col gap-space-xl">
<div class="flex flex-col md:flex-row md:items-end justify-between gap-space-md">
<div class="flex flex-col gap-space-xs max-w-2xl">
<div class="font-technical-data text-technical-data text-primary uppercase tracking-wider">
            03 / RESEARCH RESULTS (PHASE 0)
          </div>
<h2 class="font-headline-lg text-headline-lg text-on-surface">
            Measured on a held-out split of the iSign v1.1 dataset — not a public benchmark.
          </h2>
</div>
<div class="font-technical-data text-technical-data text-outline">
          Run 20260916T144755Z • see /research/outputs/phase0/RESULTS.md
        </div>
</div>
<div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-gutter-desktop">
<div class="bg-surface-container p-space-lg rounded-xl border border-outline-variant/30 flex flex-col gap-2">
<span class="font-technical-data text-[11px] uppercase tracking-wider text-outline">Word Error Rate (WER)</span>
<div class="flex items-baseline gap-2">
<span class="font-headline-xl text-headline-xl text-primary font-bold">~1.0–1.03</span>
<span class="font-technical-data text-[12px] text-primary">Test split, all models</span>
</div>
<p class="font-body-sm text-body-sm text-on-surface-variant pt-2 border-t border-outline-variant/20">
            A WER at or above 1.0 means the output is, on average, as long to correct as retyping it from scratch. This is a disclosed failure result, not a success metric.
          </p>
</div>
<div class="bg-surface-container p-space-lg rounded-xl border border-outline-variant/30 flex flex-col gap-2">
<span class="font-technical-data text-[11px] uppercase tracking-wider text-outline">BLEU (Sign→Text)</span>
<div class="flex items-baseline gap-2">
<span class="font-headline-xl text-headline-xl text-secondary font-bold">0.16–0.31</span>
<span class="font-technical-data text-[12px] text-secondary">Across model variants</span>
</div>
<p class="font-body-sm text-body-sm text-on-surface-variant pt-2 border-t border-outline-variant/20">
            Low-to-moderate n-gram overlap. The Transformer and GRU baselines both mode-collapsed to a narrow set of outputs during evaluation.
          </p>
</div>
<div class="bg-surface-container p-space-lg rounded-xl border border-outline-variant/30 flex flex-col gap-2">
<span class="font-technical-data text-[11px] uppercase tracking-wider text-outline">Text→Pose Motion Energy</span>
<div class="flex items-baseline gap-2">
<span class="font-headline-xl text-headline-xl text-on-surface font-bold">~7.9%</span>
<span class="font-technical-data text-[12px] text-primary">of ground-truth motion</span>
</div>
<p class="font-body-sm text-body-sm text-on-surface-variant pt-2 border-t border-outline-variant/20">
            The generative pose model is close to static output. That's why supported phrases use a recorded clip instead.
          </p>
</div>
<div class="bg-surface-container p-space-lg rounded-xl border border-outline-variant/30 flex flex-col gap-2">
<span class="font-technical-data text-[11px] uppercase tracking-wider text-outline">Training Vocabulary</span>
<div class="flex items-baseline gap-2">
<span class="font-headline-xl text-headline-xl text-on-surface font-bold">17,659</span>
<span class="font-technical-data text-[12px] text-outline">tokens</span>
</div>
<p class="font-body-sm text-body-sm text-on-surface-variant pt-2 border-t border-outline-variant/20">
            From a 13,000-row subset of iSign v1.1 (122,925 usable rows total), split by video to avoid leakage.
          </p>
</div>
</div>
<div class="grid grid-cols-1 lg:grid-cols-12 gap-gutter-desktop">
<div class="lg:col-span-8 bg-surface-container p-space-xl rounded-xl border border-outline-variant/30 flex flex-col gap-space-md">
<div class="flex items-center justify-between border-b border-outline-variant/20 pb-space-sm">
<h3 class="font-headline-md text-headline-md text-on-surface">Actual Model Architecture</h3>
<span class="font-technical-data text-technical-data text-primary">PyTorch, CPU or GPU</span>
</div>
<div class="grid grid-cols-1 md:grid-cols-3 gap-space-md">
<div class="flex flex-col gap-1">
<span class="font-technical-data text-technical-data text-on-surface font-semibold">Landmark Extraction</span>
<p class="font-body-sm text-body-sm text-on-surface-variant">
                MediaPipe Holistic, running as WebAssembly in the browser. No custom shaders, no WebGPU.
              </p>
</div>
<div class="flex flex-col gap-1">
<span class="font-technical-data text-technical-data text-on-surface font-semibold">Sign→Text</span>
<p class="font-body-sm text-body-sm text-on-surface-variant">
                A small sequence-to-sequence Transformer encoder-decoder. A separate GRU baseline was also trained for comparison and is not served (it fully mode-collapsed).
              </p>
</div>
<div class="flex flex-col gap-1">
<span class="font-technical-data text-technical-data text-on-surface font-semibold">Text→Pose</span>
<p class="font-body-sm text-body-sm text-on-surface-variant">
                A GRU decoder trained to regress landmark sequences from a gloss. Near-static in evaluation; see the demo-phrase fallback above.
              </p>
</div>
</div>
</div>
<div class="lg:col-span-4 bg-surface-container-low p-space-xl rounded-xl border border-outline-variant/30 flex flex-col justify-between">
<div class="flex flex-col gap-space-xs">
<span class="font-technical-data text-technical-data text-secondary uppercase">Dataset</span>
<h4 class="font-headline-md text-headline-md text-on-surface">iSign v1.1</h4>
<p class="font-body-sm text-body-sm text-on-surface-variant">
              A public, gated ISL dataset (CC BY-NC-SA 4.0, hosted on Hugging Face) — not a corpus collected for this project, and no dialect-specific breakdown is claimed.
            </p>
</div>
<div class="pt-space-md">
<a class="inline-flex items-center gap-1 font-label-md text-label-md text-primary hover:text-primary-fixed" data-path="research-benchmarks" href="#">
<span>Read the full results</span>
<span class="material-symbols-outlined text-[16px]">chevron_right</span>
</a>
</div>
</div>
</div>
</div>
</section>
<!-- 5. Privacy -->
<section id="privacy" class="w-full bg-surface py-space-xl lg:py-24 px-margin lg:px-margin-desktop border-t border-outline-variant/30">
<div class="max-w-7xl mx-auto">
<div class="bg-surface-container-high rounded-xl p-space-xl lg:p-12 border border-outline-variant/40 flex flex-col lg:flex-row items-start lg:items-center justify-between gap-space-xl">
<div class="flex flex-col gap-space-sm max-w-2xl">
<div class="inline-flex items-center gap-2 font-technical-data text-technical-data text-primary">
<span class="material-symbols-outlined text-[18px]">security</span>
<span class="uppercase tracking-wider">04 / DATA HANDLING</span>
</div>
<h2 class="font-headline-lg text-headline-lg text-on-surface">
            Raw camera video never leaves your device.
          </h2>
<p class="font-body-md text-body-md text-on-surface-variant leading-relaxed">
            Landmark extraction runs in your browser. Only landmark coordinates — never video — are sent to the backend for translation. That does mean a live connection to the backend is required; there is no offline mode.
          </p>
<div class="flex flex-wrap items-center gap-space-md pt-space-xs font-technical-data text-technical-data text-outline">
<span class="flex items-center gap-1.5"><span class="w-1.5 h-1.5 rounded-full bg-primary"></span>No video/image transmission</span>
<span class="flex items-center gap-1.5"><span class="w-1.5 h-1.5 rounded-full bg-primary"></span>No server-side session storage by default</span>
<span class="flex items-center gap-1.5"><span class="w-1.5 h-1.5 rounded-full bg-primary"></span>Requires a live backend connection</span>
</div>
</div>
<div class="flex flex-col sm:flex-row lg:flex-col gap-space-sm w-full lg:w-auto shrink-0">
<a class="inline-flex items-center justify-center gap-2 px-space-lg py-3 rounded-lg bg-surface-container-lowest border border-outline-variant/50 text-on-surface hover:border-primary font-label-md text-label-md transition-colors text-center" data-path="privacy" href="#">
<span class="material-symbols-outlined text-[18px]">verified_user</span>
<span>Read the Privacy Page</span>
</a>
<a class="inline-flex items-center justify-center gap-2 px-space-lg py-3 rounded-lg bg-transparent text-outline hover:text-on-surface font-label-md text-label-md transition-colors text-center" data-path="technology" href="#">
<span>Technology Overview</span>
<span class="material-symbols-outlined text-[18px]">arrow_forward</span>
</a>
</div>
</div>
</div>
</section>
<!-- 6. Limitations -->
<section class="w-full bg-surface-container-lowest py-space-xl lg:py-20 px-margin lg:px-margin-desktop border-t border-outline-variant/30">
<div class="max-w-7xl mx-auto flex flex-col gap-space-lg">
<div class="flex flex-col gap-space-xs max-w-3xl">
<div class="font-technical-data text-technical-data text-outline uppercase tracking-wider">
          05 / LIMITATIONS &amp; RESPONSIBLE RESEARCH DISCLOSURES
        </div>
<h2 class="font-headline-md text-headline-md text-on-surface">
          Transparent scientific caveats and operational boundaries.
        </h2>
</div>
<div class="grid grid-cols-1 md:grid-cols-3 gap-gutter-desktop">
<div class="p-space-lg rounded-lg bg-surface-container border border-outline-variant/30 flex flex-col gap-2">
<div class="flex items-center gap-2 text-tertiary font-technical-data text-[12px] font-semibold">
<span class="material-symbols-outlined text-[18px]">warning</span>
<span>RESEARCH PROTOTYPE STATUS</span>
</div>
<p class="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
            This system is an academic research instrument built for an M.Tech project. It is <strong class="text-on-surface font-medium">not certified</strong> for medical, legal, or safety-critical communication.
          </p>
</div>
<div class="p-space-lg rounded-lg bg-surface-container border border-outline-variant/30 flex flex-col gap-2">
<div class="flex items-center gap-2 text-secondary font-technical-data text-[12px] font-semibold">
<span class="material-symbols-outlined text-[18px]">psychology</span>
<span>PHASE 0 RESULT: NO-GO</span>
</div>
<p class="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
            The project's own evaluation concluded the current research checkpoints are not ready for general use (mode collapse on Sign→Text, near-static Text→Pose). This is stated here rather than hidden behind favorable-looking numbers.
          </p>
</div>
<div class="p-space-lg rounded-lg bg-surface-container border border-outline-variant/30 flex flex-col gap-2">
<div class="flex items-center gap-2 text-primary font-technical-data text-[12px] font-semibold">
<span class="material-symbols-outlined text-[18px]">diversity_3</span>
<span>HUMAN INTERPRETER AUGMENTATION</span>
</div>
<p class="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
            SignSage does not claim to replace certified human ISL interpreters. It is a bounded technical demonstration for casual, non-critical use.
          </p>
</div>
</div>
</div>
</section>
<!-- 7. Final CTA -->
<section class="w-full bg-surface py-space-xl lg:py-28 px-margin lg:px-margin-desktop border-t border-outline-variant/30">
<div class="max-w-4xl mx-auto flex flex-col items-center text-center gap-space-md">
<div class="inline-flex items-center gap-2 px-space-sm py-1 rounded bg-surface-container-high border border-outline-variant/40 font-technical-data text-technical-data text-primary">
<span class="w-2 h-2 rounded-full bg-primary"></span>
<span>OPEN SOURCE ACADEMIC PROJECT</span>
</div>
<h2 class="font-headline-xl text-headline-xl text-on-surface">
        Try the research prototype.
      </h2>
<p class="font-body-lg text-body-lg text-on-surface-variant max-w-2xl leading-relaxed">
        Run bidirectional Indian Sign Language translation in your browser using your webcam. No extra sensors or installs required — just camera permission and a running backend.
      </p>
<div class="flex flex-wrap items-center justify-center gap-space-md pt-space-sm">
<a class="inline-flex items-center gap-2 px-space-xl py-3.5 rounded-lg bg-primary text-on-primary font-label-md text-label-md hover:bg-primary-fixed transition-colors shadow-md" data-path="translator" href="#">
<span class="material-symbols-outlined text-[20px]">videocam</span>
<span>Launch SignSage Translator</span>
</a>
<a class="inline-flex items-center gap-2 px-space-lg py-3.5 rounded-lg bg-surface-container-high border border-outline-variant/50 text-on-surface hover:border-secondary hover:text-secondary-fixed transition-colors font-label-md text-label-md" data-path="research-benchmarks" href="#">
<span class="material-symbols-outlined text-[20px]">science</span>
<span>View Research Results &amp; Source</span>
</a>
</div>
<div class="pt-space-md flex flex-wrap items-center justify-center gap-space-lg font-technical-data text-technical-data text-outline">
<span>Framework: PyTorch + FastAPI (server) / MediaPipe (browser)</span>
<span>•</span>
<span>Dataset License: CC BY-NC-SA 4.0 (iSign v1.1)</span>
<span>•</span>
<span>Status: Research Phase 0 (NO-GO)</span>
</div>
</div>
</section>
</div>
</main><footer class="w-full bg-surface-container-lowest border-t border-outline-variant/30 py-space-xl"><div class="w-full max-w-7xl mx-auto px-margin lg:px-margin-desktop"><div class="grid grid-cols-1 md:grid-cols-12 gap-space-xl pb-space-xl border-b border-outline-variant/20"><div class="md:col-span-5 flex flex-col gap-space-sm"><div class="flex items-center gap-space-sm"><span class="font-headline-md text-headline-md tracking-tight text-on-surface">SignSage</span></div><p class="font-body-sm text-body-sm text-on-surface-variant max-w-md">An M.Tech research project on bidirectional translation between Indian Sign Language (ISL) and English using camera-based landmark tracking. A research prototype, not a certified product.</p><div class="pt-space-xs"><span class="inline-flex items-center gap-2 font-technical-data text-technical-data text-outline"><span class="w-2 h-2 rounded-full bg-primary"></span>Video stays on-device — only landmark coordinates are sent</span></div></div><div class="md:col-span-4 flex flex-col gap-space-xs"><span class="font-label-sm text-label-sm uppercase tracking-wider text-outline">Research Status</span><p class="font-technical-data text-technical-data text-on-surface-variant leading-relaxed bg-surface-container-low p-space-sm rounded border border-outline-variant/20">Phase 0 evaluation on iSign v1.1 (run 20260916T144755Z): NO-GO. Sign→Text and Text→Pose research checkpoints are not reliable outside the bounded demo phrase set. Full results in research/outputs/phase0/RESULTS.md.</p></div><div class="md:col-span-3 flex flex-col gap-space-xs"><span class="font-label-sm text-label-sm uppercase tracking-wider text-outline">Governance &amp; Open Science</span><div class="flex flex-col gap-space-xs font-body-sm text-body-sm text-on-surface-variant"><p>iSign dataset: CC BY-NC-SA 4.0</p><a class="hover:text-on-surface transition-colors underline decoration-outline-variant underline-offset-4" data-path="privacy" href="#">Privacy &amp; Data Handling</a><a class="hover:text-on-surface transition-colors underline decoration-outline-variant underline-offset-4" data-path="research-benchmarks" href="#">Research Results</a></div></div></div><div class="pt-space-lg flex flex-col sm:flex-row items-center justify-between gap-space-md text-on-surface-variant font-technical-data text-technical-data"><p>&copy; ${new Date().getFullYear()} SignSage — M.Tech academic project.</p><p class="text-outline">Landmarks-only transport • Non-commercial research release</p></div></div></footer>`;
