export const stitchBenchmarksHtml = `<header class="fixed top-0 left-0 right-0 z-50 bg-surface-container-lowest border-b border-outline-variant"><div class="h-16 w-full px-margin-desktop flex items-center justify-between gap-space-md"><div class="flex items-center gap-space-md shrink-0"><img alt="SignSage logo" class="h-8 w-auto object-contain" src="/signsage-logo.svg"><span class="font-headline-md text-headline-md tracking-tight text-on-surface">SignSage</span><span class="font-technical-data text-technical-data px-space-sm py-0.5 rounded bg-surface-container-high text-on-surface-variant border border-outline-variant tracking-wider uppercase">RESEARCH PROTOTYPE</span></div><nav class="hidden xl:flex items-center gap-space-lg h-full" data-active-classes="text-primary border-b-2 border-primary"><a class="h-full flex items-center px-space-xs font-label-md text-label-md text-on-surface-variant hover:text-on-surface transition-colors" data-path="translator" href="#">Translator</a><a aria-current="page" class="h-full flex items-center px-space-xs font-label-md transition-colors text-primary border-b-2 border-primary" data-path="research-and-benchmarks" href="#">Research &amp; Benchmarks</a><a class="h-full flex items-center px-space-xs font-label-md text-label-md text-on-surface-variant hover:text-on-surface transition-colors" data-path="system-architecture" href="#">Technology</a></nav><div class="flex items-center gap-space-md shrink-0"><div class="hidden md:flex items-center gap-space-xs px-space-sm py-1 rounded bg-surface-container-low border border-outline-variant"><span class="material-symbols-outlined text-[14px] text-primary">lock</span><span class="font-technical-data text-technical-data text-on-surface-variant">Video stays on-device</span></div><div class="w-8 h-8 rounded-full bg-primary flex items-center justify-center shrink-0"><span class="material-symbols-outlined text-on-primary text-[18px]">person</span></div></div></div></header><main class="w-full pt-16 bg-surface-container-lowest min-h-[calc(100vh-4rem)]"><div class="flex flex-col w-full">
<div class="w-full px-margin-desktop py-space-xl max-w-7xl mx-auto flex flex-col gap-space-xl">
<!-- Header -->
<section class="flex flex-col gap-space-md">
<div class="flex flex-wrap items-center gap-space-xs text-error font-technical-data text-technical-data uppercase tracking-wider">
<span class="flex h-2 w-2 rounded-full bg-error animate-pulse"></span>
<span>PHASE 0 RESULT: NO-GO</span>
<span class="text-on-surface-variant">·</span>
<span class="text-on-surface-variant">RUN 20260916T144755Z</span>
</div>
<div class="flex flex-col xl:flex-row xl:items-end justify-between gap-space-lg">
<div class="max-w-4xl flex flex-col gap-space-sm">
<h1 class="font-headline-xl text-headline-xl text-on-surface tracking-tight">
            Research Results: what was measured, and why it isn't good enough yet
          </h1>
<p class="font-body-lg text-body-lg text-on-surface-variant leading-relaxed">
            This page reports the actual outcome of Phase 0 of an M.Tech research project on bidirectional ISL↔English translation, evaluated on a held-out split of the public iSign v1.1 dataset. The project's own conclusion is <strong class="text-error">NO-GO</strong>: the trained checkpoints are not reliable for general translation. That result is reported here rather than hidden.
          </p>
</div>
<div class="flex flex-wrap xl:flex-col items-start xl:items-end gap-space-xs shrink-0">
<div class="flex items-center gap-space-xs px-space-sm py-1 rounded bg-surface-container-high text-on-surface font-technical-data text-technical-data">
<span class="text-outline">DATASET:</span>
<span class="text-primary font-semibold">iSign v1.1 (public, CC BY-NC-SA 4.0)</span>
</div>
<div class="flex items-center gap-space-xs px-space-sm py-1 rounded bg-surface-container-high text-on-surface font-technical-data text-technical-data">
<span class="text-outline">ARCH:</span>
<span>MediaPipe Holistic → small Transformer / GRU</span>
</div>
<div class="flex items-center gap-space-xs px-space-sm py-1 rounded bg-surface-container-high text-on-surface font-technical-data text-technical-data">
<span class="text-outline">STAGE:</span>
<span class="text-error">Phase 0 — research checkpoint, not production</span>
</div>
</div>
</div>
<!-- Quick facts -->
<div class="grid grid-cols-2 sm:grid-cols-4 gap-space-sm pt-space-xs">
<div class="bg-surface-container-low p-space-md rounded flex flex-col justify-between">
<span class="font-technical-data text-technical-data text-on-surface-variant uppercase">Landmark Vector</span>
<div class="flex items-baseline gap-space-xs mt-space-xs">
<span class="font-headline-md text-headline-md text-on-surface">543</span>
<span class="font-technical-data text-technical-data text-outline">3D points/frame</span>
</div>
<span class="font-technical-data text-technical-data text-outline mt-space-xs">pose(33)+face(468)+hands(21+21)</span>
</div>
<div class="bg-surface-container-low p-space-md rounded flex flex-col justify-between">
<span class="font-technical-data text-technical-data text-on-surface-variant uppercase">Sign→Text WER (best)</span>
<div class="flex items-baseline gap-space-xs mt-space-xs">
<span class="font-headline-md text-headline-md text-error">1.015</span>
<span class="font-technical-data text-technical-data text-outline">GRU checkpoint</span>
</div>
<span class="font-technical-data text-technical-data text-error mt-space-xs">≥1.0 = worse than retyping</span>
</div>
<div class="bg-surface-container-low p-space-md rounded flex flex-col justify-between">
<span class="font-technical-data text-technical-data text-on-surface-variant uppercase">Transformer Unique Outputs</span>
<div class="flex items-baseline gap-space-xs mt-space-xs">
<span class="font-headline-md text-headline-md text-error">3.9%</span>
<span class="font-technical-data text-technical-data text-outline">of test set</span>
</div>
<span class="font-technical-data text-technical-data text-error mt-space-xs">mode collapse</span>
</div>
<div class="bg-surface-container-low p-space-md rounded flex flex-col justify-between">
<span class="font-technical-data text-technical-data text-on-surface-variant uppercase">Text→Pose Motion Energy</span>
<div class="flex items-baseline gap-space-xs mt-space-xs">
<span class="font-headline-md text-headline-md text-error">7.9%</span>
<span class="font-technical-data text-technical-data text-outline">of ground truth</span>
</div>
<span class="font-technical-data text-technical-data text-error mt-space-xs">near-static output</span>
</div>
</div>
</section>
<!-- Pipeline -->
<section class="flex flex-col gap-space-md">
<div class="flex flex-col gap-space-xs">
<span class="font-technical-data text-technical-data text-primary uppercase">MODULE 01 — WHAT ACTUALLY RUNS</span>
<h2 class="font-headline-lg text-headline-lg text-on-surface">Four real stages, each with a stated limitation</h2>
<p class="font-body-md text-body-md text-on-surface-variant max-w-3xl">
          No claims below are aspirational — this is what the shipped code does, and where it currently falls short.
        </p>
</div>
<div class="grid grid-cols-1 lg:grid-cols-4 gap-space-md">
<div class="bg-surface-container-low p-space-lg rounded flex flex-col justify-between">
<div class="flex flex-col gap-space-sm">
<div class="flex items-center justify-between">
<span class="font-technical-data text-technical-data px-space-xs py-0.5 rounded bg-surface-container-high text-primary font-semibold">STAGE 01</span>
<span class="material-symbols-outlined text-outline text-[20px]">view_in_ar</span>
</div>
<h3 class="font-headline-md text-headline-md text-on-surface">Landmark Extraction</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant">
              MediaPipe Holistic runs as WebAssembly in the browser, producing 543 3D landmarks/frame (face 468, pose 33, hands 21+21). No custom shaders, no WebGPU.
            </p>
</div>
<div class="mt-space-lg font-technical-data text-technical-data text-outline uppercase tracking-wider">Runs client-side only</div>
</div>
<div class="bg-surface-container-low p-space-lg rounded flex flex-col justify-between">
<div class="flex flex-col gap-space-sm">
<div class="flex items-center justify-between">
<span class="font-technical-data text-technical-data px-space-xs py-0.5 rounded bg-surface-container-high text-primary font-semibold">STAGE 02</span>
<span class="material-symbols-outlined text-outline text-[20px]">tune</span>
</div>
<h3 class="font-headline-md text-headline-md text-on-surface">Normalize &amp; Buffer</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant">
              Landmarks are hip-centered and scale-normalized, buffered up to 160 frames, and sent to the server as one sentence window when a pause is detected.
            </p>
</div>
<div class="mt-space-lg font-technical-data text-technical-data text-outline uppercase tracking-wider">1629 floats/frame over WebSocket</div>
</div>
<div class="bg-surface-container-low p-space-lg rounded flex flex-col justify-between">
<div class="flex flex-col gap-space-sm">
<div class="flex items-center justify-between">
<span class="font-technical-data text-technical-data px-space-xs py-0.5 rounded bg-surface-container-high text-error font-semibold">STAGE 03</span>
<span class="material-symbols-outlined text-outline text-[20px]">translate</span>
</div>
<h3 class="font-headline-md text-headline-md text-on-surface">Sign→Text (unreliable)</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant">
              A small Transformer encoder-decoder attempts gloss/text decoding. It mode-collapsed in evaluation (3.9% unique outputs). A distance-matched bounded demo phrase is used as an honest fallback.
            </p>
</div>
<div class="mt-space-lg font-technical-data text-technical-data text-error uppercase tracking-wider">WER ~1.02, disclosed NO-GO</div>
</div>
<div class="bg-surface-container-low p-space-lg rounded flex flex-col justify-between">
<div class="flex flex-col gap-space-sm">
<div class="flex items-center justify-between">
<span class="font-technical-data text-technical-data px-space-xs py-0.5 rounded bg-surface-container-high text-error font-semibold">STAGE 04</span>
<span class="material-symbols-outlined text-outline text-[20px]">accessibility_new</span>
</div>
<h3 class="font-headline-md text-headline-md text-on-surface">Text→Pose (near-static)</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant">
              A GRU decoder regresses a 60-frame pose sequence from a gloss. It produces near-static output (7.9% of ground-truth motion). Ten supported phrases use a real recorded clip instead.
            </p>
</div>
<div class="mt-space-lg font-technical-data text-technical-data text-outline uppercase tracking-wider">Canvas 2D render, no avatar</div>
</div>
</div>
</section>
<!-- Results table -->
<section class="flex flex-col gap-space-md">
<div class="flex flex-col md:flex-row md:items-end justify-between gap-space-md">
<div class="flex flex-col gap-space-xs">
<span class="font-technical-data text-technical-data text-primary uppercase">MODULE 02 — MEASURED RESULTS</span>
<h2 class="font-headline-lg text-headline-lg text-on-surface">Phase 0 Evaluation (Test Split)</h2>
<p class="font-body-md text-body-md text-on-surface-variant max-w-3xl">
            13,000 rows sampled from iSign v1.1 (122,925 usable rows total, 5,577 videos), split by video 10,000/1,500/1,500 (train/val/test) to prevent leakage. Vocabulary: 17,659 tokens.
          </p>
</div>
<div class="flex items-center gap-space-xs self-start md:self-end">
<span class="px-space-xs py-0.5 bg-surface-container-high rounded text-on-surface font-technical-data text-technical-data">research/outputs/phase0/RESULTS.md</span>
</div>
</div>
<div class="overflow-x-auto rounded bg-surface-container-low shadow-sm">
<table class="w-full text-left border-collapse">
<thead>
<tr class="bg-surface-container-high font-technical-data text-technical-data text-on-surface-variant">
<th class="py-space-md px-space-lg uppercase font-semibold">Model (Sign→Text)</th>
<th class="py-space-md px-space-md uppercase font-semibold">WER</th>
<th class="py-space-md px-space-md uppercase font-semibold">BLEU</th>
<th class="py-space-md px-space-md uppercase font-semibold">ROUGE-L</th>
<th class="py-space-md px-space-lg uppercase font-semibold">Note</th>
</tr>
</thead>
<tbody class="divide-y-0 font-body-sm text-body-sm text-on-surface">
<tr class="hover:bg-surface-container transition-colors">
<td class="py-space-md px-space-lg">Trivial baseline<br><span class="font-technical-data text-technical-data text-outline">always predicts "Let me tell you about it."</span></td>
<td class="py-space-md px-space-md font-technical-metric text-technical-metric">1.018</td>
<td class="py-space-md px-space-md font-technical-metric text-technical-metric">0.264</td>
<td class="py-space-md px-space-md font-technical-metric text-technical-metric">0.025</td>
<td class="py-space-md px-space-lg font-technical-data text-technical-data text-on-surface-variant">The bar every real model must clear</td>
</tr>
<tr class="bg-surface-container-lowest/40 hover:bg-surface-container transition-colors">
<td class="py-space-md px-space-lg">GRU baseline (served)</td>
<td class="py-space-md px-space-md font-technical-metric text-technical-metric">1.015</td>
<td class="py-space-md px-space-md font-technical-metric text-technical-metric text-primary font-bold">0.311</td>
<td class="py-space-md px-space-md font-technical-metric text-technical-metric">0.094</td>
<td class="py-space-md px-space-lg font-technical-data text-technical-data text-error">0.6% unique outputs — mode collapsed</td>
</tr>
<tr class="hover:bg-surface-container transition-colors">
<td class="py-space-md px-space-lg">Transformer (full features)</td>
<td class="py-space-md px-space-md font-technical-metric text-technical-metric">1.019</td>
<td class="py-space-md px-space-md font-technical-metric text-technical-metric">0.158</td>
<td class="py-space-md px-space-md font-technical-metric text-technical-metric">0.044</td>
<td class="py-space-md px-space-lg font-technical-data text-technical-data text-error">3.9% unique outputs — mode collapsed</td>
</tr>
<tr class="bg-surface-container-lowest/40 hover:bg-surface-container transition-colors">
<td class="py-space-md px-space-lg">Transformer (no-face ablation)</td>
<td class="py-space-md px-space-md font-technical-metric text-technical-metric">1.034</td>
<td class="py-space-md px-space-md font-technical-metric text-technical-metric">0.207</td>
<td class="py-space-md px-space-md font-technical-metric text-technical-metric">—</td>
<td class="py-space-md px-space-lg font-technical-data text-technical-data text-on-surface-variant">Loaded, not used by the live demo</td>
</tr>
</tbody>
</table>
</div>
<div class="overflow-x-auto rounded bg-surface-container-low shadow-sm">
<table class="w-full text-left border-collapse">
<thead>
<tr class="bg-surface-container-high font-technical-data text-technical-data text-on-surface-variant">
<th class="py-space-md px-space-lg uppercase font-semibold">Text→Pose</th>
<th class="py-space-md px-space-md uppercase font-semibold">DTW Distance</th>
<th class="py-space-md px-space-md uppercase font-semibold">MSE</th>
<th class="py-space-md px-space-md uppercase font-semibold">Motion Energy</th>
<th class="py-space-md px-space-lg uppercase font-semibold">Note</th>
</tr>
</thead>
<tbody class="divide-y-0 font-body-sm text-body-sm text-on-surface">
<tr class="hover:bg-surface-container transition-colors">
<td class="py-space-md px-space-lg">Ground truth</td>
<td class="py-space-md px-space-md font-technical-metric text-technical-metric">—</td>
<td class="py-space-md px-space-md font-technical-metric text-technical-metric">—</td>
<td class="py-space-md px-space-md font-technical-metric text-technical-metric">0.0417</td>
<td class="py-space-md px-space-lg font-technical-data text-technical-data text-on-surface-variant">Reference for comparison</td>
</tr>
<tr class="bg-surface-container-lowest/40 hover:bg-surface-container transition-colors">
<td class="py-space-md px-space-lg">Retrieval (nearest-neighbor)</td>
<td class="py-space-md px-space-md font-technical-metric text-technical-metric">613.996</td>
<td class="py-space-md px-space-md font-technical-metric text-technical-metric">0.177</td>
<td class="py-space-md px-space-md font-technical-metric text-technical-metric">0.034</td>
<td class="py-space-md px-space-lg font-technical-data text-technical-data text-on-surface-variant">Closer to real motion than generative</td>
</tr>
<tr class="hover:bg-surface-container transition-colors">
<td class="py-space-md px-space-lg">Generative (GRU, served)</td>
<td class="py-space-md px-space-md font-technical-metric text-technical-metric">453.727</td>
<td class="py-space-md px-space-md font-technical-metric text-technical-metric">0.122</td>
<td class="py-space-md px-space-md font-technical-metric text-technical-metric text-error">0.00328</td>
<td class="py-space-md px-space-lg font-technical-data text-technical-data text-error">7.9% of GT motion — near-static</td>
</tr>
</tbody>
</table>
</div>
<div class="flex items-start gap-space-sm p-space-md bg-surface-container-lowest rounded font-technical-data text-technical-data text-outline">
<span class="material-symbols-outlined text-outline text-[16px] mt-0.5">info</span>
<span>
          Mean inference latency (model only, not round-trip): Sign→Text 183.6 ms, Text→Pose 13.1 ms. WER at or above 1.0 means correcting the output takes as much editing as retyping it. All numbers above are from research/outputs/phase0/RESULTS.md and backend/signsage_knowledge_manifest.json — nothing here is extrapolated or rounded up.
        </span>
</div>
</section>
<!-- Limitations -->
<section class="flex flex-col gap-space-md">
<div class="flex flex-col gap-space-xs">
<span class="font-technical-data text-technical-data text-tertiary uppercase">MODULE 03 — LIMITATIONS &amp; SCOPE</span>
<h2 class="font-headline-lg text-headline-lg text-on-surface">What this project does not claim</h2>
</div>
<div class="grid grid-cols-1 md:grid-cols-3 gap-space-md">
<div class="bg-surface-container-low p-space-lg rounded flex flex-col justify-between">
<div class="flex flex-col gap-space-sm">
<div class="flex items-center justify-between text-tertiary">
<span class="font-technical-data text-technical-data uppercase tracking-wider font-semibold">LIMITATION 01</span>
<span class="material-symbols-outlined text-[20px]">visibility</span>
</div>
<h3 class="font-headline-md text-headline-md text-on-surface">No dialect-specific evaluation</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
              iSign v1.1 is a single public dataset. No regional-dialect breakdown, informant count, or dialect-coverage claim is made — none was measured.
            </p>
</div>
</div>
<div class="bg-surface-container-low p-space-lg rounded flex flex-col justify-between">
<div class="flex flex-col gap-space-sm">
<div class="flex items-center justify-between text-primary">
<span class="font-technical-data text-technical-data uppercase tracking-wider font-semibold">FACT 02</span>
<span class="material-symbols-outlined text-[20px]">shield</span>
</div>
<h3 class="font-headline-md text-headline-md text-on-surface">Video stays on-device; inference does not</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
              Camera video and images are never transmitted. Landmark coordinates are sent to a backend server for inference over WebSocket — this is not a fully offline / on-device pipeline.
            </p>
</div>
</div>
<div class="bg-surface-container-low p-space-lg rounded flex flex-col justify-between">
<div class="flex flex-col gap-space-sm">
<div class="flex items-center justify-between text-secondary">
<span class="font-technical-data text-technical-data uppercase tracking-wider font-semibold">DISCLOSURE 03</span>
<span class="material-symbols-outlined text-[20px]">gavel</span>
</div>
<h3 class="font-headline-md text-headline-md text-on-surface">Not certified for critical use</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
              This is an academic prototype, not certified for medical, legal, or safety-critical communication, and not a substitute for a certified human ISL interpreter.
            </p>
</div>
</div>
</div>
</section>
</div>
</main><footer class="w-full bg-surface-container-lowest border-t border-outline-variant/30 py-space-xl"><div class="w-full max-w-7xl mx-auto px-margin lg:px-margin-desktop flex flex-col sm:flex-row items-center justify-between gap-space-md text-on-surface-variant font-technical-data text-technical-data"><p>M.Tech academic research project — bidirectional ISL/English translation.</p><p class="text-outline">Full results: research/outputs/phase0/RESULTS.md</p></div></footer>`;

export const stitchMonographHtml = `<header class="fixed top-0 left-0 right-0 z-50 bg-surface-container-lowest/95 backdrop-blur-md border-b border-outline-variant/40"><div class="h-16 w-full px-margin-desktop flex items-center justify-between gap-gutter-desktop"><div class="flex items-center gap-space-lg"><div class="flex items-center gap-space-sm"><img alt="SignSage logo" class="h-8 w-auto object-contain" src="/signsage-logo.svg"><span class="font-headline-md text-headline-md tracking-tight text-on-surface">SignSage</span></div></div><nav class="hidden xl:flex items-center gap-space-lg h-full" data-active-classes="text-primary border-b-2 border-primary"><a class="h-full flex items-center px-space-xs font-label-md text-label-md text-on-surface-variant hover:text-on-surface transition-colors" data-path="translator" href="#">Translator</a><a class="h-full flex items-center px-space-xs font-label-md text-label-md text-on-surface-variant hover:text-on-surface transition-colors" data-path="research-and-benchmarks" href="#">Research &amp; Benchmarks</a><a aria-current="page" class="h-full flex items-center px-space-xs font-label-md transition-colors text-primary border-b-2 border-primary" data-path="system-architecture" href="#">Technology</a></nav><div class="w-8 h-8 rounded-full bg-primary flex items-center justify-center shrink-0"><span class="material-symbols-outlined text-on-primary text-[18px]">person</span></div></div></header><main class="w-full pt-16 bg-surface min-h-[calc(100vh-4rem)]"><div class="flex flex-col w-full">
<div class="w-full px-margin-desktop py-space-xl max-w-7xl mx-auto flex flex-col gap-space-xl">
<section class="flex flex-col gap-space-md">
<span class="font-technical-data text-technical-data text-primary uppercase tracking-wider">TECHNOLOGY OVERVIEW</span>
<h1 class="font-headline-xl text-headline-xl text-on-surface tracking-tight">The actual stack, in plain terms</h1>
<p class="font-body-lg text-body-lg text-on-surface-variant leading-relaxed max-w-3xl">
          No WebGPU, no custom shaders, no proprietary avatar engine. SignSage is a browser MediaPipe pipeline talking to a small FastAPI/PyTorch backend over one WebSocket connection.
        </p>
</section>
<section class="grid grid-cols-1 lg:grid-cols-3 gap-space-md">
<div class="bg-surface-container-low p-space-lg rounded flex flex-col gap-space-sm">
<div class="flex items-center justify-between">
<span class="font-technical-data text-technical-data px-space-xs py-0.5 rounded bg-surface-container-high text-primary font-semibold">BROWSER</span>
<span class="material-symbols-outlined text-outline text-[20px]">videocam</span>
</div>
<h3 class="font-headline-md text-headline-md text-on-surface">MediaPipe Holistic</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
            @mediapipe/tasks-vision's HolisticLandmarker runs as WebAssembly. Outputs 543 3D landmarks/frame: 468 face, 33 pose, 21+21 hands. Landmarks are hip-centered and scale-normalized in JS before transmission.
          </p>
</div>
<div class="bg-surface-container-low p-space-lg rounded flex flex-col gap-space-sm">
<div class="flex items-center justify-between">
<span class="font-technical-data text-technical-data px-space-xs py-0.5 rounded bg-surface-container-high text-primary font-semibold">TRANSPORT</span>
<span class="material-symbols-outlined text-outline text-[20px]">cable</span>
</div>
<h3 class="font-headline-md text-headline-md text-on-surface">WebSocket (FastAPI)</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
            One connection per session at /api/v1/ws/translate. Messages: landmarks.frame, sentence.boundary, text.input from the client; translation.text, translation.pose, error from the server. No REST endpoint carries translation traffic.
          </p>
</div>
<div class="bg-surface-container-low p-space-lg rounded flex flex-col gap-space-sm">
<div class="flex items-center justify-between">
<span class="font-technical-data text-technical-data px-space-xs py-0.5 rounded bg-surface-container-high text-primary font-semibold">SERVER</span>
<span class="material-symbols-outlined text-outline text-[20px]">memory</span>
</div>
<h3 class="font-headline-md text-headline-md text-on-surface">PyTorch (CPU or GPU)</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
            Runs the sign→text and text→pose checkpoints plus a DTW-based demo-phrase lexicon. No quantization, no ONNX export in this build — plain PyTorch inference.
          </p>
</div>
</section>
<section class="flex flex-col gap-space-md">
<h2 class="font-headline-lg text-headline-lg text-on-surface">Model architecture (as actually implemented)</h2>
<div class="grid grid-cols-1 lg:grid-cols-2 gap-space-md">
<div class="bg-surface-container-low p-space-lg rounded flex flex-col gap-space-sm">
<h3 class="font-headline-md text-headline-md text-on-surface">SignTransformer (Sign→Text)</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
            A standard sequence-to-sequence Transformer: 2 encoder layers, 2 decoder layers, d_model=192, 4 attention heads, feedforward dim 384, dropout 0.1, sinusoidal positional encoding. Input: normalized landmark frames (up to 160/sequence). Output: a token sequence over a 17,659-token vocabulary. A separate plain-GRU baseline was trained for comparison and is not served (it mode-collapsed more severely).
          </p>
</div>
<div class="bg-surface-container-low p-space-lg rounded flex flex-col gap-space-sm">
<h3 class="font-headline-md text-headline-md text-on-surface">TextToPoseGRU (Text→ISL)</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
            A single GRU cell (hidden size 192) stepped 60 times to regress a fixed-length pose sequence from a mean-pooled text embedding plus a learned per-step time embedding. Not bidirectional, not multi-layer. Evaluation showed near-static output (7.9% of ground-truth motion), which is why the ten supported demo phrases use a real recorded clip instead of this model's output.
          </p>
</div>
</div>
</section>
<section class="flex flex-col gap-space-md">
<h2 class="font-headline-lg text-headline-lg text-on-surface">Bounded demo phrases</h2>
<p class="font-body-md text-body-md text-on-surface-variant max-w-3xl">
          Both translation directions have a small, honestly-labelled bounded fallback for the same ten phrases, backed by real recorded clips in backend/demo_prototypes/: Hello, Thank you, Yes, No, Help, Please, Sorry, Good morning, How are you, I need help. For Sign→Text this is a DTW distance match against the recorded clip's hand-motion features; for Text→ISL it is a direct phrase lookup (with a small list of common rephrasings) that plays the recorded clip back. Outside this set, both directions fall through to the general research models described above, which the app labels honestly as unreliable.
        </p>
</section>
</div>
</div>
</main><footer class="w-full bg-surface-container-lowest border-t border-outline-variant/30 py-space-xl"><div class="w-full max-w-7xl mx-auto px-margin lg:px-margin-desktop flex flex-col sm:flex-row items-center justify-between gap-space-md text-on-surface-variant font-technical-data text-technical-data"><p>M.Tech academic research project — bidirectional ISL/English translation.</p><p class="text-outline">Full results: research/outputs/phase0/RESULTS.md</p></div></footer>`;
