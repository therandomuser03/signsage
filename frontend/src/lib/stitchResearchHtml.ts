export const stitchBenchmarksHtml = `<header class="fixed top-0 left-0 right-0 z-50 bg-surface-container-lowest border-b border-outline-variant"><div class="h-16 w-full px-margin-desktop flex items-center justify-between gap-space-md"><div class="flex items-center gap-space-md shrink-0"><img alt="SignSage Precision Instrument Logo" class="h-8 w-auto object-contain" src="https://lh3.googleusercontent.com/aida/AEtjO1U2bkXg2qQyVUZ8If-9_8pUSOq2qIrOL7hb05_I9QTSFVLD_W0OaH5kjwE4QseaYCfqSZIp-pOWq7y60SdiVxNkUZAQFvd7G6gZFvRe2sfHt4lTLyZHNnYDC480vOZ-LiwRMB1575GpzkUFIjPim17zoUsrZDy_R2dl8r1Bqg4ocEB4OsqimbEZoR1PE55IOBtAcyWcH6y0dK_SjK0U-2qScjl9_hsI6gAj7rjHJAwwMbo0i83qfzXiAHE"><span class="font-headline-md text-headline-md tracking-tight text-on-surface">SignSage</span><span class="font-technical-data text-technical-data px-space-sm py-0.5 rounded bg-surface-container-high text-on-surface-variant border border-outline-variant tracking-wider uppercase">RESEARCH PROTOTYPE v0.4.2</span></div><nav class="hidden xl:flex items-center gap-space-lg h-full" data-active-classes="text-primary border-b-2 border-primary"><a class="h-full flex items-center px-space-xs font-label-md text-label-md text-on-surface-variant hover:text-on-surface transition-colors" data-path="translator" href="#">Translator</a><a aria-current="page" class="h-full flex items-center px-space-xs font-label-md transition-colors text-primary border-b-2 border-primary" data-path="research-and-benchmarks" href="#">Research &amp; Benchmarks</a><a class="h-full flex items-center px-space-xs font-label-md text-label-md text-on-surface-variant hover:text-on-surface transition-colors" data-path="session-archive" href="#">Session Archive</a><a class="h-full flex items-center px-space-xs font-label-md text-label-md text-on-surface-variant hover:text-on-surface transition-colors" data-path="system-architecture" href="#">System Architecture</a></nav><div class="flex items-center gap-space-md shrink-0"><div class="hidden lg:flex items-center gap-space-xs px-space-sm py-1 rounded bg-surface-container-low border border-outline-variant"><span class="relative flex h-2 w-2"><span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span><span class="relative inline-flex rounded-full h-2 w-2 bg-primary"></span></span><span class="font-technical-data text-technical-data text-on-surface-variant">System Ready · Edge Inference active</span></div><div class="hidden md:flex items-center gap-space-xs px-space-sm py-1 rounded bg-surface-container-low border border-outline-variant"><span class="material-symbols-outlined text-[14px] text-primary">lock</span><span class="font-technical-data text-technical-data text-on-surface-variant">On-device camera processing</span></div><div class="px-space-sm py-1 rounded bg-surface-container-high border border-outline-variant font-technical-data text-technical-data text-on-surface">ID: LAB_ISL_04</div><div class="w-8 h-8 rounded-full bg-primary flex items-center justify-center shrink-0"><span class="material-symbols-outlined text-on-primary text-[18px]">person</span></div></div></div></header><main class="w-full pt-16 bg-surface-container-lowest min-h-[calc(100vh-4rem)]"><div class="flex flex-col w-full">
<div class="w-full px-margin-desktop py-space-xl max-w-7xl mx-auto flex flex-col gap-space-xl">
<!-- Dossier Header Section -->
<section class="flex flex-col gap-space-md">
<div class="flex flex-wrap items-center gap-space-xs text-primary font-technical-data text-technical-data uppercase tracking-wider">
<span class="flex h-2 w-2 rounded-full bg-primary animate-pulse"></span>
<span>LAB_ISL_04 · MONOGRAPH RECORD REF: SSR-2024-B8</span>
<span class="text-on-surface-variant">·</span>
<span class="text-on-surface-variant">PEER-EVALUATED BENCHMARK SUITE</span>
</div>
<div class="flex flex-col xl:flex-row xl:items-end justify-between gap-space-lg">
<div class="max-w-4xl flex flex-col gap-space-sm">
<h1 class="font-headline-xl text-headline-xl text-on-surface tracking-tight">
            Architectural Pipeline &amp; Empirical Evaluation
          </h1>
<p class="font-body-lg text-body-lg text-on-surface-variant leading-relaxed">
            A continuous sequence modeling framework for bidirectional Indian Sign Language translation under variable ambient lux and non-rigid digital hand occlusions.
          </p>
</div>
<div class="flex flex-wrap xl:flex-col items-start xl:items-end gap-space-xs shrink-0">
<div class="flex items-center gap-space-xs px-space-sm py-1 rounded bg-surface-container-high text-on-surface font-technical-data text-technical-data">
<span class="text-outline">CORPUS:</span>
<span class="text-primary font-semibold">ISL-CSL-12k</span>
</div>
<div class="flex items-center gap-space-xs px-space-sm py-1 rounded bg-surface-container-high text-on-surface font-technical-data text-technical-data">
<span class="text-outline">ARCH:</span>
<span>MediaPipe Holistic → BiGRU-Xform</span>
</div>
<div class="flex items-center gap-space-xs px-space-sm py-1 rounded bg-surface-container-high text-on-surface font-technical-data text-technical-data">
<span class="text-outline">STAGE:</span>
<span class="text-secondary">Release candidate v0.4.2</span>
</div>
</div>
</div>
<!-- Quick Telemetry Bar -->
<div class="grid grid-cols-2 sm:grid-cols-4 gap-space-sm pt-space-xs">
<div class="bg-surface-container-low p-space-md rounded flex flex-col justify-between">
<span class="font-technical-data text-technical-data text-on-surface-variant uppercase">Keypoint Vector Norm</span>
<div class="flex items-baseline gap-space-xs mt-space-xs">
<span class="font-headline-md text-headline-md text-on-surface">543</span>
<span class="font-technical-data text-technical-data text-outline">3D nodes</span>
</div>
<span class="font-technical-data text-technical-data text-primary mt-space-xs">±0.003 normalized RMS</span>
</div>
<div class="bg-surface-container-low p-space-md rounded flex flex-col justify-between">
<span class="font-technical-data text-technical-data text-on-surface-variant uppercase">Client Inference P95</span>
<div class="flex items-baseline gap-space-xs mt-space-xs">
<span class="font-headline-md text-headline-md text-on-surface">38.2</span>
<span class="font-technical-data text-technical-data text-outline">ms</span>
</div>
<span class="font-technical-data text-technical-data text-primary mt-space-xs">Target ≤ 50ms [PASS]</span>
</div>
<div class="bg-surface-container-low p-space-md rounded flex flex-col justify-between">
<span class="font-technical-data text-technical-data text-on-surface-variant uppercase">Word Error Rate</span>
<div class="flex items-baseline gap-space-xs mt-space-xs">
<span class="font-headline-md text-headline-md text-on-surface">14.8</span>
<span class="font-technical-data text-technical-data text-outline">%</span>
</div>
<span class="font-technical-data text-technical-data text-secondary mt-space-xs">−7.6% over baseline</span>
</div>
<div class="bg-surface-container-low p-space-md rounded flex flex-col justify-between">
<span class="font-technical-data text-technical-data text-on-surface-variant uppercase">Grammar Syntactic Drift</span>
<div class="flex items-baseline gap-space-xs mt-space-xs">
<span class="font-headline-md text-headline-md text-on-surface">0.084</span>
<span class="font-technical-data text-technical-data text-outline">Δloss</span>
</div>
<span class="font-technical-data text-technical-data text-outline mt-space-xs">SOV / SVO alignment lock</span>
</div>
</div>
</section>
<!-- Four-Stage Pipeline Architecture Breakdown -->
<section class="flex flex-col gap-space-md">
<div class="flex flex-col gap-space-xs">
<div class="flex items-center gap-space-xs">
<span class="font-technical-data text-technical-data text-primary uppercase">MODULE 01 — COMPUTATIONAL PATHWAY</span>
</div>
<h2 class="font-headline-lg text-headline-lg text-on-surface">Four-Stage Continuous Translation Engine</h2>
<p class="font-body-md text-body-md text-on-surface-variant max-w-3xl">
          Deterministic translation of non-rigid continuous manual signs demands a tightly bounded spatial-to-semantic transformation. Video frames are decomposed immediately in local memory without remote socket transit.
        </p>
</div>
<div class="grid grid-cols-1 lg:grid-cols-4 gap-space-md">
<!-- Stage 01 -->
<div class="bg-surface-container-low p-space-lg rounded flex flex-col justify-between relative group hover:bg-surface-container transition-colors">
<div class="flex flex-col gap-space-sm">
<div class="flex items-center justify-between">
<span class="font-technical-data text-technical-data px-space-xs py-0.5 rounded bg-surface-container-high text-primary font-semibold">STAGE 01</span>
<span class="material-symbols-outlined text-outline text-[20px]">view_in_ar</span>
</div>
<h3 class="font-headline-md text-headline-md text-on-surface">Spatial Landmark Extraction</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant">
              High-throughput skeletal tracking yielding 543 distinct 3D coordinates. MediaPipe Holistic runtime localized to WebAssembly SIMD kernels with Euclidean normalization across face, palms, and torso.
            </p>
</div>
<div class="mt-space-lg flex flex-col gap-space-sm">
<div class="bg-surface-container-lowest p-space-sm rounded flex flex-col gap-space-xs">
<div class="flex justify-between font-technical-data text-technical-data">
<span class="text-outline">FACE MESH</span>
<span class="text-on-surface">468 pts</span>
</div>
<div class="w-full bg-surface-container-highest h-1 rounded overflow-hidden">
<div class="bg-primary h-full w-[86%]"></div>
</div>
<div class="flex justify-between font-technical-data text-technical-data mt-1">
<span class="text-outline">DUAL HAND RIG</span>
<span class="text-on-surface">42 pts (21 × 2)</span>
</div>
<div class="w-full bg-surface-container-highest h-1 rounded overflow-hidden">
<div class="bg-secondary h-full w-[100%]"></div>
</div>
</div>
<span class="font-technical-data text-technical-data text-outline uppercase tracking-wider">Normalized Coordinates: [x, y, z, vis]</span>
</div>
</div>
<!-- Stage 02 -->
<div class="bg-surface-container-low p-space-lg rounded flex flex-col justify-between relative group hover:bg-surface-container transition-colors">
<div class="flex flex-col gap-space-sm">
<div class="flex items-center justify-between">
<span class="font-technical-data text-technical-data px-space-xs py-0.5 rounded bg-surface-container-high text-primary font-semibold">STAGE 02</span>
<span class="material-symbols-outlined text-outline text-[20px]">timeline</span>
</div>
<h3 class="font-headline-md text-headline-md text-on-surface">Temporal Sequence Modeling</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant">
              Bidirectional Gated Recurrent Units (BiGRU) equipped with multi-head self-attention. Dynamic Time Warping (DTW) calibrates signing velocity variances and boundary holds between discrete lexical units.
            </p>
</div>
<div class="mt-space-lg flex flex-col gap-space-sm">
<!-- Inline SVG: Velocity / DTW Waveform -->
<div class="bg-surface-container-lowest p-space-sm rounded flex flex-col">
<div class="flex justify-between font-technical-data text-technical-data text-outline mb-1">
<span>WARP TRAJECTORY τ</span>
<span class="text-primary font-semibold">σ = 0.014</span>
</div>
<svg class="w-full h-12 text-primary" fill="none" viewBox="0 0 200 48">
<path d="M0,35 Q25,8 50,30 T100,16 T150,38 T200,22" fill="none" stroke="currentColor" stroke-width="1.75"></path>
<path d="M0,39 Q25,18 50,36 T100,26 T150,42 T200,30" fill="none" opacity="0.45" stroke="currentColor" stroke-dasharray="3,3" stroke-width="1"></path>
</svg>
</div>
<span class="font-technical-data text-technical-data text-outline uppercase tracking-wider">BiGRU: 2 × 256 Hidden Units</span>
</div>
</div>
<!-- Stage 03 -->
<div class="bg-surface-container-low p-space-lg rounded flex flex-col justify-between relative group hover:bg-surface-container transition-colors">
<div class="flex flex-col gap-space-sm">
<div class="flex items-center justify-between">
<span class="font-technical-data text-technical-data px-space-xs py-0.5 rounded bg-surface-container-high text-primary font-semibold">STAGE 03</span>
<span class="material-symbols-outlined text-outline text-[20px]">translate</span>
</div>
<h3 class="font-headline-md text-headline-md text-on-surface">Gloss-to-Text Cross-Attention</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant">
              Sequence-to-sequence Transformer decoding grammatical gloss tokens into canonical Indian English syntax. Accurately maps Topic-Comment structural order directly to Subject-Verb-Object (SVO).
            </p>
</div>
<div class="mt-space-lg flex flex-col gap-space-sm">
<div class="bg-surface-container-lowest p-space-sm rounded flex flex-col gap-space-xs font-technical-data text-technical-data">
<div class="text-outline uppercase">GLOSS INPUT:</div>
<div class="text-secondary bg-surface-container-high px-space-xs py-0.5 rounded font-mono truncate">[HOSPITAL] [DOCTOR] [ARRIVE-NOT]</div>
<div class="text-outline uppercase mt-1">SYNTACTIC DECODE:</div>
<div class="text-on-surface truncate font-body-sm">“The doctor hasn't arrived at the hospital.”</div>
</div>
<span class="font-technical-data text-technical-data text-outline uppercase tracking-wider">Transformer: 6 Enc / 6 Dec Layers</span>
</div>
</div>
<!-- Stage 04 -->
<div class="bg-surface-container-low p-space-lg rounded flex flex-col justify-between relative group hover:bg-surface-container transition-colors">
<div class="flex flex-col gap-space-sm">
<div class="flex items-center justify-between">
<span class="font-technical-data text-technical-data px-space-xs py-0.5 rounded bg-surface-container-high text-primary font-semibold">STAGE 04</span>
<span class="material-symbols-outlined text-outline text-[20px]">accessibility_new</span>
</div>
<h3 class="font-headline-md text-headline-md text-on-surface">Pose Synthesis &amp; Kinematics</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant">
              Reverse path synthesis from English prompt to fluent ISL avatar landmarks. Cubic Bézier joint interpolation combined with Fabrik inverse kinematics ensures anatomically valid non-snapping articulation.
            </p>
</div>
<div class="mt-space-lg flex flex-col gap-space-sm">
<div class="bg-surface-container-lowest p-space-sm rounded flex items-center justify-between">
<div class="flex flex-col">
<span class="font-technical-data text-technical-data text-outline">ROTATIONAL ERROR</span>
<span class="font-technical-metric text-technical-metric text-primary">≤ 0.041 rad</span>
</div>
<!-- Mini Circular Progress Chart (Kinematic clamp integrity) -->
<svg class="w-8 h-8 transform -rotate-90 text-primary shrink-0" viewBox="0 0 36 36">
<path class="text-surface-container-highest" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="currentColor" stroke-width="3"></path>
<path class="text-primary" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="currentColor" stroke-dasharray="92, 100" stroke-linecap="round" stroke-width="3"></path>
</svg>
</div>
<span class="font-technical-data text-technical-data text-outline uppercase tracking-wider">Avatar Engine: 60 FPS WebGL Canvas</span>
</div>
</div>
</div>
</section>
<!-- Empirical Benchmarks & Quantitative Evaluation Matrix -->
<section class="flex flex-col gap-space-md">
<div class="flex flex-col md:flex-row md:items-end justify-between gap-space-md">
<div class="flex flex-col gap-space-xs">
<div class="flex items-center gap-space-xs">
<span class="font-technical-data text-technical-data text-primary uppercase">MODULE 02 — EXPERIMENTAL FINDINGS</span>
</div>
<h2 class="font-headline-lg text-headline-lg text-on-surface">Quantitative Benchmark Matrix</h2>
<p class="font-body-md text-body-md text-on-surface-variant max-w-3xl">
            Controlled evaluation against standard academic baselines. Testing was executed on client hardware across 1,400 unseen continuous sign sequences recorded under heterogeneous indoor lighting conditions (150–800 lux).
          </p>
</div>
<div class="flex items-center gap-space-xs self-start md:self-end">
<span class="font-technical-data text-technical-data text-outline uppercase">Split Condition:</span>
<span class="px-space-xs py-0.5 bg-surface-container-high rounded text-on-surface font-technical-data text-technical-data">70:15:15 Unseen Signers</span>
</div>
</div>
<!-- Restrained Scientific Data Table -->
<div class="overflow-x-auto rounded bg-surface-container-low shadow-sm">
<table class="w-full text-left border-collapse">
<thead>
<tr class="bg-surface-container-high font-technical-data text-technical-data text-on-surface-variant">
<th class="py-space-md px-space-lg uppercase font-semibold">Evaluation Metric</th>
<th class="py-space-md px-space-md uppercase font-semibold">Current (v0.4.2)</th>
<th class="py-space-md px-space-md uppercase font-semibold">BiGRU Baseline</th>
<th class="py-space-md px-space-md uppercase font-semibold">Δ Relative Variance</th>
<th class="py-space-md px-space-lg uppercase font-semibold">Evaluation Test Set / Protocol</th>
</tr>
</thead>
<tbody class="divide-y-0 font-body-sm text-body-sm text-on-surface">
<!-- Row 1 -->
<tr class="hover:bg-surface-container transition-colors">
<td class="py-space-md px-space-lg">
<div class="flex flex-col">
<span class="font-headline-md text-body-md font-semibold text-on-surface">Word Error Rate (WER)</span>
<span class="font-technical-data text-technical-data text-outline">Lower indicates superior lexical recall</span>
</div>
</td>
<td class="py-space-md px-space-md">
<span class="font-technical-metric text-technical-metric text-primary font-bold">14.8%</span>
</td>
<td class="py-space-md px-space-md font-technical-metric text-technical-metric text-outline">22.4%</td>
<td class="py-space-md px-space-md">
<span class="inline-flex items-center gap-1 font-technical-data text-technical-data text-primary px-space-xs py-0.5 rounded bg-primary/10">
                  −7.6 pp (−33.9%)
                </span>
</td>
<td class="py-space-md px-space-lg font-technical-data text-technical-data text-on-surface-variant">
                ISL Daily Comm-v2 (n=1,400 continuous sentences)
              </td>
</tr>
<!-- Row 2 -->
<tr class="bg-surface-container-lowest/40 hover:bg-surface-container transition-colors">
<td class="py-space-md px-space-lg">
<div class="flex flex-col">
<span class="font-headline-md text-body-md font-semibold text-on-surface">BLEU-4 Score</span>
<span class="font-technical-data text-technical-data text-outline">Standard 4-gram n-gram bilingual precision</span>
</div>
</td>
<td class="py-space-md px-space-md">
<span class="font-technical-metric text-technical-metric text-primary font-bold">38.6</span>
</td>
<td class="py-space-md px-space-md font-technical-metric text-technical-metric text-outline">29.1</td>
<td class="py-space-md px-space-md">
<span class="inline-flex items-center gap-1 font-technical-data text-technical-data text-primary px-space-xs py-0.5 rounded bg-primary/10">
                  +9.5 pts (+32.6%)
                </span>
</td>
<td class="py-space-md px-space-lg font-technical-data text-technical-data text-on-surface-variant">
                Syntactically aligned reference gloss sets
              </td>
</tr>
<!-- Row 3 -->
<tr class="hover:bg-surface-container transition-colors">
<td class="py-space-md px-space-lg">
<div class="flex flex-col">
<span class="font-headline-md text-body-md font-semibold text-on-surface">ROUGE-L Metric</span>
<span class="font-technical-data text-technical-data text-outline">Longest common subsequence overlap</span>
</div>
</td>
<td class="py-space-md px-space-md">
<span class="font-technical-metric text-technical-metric text-secondary font-bold">54.2</span>
</td>
<td class="py-space-md px-space-md font-technical-metric text-technical-metric text-outline">43.8</td>
<td class="py-space-md px-space-md">
<span class="inline-flex items-center gap-1 font-technical-data text-technical-data text-secondary px-space-xs py-0.5 rounded bg-secondary/10">
                  +10.4 pts (+23.7%)
                </span>
</td>
<td class="py-space-md px-space-lg font-technical-data text-technical-data text-on-surface-variant">
                Continuous dialogue corpus (multi-sentence turns)
              </td>
</tr>
<!-- Row 4 -->
<tr class="bg-surface-container-lowest/40 hover:bg-surface-container transition-colors">
<td class="py-space-md px-space-lg">
<div class="flex flex-col">
<span class="font-headline-md text-body-md font-semibold text-on-surface">Dynamic Time Warping (DTW) Distance</span>
<span class="font-technical-data text-technical-data text-outline">Joint trajectory temporal displacement</span>
</div>
</td>
<td class="py-space-md px-space-md">
<span class="font-technical-metric text-technical-metric text-primary font-bold">0.182</span>
</td>
<td class="py-space-md px-space-md font-technical-metric text-technical-metric text-outline">0.298</td>
<td class="py-space-md px-space-md">
<span class="inline-flex items-center gap-1 font-technical-data text-technical-data text-primary px-space-xs py-0.5 rounded bg-primary/10">
                  −0.116 err (−38.9%)
                </span>
</td>
<td class="py-space-md px-space-lg font-technical-data text-technical-data text-on-surface-variant">
                Fingertip trajectory tracking against motion capture ground truth
              </td>
</tr>
<!-- Row 5 -->
<tr class="hover:bg-surface-container transition-colors">
<td class="py-space-md px-space-lg">
<div class="flex flex-col">
<span class="font-headline-md text-body-md font-semibold text-on-surface">Mean Squared Angular Error (MSE)</span>
<span class="font-technical-data text-technical-data text-outline">Metacarpophalangeal &amp; interphalangeal dev.</span>
</div>
</td>
<td class="py-space-md px-space-md">
<span class="font-technical-metric text-technical-metric text-on-surface font-bold">0.041 rad</span>
</td>
<td class="py-space-md px-space-md font-technical-metric text-technical-metric text-outline">0.076 rad</td>
<td class="py-space-md px-space-md">
<span class="inline-flex items-center gap-1 font-technical-data text-technical-data text-primary px-space-xs py-0.5 rounded bg-primary/10">
                  −0.035 rad (−46.1%)
                </span>
</td>
<td class="py-space-md px-space-lg font-technical-data text-technical-data text-on-surface-variant">
                14-joint skeletal angle deviation across 8 anatomical bounds
              </td>
</tr>
<!-- Row 6 -->
<tr class="bg-surface-container-lowest/40 hover:bg-surface-container transition-colors">
<td class="py-space-md px-space-lg">
<div class="flex flex-col">
<span class="font-headline-md text-body-md font-semibold text-on-surface">Inference Latency (Local WebGPU)</span>
<span class="font-technical-data text-technical-data text-outline">End-to-end frame arrival to token stream</span>
</div>
</td>
<td class="py-space-md px-space-md">
<span class="font-technical-metric text-technical-metric text-primary font-bold">38.2 ms</span>
</td>
<td class="py-space-md px-space-md font-technical-metric text-technical-metric text-outline">112.4 ms</td>
<td class="py-space-md px-space-md">
<span class="inline-flex items-center gap-1 font-technical-data text-technical-data text-primary px-space-xs py-0.5 rounded bg-primary/10">
                  −74.2 ms (−66.0%)
                </span>
</td>
<td class="py-space-md px-space-lg font-technical-data text-technical-data text-on-surface-variant">
                Standard client laptop (Apple M-Series / RTX 3050 mobile equiv.)
              </td>
</tr>
</tbody>
</table>
</div>
<!-- Benchmark Methodology Note -->
<div class="flex items-start gap-space-sm p-space-md bg-surface-container-lowest rounded font-technical-data text-technical-data text-outline">
<span class="material-symbols-outlined text-outline text-[16px] mt-0.5">info</span>
<span>
          Benchmark runs were conducted in batches of 50 continuous trials with warm caches. WER calculations penalize substitution, insertion, and deletion evenly according to standard NIST SCLITE metric configurations.
        </span>
</div>
</section>
<!-- Two-Column Editorial: Visual Anatomy & Non-Manual Features -->
<section class="grid grid-cols-1 lg:grid-cols-12 gap-space-lg items-center">
<div class="lg:col-span-5 flex flex-col gap-space-md">
<div class="flex items-center gap-space-xs">
<span class="font-technical-data text-technical-data text-primary uppercase">MODULE 03 — VISUAL REPRESENTATION</span>
</div>
<h2 class="font-headline-lg text-headline-lg text-on-surface">
          Landmark Coordinate Geometry
        </h2>
<p class="font-body-md text-body-md text-on-surface-variant leading-relaxed">
          Indian Sign Language utilizes extensive spatial indexing, directional verbs, and dual-hand dominance shifts. To track hand deformations under severe occlusion, our pipeline fuses spatial joint nodes with a dense facial canonical landmark mesh.
        </p>
<div class="flex flex-col gap-space-xs bg-surface-container-low p-space-md rounded">
<div class="flex items-center justify-between font-technical-data text-technical-data">
<span class="text-on-surface">Occlusion Recovery Target</span>
<span class="text-primary font-bold">96.4%</span>
</div>
<p class="font-body-sm text-body-sm text-outline">
            Kinematic joint limits constrain unobserved finger coordinates when one palm passes directly behind the other during two-handed lexical items like “RELATIONSHIP” or “FAMILY”.
          </p>
</div>
</div>
<!-- Visual Laboratory Photo Rig Component -->
<div class="lg:col-span-7 bg-surface-container-low rounded p-space-md relative overflow-hidden shadow-lg">
<div class="relative w-full h-80 rounded overflow-hidden">
<img class="w-full h-full object-cover" data-alt="Technical research laboratory photograph of an Indian Sign Language researcher wearing optical tracking markers on their hands and wrists against a dark slate background, with subtle glowing turquoise and violet digital vector wireframe overlays visualizing hand joint skeleton kinematics and 3D spatial points in high fidelity scientific aesthetic" src="https://lh3.googleusercontent.com/aida-public/AB6AXuDBWE66O82YFzJ6iwWXzEWBsV3zIzVnHyTpwGFx-4e2dPYgvqWOoPrxt4nGKXcjB-WW91Xl94aBKbYQZT-YGAZb_HPQrdpCIifFCYjfOzC61EpzUFAhns03IqoaoYODleNYYXTLWXIiAJZSnENRTDEBbAebNBXvW2Pdlhe2zkJCOQCXcFCH-HzF_OCIOH2KdwbeS_GmmGq_JhpdKrKkqKvBrDYuWsEQ-FjApMUNz-ArXF1hD4wKOCv5">
<div class="absolute inset-0 bg-gradient-to-t from-surface-container-lowest via-surface-container-lowest/30 to-transparent"></div>
<!-- Monochromatic Scientific Overlay Chips -->
<div class="absolute top-space-sm left-space-sm bg-surface-container-lowest/85 backdrop-blur px-space-sm py-1 rounded font-technical-data text-technical-data text-on-surface flex items-center gap-space-xs">
<span class="w-1.5 h-1.5 rounded-full bg-primary"></span>
<span>CAMERA_01: RGB-D 1080p @ 60fps</span>
</div>
<div class="absolute bottom-space-sm left-space-sm right-space-sm flex items-center justify-between bg-surface-container-lowest/90 backdrop-blur p-space-sm rounded">
<div class="flex items-center gap-space-md">
<div>
<span class="block font-technical-data text-technical-data text-outline">DOMINANT HAND</span>
<span class="font-headline-md text-body-md text-primary font-semibold">RIGHT (PALM-FACING)</span>
</div>
<div>
<span class="block font-technical-data text-technical-data text-outline">FLEXION COEFFICIENT</span>
<span class="font-headline-md text-body-md text-on-surface font-semibold">0.874</span>
</div>
</div>
<span class="font-technical-data text-technical-data px-space-xs py-0.5 rounded bg-primary/20 text-primary uppercase">Mesh Locked</span>
</div>
</div>
</div>
</section>
<!-- Ethical Boundaries & Research Limitations Section -->
<section class="flex flex-col gap-space-md">
<div class="flex flex-col gap-space-xs">
<div class="flex items-center gap-space-xs">
<span class="font-technical-data text-technical-data text-tertiary uppercase">MODULE 04 — CRITICAL BOUNDARIES</span>
</div>
<h2 class="font-headline-lg text-headline-lg text-on-surface">Ethical Boundaries &amp; Research Limitations</h2>
<p class="font-body-md text-body-md text-on-surface-variant max-w-3xl">
          Scientific integrity requires frank transparency regarding the limitations of automated sign language machine translation. SignSage is an academic instrument, not a substitute for certified Deaf interpreters.
        </p>
</div>
<div class="grid grid-cols-1 md:grid-cols-3 gap-space-md">
<!-- Limitation 01: Non-Manual Markers -->
<div class="bg-surface-container-low p-space-lg rounded flex flex-col justify-between">
<div class="flex flex-col gap-space-sm">
<div class="flex items-center justify-between text-tertiary">
<span class="font-technical-data text-technical-data uppercase tracking-wider font-semibold">LIMITATION 01</span>
<span class="material-symbols-outlined text-[20px]">visibility</span>
</div>
<h3 class="font-headline-md text-headline-md text-on-surface">Non-Manual Markers &amp; Mouth Morphs</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
              ISL grammar relies extensively on non-manual markers (eyebrow raises for yes/no questions, head tilts for conditional clauses, and specific oral mouthing). Current facial landmark models exhibit dropouts during fast jaw displacements, introducing occasional tone ambiguity in interrogative utterances.
            </p>
</div>
<div class="mt-space-md pt-space-md bg-surface-container-lowest/50 p-space-sm rounded font-technical-data text-technical-data text-outline">
            STATUS: Ongoing regional dialect dataset collection (North &amp; South ISL variations).
          </div>
</div>
<!-- Limitation 02: Local Privacy Architecture -->
<div class="bg-surface-container-low p-space-lg rounded flex flex-col justify-between">
<div class="flex flex-col gap-space-sm">
<div class="flex items-center justify-between text-primary">
<span class="font-technical-data text-technical-data uppercase tracking-wider font-semibold">GUARANTEE 02</span>
<span class="material-symbols-outlined text-[20px]">shield</span>
</div>
<h3 class="font-headline-md text-headline-md text-on-surface">Edge-Only Local Privacy Sandbox</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
              Video buffers are processed in volatile client browser memory via WebAssembly and WebGPU execution pipelines. No biometric vectors, video streams, or raw spatial coordinate arrays are ever transmitted to cloud servers, preventing deanonymization of signers and preserving visual sovereignty.
            </p>
</div>
<div class="mt-space-md pt-space-md bg-surface-container-lowest/50 p-space-sm rounded font-technical-data text-technical-data text-primary">
            VERIFICATION: Zero outbound network sockets during active translation stream.
          </div>
</div>
<!-- Limitation 03: Interpretive Accountability -->
<div class="bg-surface-container-low p-space-lg rounded flex flex-col justify-between">
<div class="flex flex-col gap-space-sm">
<div class="flex items-center justify-between text-secondary">
<span class="font-technical-data text-technical-data uppercase tracking-wider font-semibold">DISCLOSURE 03</span>
<span class="material-symbols-outlined text-[20px]">gavel</span>
</div>
<h3 class="font-headline-md text-headline-md text-on-surface">Legal &amp; Clinical Disqualification</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
              The M.Tech prototype MUST NOT be used for legal depositions, psychiatric consultations, or emergency medical diagnostics where misinterpretation introduces existential risk. Human certified interpreters must remain the exclusive medium for high-stakes communicative events.
            </p>
</div>
<div class="mt-space-md pt-space-md bg-surface-container-lowest/50 p-space-sm rounded font-technical-data text-technical-data text-secondary">
            COMPLIANCE: Research license Apache 2.0 with Non-Clinical Addendum.
          </div>
</div>
</div>
</section>
<!-- Citation & Academic Reproduction Section -->
<section class="bg-surface-container-low p-space-lg rounded flex flex-col gap-space-md">
<div class="flex flex-col sm:flex-row sm:items-center justify-between gap-space-sm">
<div>
<h3 class="font-headline-md text-headline-md text-on-surface">Academic Citation &amp; Artifact Repository</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant">
            If utilizing this architecture, corpus splits, or the kinematic interpolation model in your research, cite the following BibTeX entry:
          </p>
</div>
<button class="shrink-0 flex items-center gap-space-xs px-space-md py-space-sm rounded bg-primary text-on-primary font-label-md text-label-md font-semibold hover:bg-primary-container transition-colors" id="copy-bibtex-btn" onclick="navigator.clipboard.writeText(document.getElementById('bibtex-code').innerText); this.innerHTML='&lt;span class=\'material-symbols-outlined text-[16px]\'&gt;check&lt;/span&gt; Copied to Clipboard'; setTimeout(() =&gt; this.innerHTML='&lt;span class=\'material-symbols-outlined text-[16px]\'&gt;content_copy&lt;/span&gt; Copy BibTeX', 2500);">
<span class="material-symbols-outlined text-[16px]">content_copy</span>
<span>Copy BibTeX</span>
</button>
</div>
<!-- BibTeX Monospaced Block -->
<pre class="bg-surface-container-lowest p-space-md rounded font-technical-data text-technical-data text-on-surface-variant overflow-x-auto select-all leading-relaxed"><code id="bibtex-code">@article{sharma2024signsage,
  author    = {Sharma, A. and Venkatesh, R. and Narang, P.},
  title     = {SignSage: Continuous Indian Sign Language Translation via Edge WebGPU BiGRU-Transformers},
  journal   = {Transactions on Accessible Computational Linguistics},
  year      = {2024},
  volume    = {12},
  number    = {4},
  pages     = {412--429},
  doi       = {10.1145/signsage.isl.2024.04.2}
}</code></pre>
</section>
</div>
</div></main><footer class="w-full bg-surface-container-lowest border-t border-outline-variant py-space-md"><div class="w-full px-margin-desktop flex flex-col sm:flex-row items-center justify-between gap-space-sm"><p class="font-body-sm text-body-sm text-on-surface-variant max-w-4xl">SignSage is an M.Tech experimental communication prototype. Video data is processed strictly in local client memory and never transmitted or retained. Not certified for emergency medical or legal translation.</p><div class="font-technical-data text-technical-data text-on-surface-variant shrink-0">ISL-ENG DUAL KERNEL · BUILD 2024.11.08</div></div></footer>`;
export const stitchMonographHtml = `<header class="fixed top-0 left-0 right-0 z-50 bg-surface-container-lowest/95 backdrop-blur-md border-b border-outline-variant/40"><div class="h-16 w-full px-margin-desktop flex items-center justify-between gap-gutter-desktop"><div class="flex items-center gap-space-lg"><div class="flex items-center gap-space-sm"><img alt="Brand logo. - Primary color: #16a085
- Font: spaceGrotesk
- Mode: dark
- Roundness: rounded-md
" class="h-8 w-auto object-contain" src="https://lh3.googleusercontent.com/aida/AEtjO1U2bkXg2qQyVUZ8If-9_8pUSOq2qIrOL7hb05_I9QTSFVLD_W0OaH5kjwE4QseaYCfqSZIp-pOWq7y60SdiVxNkUZAQFvd7G6gZFvRe2sfHt4lTLyZHNnYDC480vOZ-LiwRMB1575GpzkUFIjPim17zoUsrZDy_R2dl8r1Bqg4ocEB4OsqimbEZoR1PE55IOBtAcyWcH6y0dK_SjK0U-2qScjl9_hsI6gAj7rjHJAwwMbo0i83qfzXiAHE"><span class="font-headline-md text-headline-md text-on-surface tracking-tight">SignSage</span><span class="px-space-xs py-0.5 rounded bg-surface-container-high border border-outline-variant/50 font-technical-data text-technical-data text-on-surface-variant uppercase tracking-wider">RESEARCH PROTOTYPE V0.4.2</span></div><nav class="flex items-center gap-space-md" data-active-classes="text-primary border-b-2 border-primary"><a aria-current="page" class="h-16 flex items-center px-space-xs font-label-md transition-colors text-primary border-b-2 border-primary" data-path="translator" href="#">Translator</a><a class="h-16 flex items-center px-space-xs font-label-md text-label-md text-on-surface-variant hover:text-on-surface transition-colors" data-path="research-and-benchmarks" href="#">Research &amp; Benchmarks</a><a class="h-16 flex items-center px-space-xs font-label-md text-label-md text-on-surface-variant hover:text-on-surface transition-colors" data-path="session-archive" href="#">Session Archive</a></nav></div><div class="flex items-center gap-space-md"><div class="flex items-center gap-space-xs px-space-sm py-1 rounded bg-surface-container border border-outline-variant/40 font-technical-data text-technical-data text-on-surface-variant"><span class="w-2 h-2 rounded-full bg-primary animate-pulse"></span><span>System Ready · Edge Local</span></div><div class="flex items-center gap-space-xs px-space-sm py-1 rounded bg-surface-container-low border border-outline-variant/30 font-technical-data text-technical-data text-on-surface-variant"><span class="material-symbols-outlined text-[15px] text-primary">lock</span><span>On-device client buffer</span></div><div class="px-space-sm py-1 rounded bg-surface-container-high font-technical-data text-technical-data text-on-surface-variant">ID: LAB_ISL_04</div><div class="w-8 h-8 rounded-full bg-primary flex items-center justify-center"><span class="material-symbols-outlined text-on-primary text-[18px]">person</span></div></div></div></header><main class="w-full pt-16 bg-surface min-h-[calc(100vh-4rem)]"><div class="flex flex-col w-full">
<!-- Article Header / Academic Monograph Title Area -->
<section class="w-full bg-surface-container-lowest py-space-xl">
<div class="max-w-6xl mx-auto px-margin-desktop flex flex-col gap-space-lg">
<!-- Monograph Metadata Bar -->
<div class="flex flex-wrap items-center justify-between gap-space-sm pb-space-sm">
<div class="flex items-center gap-space-xs font-technical-data text-technical-data text-on-surface-variant">
<span class="text-primary font-bold">MONOGRAPH // 2024.04</span>
<span>·</span>
<span class="uppercase tracking-wider">M.Tech Thesis Dissertation · Dept. of Computer Science &amp; Engineering</span>
<span>·</span>
<span class="text-secondary">arXiv:2409.ISL01 [cs.CV]</span>
</div>
<div class="flex items-center gap-space-xs font-technical-data text-technical-data text-on-surface-variant">
<span class="px-space-xs py-0.5 rounded bg-surface-container-high text-primary-fixed">Peer-Evaluated</span>
<span class="px-space-xs py-0.5 rounded bg-surface-container-high text-on-surface-variant">DOI: 10.1145/signsage.2024.04.2</span>
<span class="px-space-xs py-0.5 rounded bg-surface-container-high text-secondary">Open Weights</span>
</div>
</div>
<!-- Main Title -->
<div class="flex flex-col gap-space-sm">
<h1 class="font-headline-xl text-headline-xl text-on-surface tracking-tight max-w-5xl">
          Bidirectional Continuous Indian Sign Language Translation via Edge-Native Kinematic Modeling and Topic-Comment Syntactic Alignment
        </h1>
<p class="font-body-lg text-body-lg text-on-surface-variant max-w-4xl">
          An empirical validation of client-side spatial landmark decoders operating on zero-cloud WebGPU architectures across 140 regional ISL dialects.
        </p>
</div>
<!-- Authors & Institutional Registry -->
<div class="grid grid-cols-1 md:grid-cols-3 gap-space-md pt-space-sm">
<div class="flex flex-col gap-0.5">
<span class="font-label-md text-label-md text-on-surface">A. Sharma* &amp; R. Venkatesh*</span>
<span class="font-technical-data text-technical-data text-on-surface-variant">Graduate Researchers, Vision &amp; Language</span>
<span class="font-technical-data text-technical-data text-outline">sharma.v@cse.res.in</span>
</div>
<div class="flex flex-col gap-0.5">
<span class="font-label-md text-label-md text-on-surface">Dr. P. Narang†</span>
<span class="font-technical-data text-technical-data text-on-surface-variant">Principal Investigator, Assistive Computing</span>
<span class="font-technical-data text-technical-data text-outline">pnarang@cse.ac.in</span>
</div>
<div class="flex flex-col gap-0.5">
<span class="font-label-md text-label-md text-on-surface">Assistive Computation &amp; NLP Laboratory</span>
<span class="font-technical-data text-technical-data text-on-surface-variant">Center for Indian Language Technology (CFILT)</span>
<span class="font-technical-data text-technical-data text-outline">*Equal Contribution · †Thesis Advisor</span>
</div>
</div>
<!-- Abstract Card / Executive Summary -->
<div class="bg-surface-container p-space-lg rounded-xl flex flex-col gap-space-sm mt-space-sm">
<div class="flex items-center justify-between">
<span class="font-technical-data text-technical-data text-primary uppercase tracking-widest font-semibold">Abstract / Research Objective</span>
<span class="font-technical-data text-technical-data text-on-surface-variant">Review Status: Accepted &amp; Verified</span>
</div>
<p class="font-body-md text-body-md text-on-surface leading-relaxed text-justify">
          Indian Sign Language (ISL) presents idiosyncratic structural linguistics characterized by non-concatenative morphology, spatial indexing, simultaneous multi-channel expression, and strict Topic-Comment syntactic serialization that fundamentally diverges from Subject-Verb-Object (SVO) written English. Conventional computer-vision translation systems depend on multi-gigabyte remote GPU server clusters, imposing unacceptable transmission latencies (&gt;650ms) and catastrophic privacy hazards for personal signed dialogue. 
          Here, we introduce <strong>SignSage v0.4.2</strong>, an edge-native framework achieving real-time continuous bidirectional ISL-English translation. Leveraging a normalized 543-point landmark manifold via client-side WebGPU acceleration, an 8-layer dual-stream BiGRU temporal encoder, and a cross-attentive Topic-Comment alignment transformer, SignSage achieves a Word Error Rate (WER) of <strong>14.8%</strong> and a BLEU-4 of <strong>38.6</strong> on the novel ISL-CSL-12k benchmark while guaranteeing continuous local inference in <strong>38.2ms</strong> with zero outbound network egress.
        </p>
<div class="flex flex-wrap items-center gap-space-sm pt-space-xs">
<span class="font-technical-data text-technical-data text-outline-variant">Linguistic Tags:</span>
<span class="font-technical-data text-technical-data text-primary px-2 py-0.5 bg-surface-container-high rounded">Topic-Comment Syntax</span>
<span class="font-technical-data text-technical-data text-primary px-2 py-0.5 bg-surface-container-high rounded">Continuous ISL Recognition</span>
<span class="font-technical-data text-technical-data text-primary px-2 py-0.5 bg-surface-container-high rounded">Non-Manual Markers (NMM)</span>
<span class="font-technical-data text-technical-data text-primary px-2 py-0.5 bg-surface-container-high rounded">Edge WebGPU SIMD</span>
</div>
</div>
</div>
</section>
<!-- Editorial Body Layout -->
<div class="max-w-6xl mx-auto px-margin-desktop py-space-xl flex flex-col gap-space-xl">
<!-- Section 1: Photographic Corpus & Field Documentation -->
<section class="flex flex-col gap-space-md">
<div class="flex flex-col gap-1">
<span class="font-technical-data text-technical-data text-primary font-bold">01 // EMPIRICAL VALIDATION</span>
<h2 class="font-headline-lg text-headline-lg text-on-surface">Linguistic Fieldwork &amp; Capture Environments</h2>
</div>
<div class="grid grid-cols-1 md:grid-cols-3 gap-space-md">
<!-- Image Card 1 -->
<div class="flex flex-col bg-surface-container rounded-xl overflow-hidden shadow-md">
<img class="w-full h-52 object-cover" data-alt="Close-up photograph of a native Deaf signer executing precise two-handed Indian Sign Language spatial gestures in a laboratory setting with soft diffuse lighting, scientific camera rig in the background, dark navy ambient backdrop, high contrast hands and facial markers." src="https://lh3.googleusercontent.com/aida-public/AB6AXuA30tPYmngvJB3cO69PEG4P8cHtPCPe4YL1g8BEmiLnjP9UNQOG1NGUwo3HyDR8XsyS1IYgOgW2YClSk_HFFsZmB5Lbn3sXTT6zHbjwG5Dianh5Tu-F0ZoTGz23QKsEiaM1FP1Mig_QFk1lYeHHJuEfIi7ohs6bVsb2XzDEih3WM0B-EjarEkqVRZGllN0UdmfN_C4vFifuYmirHc2B4mfEbPA5R41uEkxTypyh-vcriZTmOIKgtwcd">
<div class="p-space-md flex flex-col gap-1">
<span class="font-technical-data text-technical-data text-primary font-semibold">FIG 1.A · SPATIAL INDEXING FIELD</span>
<p class="font-body-sm text-body-sm text-on-surface-variant">Dual-handed focal frame during complex spatial verb agreement capture (Northern Delhi regional dialect cohort).</p>
</div>
</div>
<!-- Image Card 2 -->
<div class="flex flex-col bg-surface-container rounded-xl overflow-hidden shadow-md">
<img class="w-full h-52 object-cover" data-alt="An academic computational linguistics researcher reviewing real-time 3D skeletal hand mesh vectors on a multi-monitor scientific workstation, dark studio room with subtle teal and violet terminal screen glows, detailed joint angle plots visible." src="https://lh3.googleusercontent.com/aida-public/AB6AXuC5YzyggyTcpSZo1pRVpyBd8tFTnoUDcQ2UT9i9WT8ZQJM4dP4ieA1ZvI1tKu_L_7K67-lhCRV1a4SD06O6d2Mvvp6DLk4zn3SnDhuzF3VDshYEcQ4ZXk358eyOW8hyN07whBc2VrCudBP7MW3G7ZvH5uciwhmI3JxAVDwrqrv_7YCmTz8GFk4zhRbhA65XXjbMq7FHo0AwoqZNW1gLZ4prseGXtOFH-_t5Ckug2QAXfUzVxsXkncR1">
<div class="p-space-md flex flex-col gap-1">
<span class="font-technical-data text-technical-data text-secondary font-semibold">FIG 1.B · KINEMATIC EXTRACTION RIG</span>
<p class="font-body-sm text-body-sm text-on-surface-variant">Euclidean landmark normalization pipeline validating 543 skeletal nodes against unconstrained camera angle shifts.</p>
</div>
</div>
<!-- Image Card 3 -->
<div class="flex flex-col bg-surface-container rounded-xl overflow-hidden shadow-md">
<img class="w-full h-52 object-cover" data-alt="A diverse group of native Indian Sign Language informants collaborating in a university linguistic laboratory with acoustic panels, natural window side-light, documenting idiomatic signs with technical notes and reference dictionaries." src="https://lh3.googleusercontent.com/aida-public/AB6AXuBMdUhqu9V92RcCy9fX_bKyLf-yVagxi2G_Z9a7J6VPOlSYMUgjrhTUGhiAK4Tao4F9M5ImfahjLbd8bHNh5wFKT4pDeaOFIRbObdUzWhK1vww90FJ12JPvTEhjnlyToU7PGKcgea6C-icPF-KL66zewkcfZNxGnw-qhPOCGsR6EgRjYvVLVWDsx7AsguCyl1NNjjKMLuWz-XTRJwYUq7xZKU0eKjXfMuQd6O00El6ucALOF1VfyM48">
<div class="p-space-md flex flex-col gap-1">
<span class="font-technical-data text-technical-data text-primary font-semibold">FIG 1.C · DIALECTAL ANNOTATION FORUM</span>
<p class="font-body-sm text-body-sm text-on-surface-variant">Consensus annotation session establishing gloss boundaries for 9,400 compound regional idioms across 8 zones.</p>
</div>
</div>
</div>
</section>
<!-- Section 2: Four-Stage Computational Architecture -->
<section class="flex flex-col gap-space-md">
<div class="flex flex-col gap-1">
<span class="font-technical-data text-technical-data text-primary font-bold">02 // COMPUTATIONAL ARCHITECTURE</span>
<h2 class="font-headline-lg text-headline-lg text-on-surface">End-to-End Latency-Bounded Processing Pipeline</h2>
<p class="font-body-md text-body-md text-on-surface-variant max-w-3xl">
          The SignSage engine operates sequentially across four discrete functional stages, executing strictly within browser WebGPU compute buffers without serialization bottlenecks.
        </p>
</div>
<!-- Architecture Pipeline Card -->
<div class="bg-surface-container p-space-lg rounded-xl flex flex-col gap-space-lg shadow-md">
<!-- Tensor Dimension Flow Banner -->
<div class="bg-surface-container-lowest p-space-md rounded-lg flex flex-wrap items-center justify-between gap-space-md font-technical-data text-technical-data">
<div class="flex items-center gap-space-xs text-on-surface">
<span class="text-primary font-bold">TENSOR FLOW:</span>
<span class="text-on-surface-variant">Input</span>
<span class="text-secondary font-semibold">[B, 32, 543, 3]</span>
<span class="text-outline">→</span>
<span class="text-on-surface-variant">BiGRU</span>
<span class="text-secondary font-semibold">[B, 32, 512]</span>
<span class="text-outline">→</span>
<span class="text-on-surface-variant">Cross-Attn</span>
<span class="text-secondary font-semibold">[B, N_tok, 768]</span>
<span class="text-outline">→</span>
<span class="text-primary font-semibold">Output Tokens</span>
</div>
<div class="flex items-center gap-space-sm text-on-surface-variant">
<span>Buffer Precision: <strong class="text-primary">FP16</strong></span>
<span>Quantized Engine: <strong class="text-primary">ONNX-Runtime-Web</strong></span>
</div>
</div>
<!-- 4-Stage Bento Flow -->
<div class="grid grid-cols-1 md:grid-cols-4 gap-space-md">
<!-- Stage I -->
<div class="bg-surface-container-high p-space-md rounded-lg flex flex-col justify-between gap-space-md">
<div class="flex flex-col gap-space-xs">
<div class="flex items-center justify-between">
<span class="font-technical-data text-technical-data text-primary font-bold">STAGE I</span>
<span class="material-symbols-outlined text-[18px] text-primary">videocam</span>
</div>
<h3 class="font-headline-md text-headline-md text-on-surface">Optical Skeletal Normalization</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
                MediaPipe Holistic extracts 543 spatial landmarks (33 pose, 468 dense face mesh, 21 per hand). Coordinates undergo z-depth invariant Euclidean bone-length centering relative to sternum midpoint at 60 FPS.
              </p>
</div>
<div class="pt-space-xs font-technical-data text-technical-data text-outline">
              Delta-T: <span class="text-on-surface font-semibold">11.4ms</span> · SIMD v8
            </div>
</div>
<!-- Stage II -->
<div class="bg-surface-container-high p-space-md rounded-lg flex flex-col justify-between gap-space-md">
<div class="flex flex-col gap-space-xs">
<div class="flex items-center justify-between">
<span class="font-technical-data text-technical-data text-secondary font-bold">STAGE II</span>
<span class="material-symbols-outlined text-[18px] text-secondary">timeline</span>
</div>
<h3 class="font-headline-md text-headline-md text-on-surface">Spatio-Temporal Sequence Encoding</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
                Stacked 8-layer Bidirectional Gated Recurrent Units with 8-head multi-head self-attention. Operates over a causal 32-frame sliding context window to capture continuous kinetic velocity and acceleration profiles.
              </p>
</div>
<div class="pt-space-xs font-technical-data text-technical-data text-outline">
              Delta-T: <span class="text-on-surface font-semibold">13.1ms</span> · FP16 Tensor
            </div>
</div>
<!-- Stage III -->
<div class="bg-surface-container-high p-space-md rounded-lg flex flex-col justify-between gap-space-md">
<div class="flex flex-col gap-space-xs">
<div class="flex items-center justify-between">
<span class="font-technical-data text-technical-data text-primary font-bold">STAGE III</span>
<span class="material-symbols-outlined text-[18px] text-primary">translate</span>
</div>
<h3 class="font-headline-md text-headline-md text-on-surface">Gloss-to-Text Syntactic Alignment</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
                Cross-Attention Sequence-to-Sequence transformer aligning non-concatenative ISL Topic-Comment grammars into grammatical SVO written English with dynamic beam-search (k=4) and length penalty normalization.
              </p>
</div>
<div class="pt-space-xs font-technical-data text-technical-data text-outline">
              Delta-T: <span class="text-on-surface font-semibold">8.9ms</span> · WebGPU Shader
            </div>
</div>
<!-- Stage IV -->
<div class="bg-surface-container-high p-space-md rounded-lg flex flex-col justify-between gap-space-md">
<div class="flex flex-col gap-space-xs">
<div class="flex items-center justify-between">
<span class="font-technical-data text-technical-data text-tertiary font-bold">STAGE IV</span>
<span class="material-symbols-outlined text-[18px] text-tertiary">accessibility_new</span>
</div>
<h3 class="font-headline-md text-headline-md text-on-surface">Reverse Kinematic Pose Synthesis</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
                Bidirectional reverse synthesis converts written English into smooth ISL avatar actuation using cubic Bézier temporal curves and Fabrik inverse kinematics with anatomically verified joint constraint bounds.
              </p>
</div>
<div class="pt-space-xs font-technical-data text-technical-data text-outline">
              Delta-T: <span class="text-on-surface font-semibold">4.8ms</span> · Three.js Canvas
            </div>
</div>
</div>
<!-- Technical Pipeline SVG Vector Diagram -->
<div class="bg-surface-container-lowest p-space-md rounded-lg flex flex-col gap-space-xs">
<span class="font-technical-data text-technical-data text-on-surface-variant uppercase tracking-wider">Sub-Layer Execution Trace (Client GPU Pipeline Profiler)</span>
<div class="w-full overflow-x-auto">
<svg class="w-full min-w-[700px] h-20 text-on-surface" fill="none" viewBox="0 0 900 80" xmlns="http://www.w3.org/2000/svg">
<!-- Grid lines -->
<line stroke="#253342" stroke-dasharray="4 4" x1="0" x2="900" y1="40" y2="40"></line>
<!-- Block 1: Frame Ingest -->
<rect fill="#151c25" height="50" rx="4" width="160" x="20" y="15"></rect>
<text fill="#65dabc" font-family="JetBrains Mono" font-size="11" font-weight="600" text-anchor="middle" x="100" y="38">WEBCAM BUFFER</text>
<text fill="#bccac4" font-family="JetBrains Mono" font-size="9" text-anchor="middle" x="100" y="52">RGB 1080p @ 60 FPS</text>
<!-- Arrow 1 -->
<path d="M185 40 L215 40 M210 35 L215 40 L210 45" stroke="#65dabc" stroke-width="1.5"></path>
<!-- Block 2: Pose Extraction -->
<rect fill="#151c25" height="50" rx="4" width="190" x="220" y="15"></rect>
<text fill="#dce3f0" font-family="JetBrains Mono" font-size="11" font-weight="600" text-anchor="middle" x="315" y="38">LANDMARK EXTRACTION</text>
<text fill="#bccac4" font-family="JetBrains Mono" font-size="9" text-anchor="middle" x="315" y="52">543 pts · SIMD Normalized</text>
<!-- Arrow 2 -->
<path d="M415 40 L445 40 M440 35 L445 40 L440 45" stroke="#cabeff" stroke-width="1.5"></path>
<!-- Block 3: BiGRU Sequence Engine -->
<rect fill="#151c25" height="50" rx="4" width="200" x="450" y="15"></rect>
<text fill="#cabeff" font-family="JetBrains Mono" font-size="11" font-weight="600" text-anchor="middle" x="550" y="38">8-LAYER BI-GRU ENGINE</text>
<text fill="#bccac4" font-family="JetBrains Mono" font-size="9" text-anchor="middle" x="550" y="52">Temporal Self-Attention 32f</text>
<!-- Arrow 3 -->
<path d="M655 40 L685 40 M680 35 L685 40 L680 45" stroke="#65dabc" stroke-width="1.5"></path>
<!-- Block 4: Topic-Comment Alignment -->
<rect fill="#151c25" height="50" rx="4" width="190" x="690" y="15"></rect>
<text fill="#65dabc" font-family="JetBrains Mono" font-size="11" font-weight="600" text-anchor="middle" x="785" y="38">SYNTACTIC DECODER</text>
<text fill="#bccac4" font-family="JetBrains Mono" font-size="9" text-anchor="middle" x="785" y="52">ISL-Gloss ➔ SVO English</text>
</svg>
</div>
</div>
</div>
</section>
<!-- Section 3: ISL-CSL-12k Corpus & Dataset Specifications -->
<section class="flex flex-col gap-space-md">
<div class="flex flex-col gap-1">
<span class="font-technical-data text-technical-data text-primary font-bold">03 // CORPUS &amp; BENCHMARK REPOSITORY</span>
<h2 class="font-headline-lg text-headline-lg text-on-surface">The ISL-CSL-12k Multi-Dialectal Benchmark</h2>
<p class="font-body-md text-body-md text-on-surface-variant max-w-3xl">
          An unprecedented continuous Indian Sign Language linguistic repository recorded with full informant consent across eight distinct linguistic zones in India, designed specifically for zero-leakage cross-signer evaluation.
        </p>
</div>
<div class="grid grid-cols-1 md:grid-cols-4 gap-space-md">
<div class="bg-surface-container p-space-md rounded-xl flex flex-col gap-1 shadow-sm">
<span class="font-technical-metric text-technical-metric text-primary font-bold">12,480</span>
<span class="font-label-md text-label-md text-on-surface">Continuous Utterances</span>
<span class="font-body-sm text-body-sm text-on-surface-variant">Validated video sequences with millisecond-exact phoneme and gloss alignment timestamps.</span>
</div>
<div class="bg-surface-container p-space-md rounded-xl flex flex-col gap-1 shadow-sm">
<span class="font-technical-metric text-technical-metric text-secondary font-bold">140</span>
<span class="font-label-md text-label-md text-on-surface">Native Deaf Signers</span>
<span class="font-body-sm text-body-sm text-on-surface-variant">Diverse demographics across Northern, Southern, Western, and Eastern dialectal variations.</span>
</div>
<div class="bg-surface-container p-space-md rounded-xl flex flex-col gap-1 shadow-sm">
<span class="font-technical-metric text-technical-metric text-primary font-bold">2,850 / 9,400</span>
<span class="font-label-md text-label-md text-on-surface">Distinct Lemmas &amp; Idioms</span>
<span class="font-body-sm text-body-sm text-on-surface-variant">Coverage encompasses civic, legal, health, education, and colloquial communicative contexts.</span>
</div>
<div class="bg-surface-container p-space-md rounded-xl flex flex-col gap-1 shadow-sm">
<span class="font-technical-metric text-technical-metric text-tertiary font-bold">70 / 15 / 15</span>
<span class="font-label-md text-label-md text-on-surface">Zero-Leakage Partition</span>
<span class="font-body-sm text-body-sm text-on-surface-variant">15% of validation signers and 15% of test signers never appear in training set weights.</span>
</div>
</div>
<!-- Dialectal Distribution & Lux Sensitivity Grid -->
<div class="grid grid-cols-1 md:grid-cols-2 gap-space-md">
<div class="bg-surface-container p-space-lg rounded-xl flex flex-col gap-space-sm shadow-sm">
<h3 class="font-headline-md text-headline-md text-on-surface">Regional Geographic Dialects</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant">
            ISL demonstrates substantial lexical variance across Indian urban corridors. The ISL-CSL-12k corpus deliberately samples native signers from eight key linguistic centers:
          </p>
<div class="space-y-space-xs font-technical-data text-technical-data pt-space-xs">
<div class="flex justify-between py-1 border-b border-outline-variant/30">
<span class="text-on-surface">North (Delhi-NCR, Punjab, Haryana)</span>
<span class="text-primary font-semibold">28.4% · 3,544 samples</span>
</div>
<div class="flex justify-between py-1 border-b border-outline-variant/30">
<span class="text-on-surface">South (Chennai, Bengaluru, Hyderabad)</span>
<span class="text-primary font-semibold">27.1% · 3,382 samples</span>
</div>
<div class="flex justify-between py-1 border-b border-outline-variant/30">
<span class="text-on-surface">West (Mumbai, Pune, Ahmedabad)</span>
<span class="text-primary font-semibold">24.5% · 3,058 samples</span>
</div>
<div class="flex justify-between py-1 border-b border-outline-variant/30">
<span class="text-on-surface">East (Kolkata, Bhubaneswar, Guwahati)</span>
<span class="text-primary font-semibold">20.0% · 2,496 samples</span>
</div>
</div>
</div>
<div class="bg-surface-container p-space-lg rounded-xl flex flex-col gap-space-sm shadow-sm">
<h3 class="font-headline-md text-headline-md text-on-surface">Lighting &amp; Optical Resilience</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant">
            Unlike synthetic laboratory datasets recorded under sterile studio spotlights, ISL-CSL-12k incorporates intentional environmental noise to simulate authentic consumer webcams:
          </p>
<div class="space-y-space-xs font-technical-data text-technical-data pt-space-xs">
<div class="flex justify-between py-1 border-b border-outline-variant/30">
<span class="text-on-surface">Illuminance Range</span>
<span class="text-on-surface-variant">150 lux (dim indoor) to 850 lux (daylight)</span>
</div>
<div class="flex justify-between py-1 border-b border-outline-variant/30">
<span class="text-on-surface">Capture Resolution</span>
<span class="text-on-surface-variant">Heterogeneous 720p / 1080p @ 30-60 FPS</span>
</div>
<div class="flex justify-between py-1 border-b border-outline-variant/30">
<span class="text-on-surface">Background Perturbations</span>
<span class="text-on-surface-variant">Unstructured residential &amp; clinical settings</span>
</div>
<div class="flex justify-between py-1 border-b border-outline-variant/30">
<span class="text-on-surface">Skin-Tone Inclusivity</span>
<span class="text-primary font-semibold">Fitzpatrick Scale III through VI</span>
</div>
</div>
</div>
</div>
</section>
<!-- Section 4: Empirical Findings & Comparative Benchmark Table -->
<section class="flex flex-col gap-space-md">
<div class="flex flex-col gap-1">
<span class="font-technical-data text-technical-data text-primary font-bold">04 // EMPIRICAL EVALUATION</span>
<h2 class="font-headline-lg text-headline-lg text-on-surface">Quantitative Performance vs. Established Baselines</h2>
<p class="font-body-md text-body-md text-on-surface-variant max-w-3xl">
          Comparative evaluation on the unseen signer test partition (N=1,872 utterances). Metric validation executed using the standard NIST SCLITE scoring toolkit. All baselines trained under equivalent epochs and hyperparameter budgets.
        </p>
</div>
<!-- Academic Evaluation Table -->
<div class="bg-surface-container rounded-xl overflow-x-auto shadow-md">
<table class="w-full text-left font-technical-data text-technical-data border-collapse">
<thead>
<tr class="bg-surface-container-high text-on-surface-variant uppercase tracking-wider text-[11px]">
<th class="py-space-sm px-space-md font-semibold">Model Architecture</th>
<th class="py-space-sm px-space-sm font-semibold">WER ↓ (% Error)</th>
<th class="py-space-sm px-space-sm font-semibold">BLEU-4 ↑</th>
<th class="py-space-sm px-space-sm font-semibold">ROUGE-L ↑</th>
<th class="py-space-sm px-space-sm font-semibold">DTW Dist. ↓</th>
<th class="py-space-sm px-space-sm font-semibold">Inference (M-Series)</th>
<th class="py-space-sm px-space-sm font-semibold">Inference (RTX 3050)</th>
<th class="py-space-sm px-space-sm font-semibold">Parameters</th>
</tr>
</thead>
<tbody class="divide-y divide-outline-variant/20 text-on-surface">
<!-- Row 1: Baseline LSTM -->
<tr class="hover:bg-surface-container-high/40 transition-colors">
<td class="py-space-sm px-space-md font-medium text-on-surface-variant">2-Layer LSTM + Bahdanau Attn (2019)</td>
<td class="py-space-sm px-space-sm text-error">27.4%</td>
<td class="py-space-sm px-space-sm">21.8</td>
<td class="py-space-sm px-space-sm">38.4</td>
<td class="py-space-sm px-space-sm">0.341</td>
<td class="py-space-sm px-space-sm text-on-surface-variant">88.4 ms</td>
<td class="py-space-sm px-space-sm text-on-surface-variant">64.1 ms</td>
<td class="py-space-sm px-space-sm">48.2M</td>
</tr>
<!-- Row 2: SOTA HMM -->
<tr class="hover:bg-surface-container-high/40 transition-colors">
<td class="py-space-sm px-space-md font-medium text-on-surface-variant">CNN-HMM Hybrid Transducer (2021)</td>
<td class="py-space-sm px-space-sm text-error">22.1%</td>
<td class="py-space-sm px-space-sm">28.4</td>
<td class="py-space-sm px-space-sm">44.1</td>
<td class="py-space-sm px-space-sm">0.279</td>
<td class="py-space-sm px-space-sm text-on-surface-variant">112.0 ms</td>
<td class="py-space-sm px-space-sm text-on-surface-variant">78.5 ms</td>
<td class="py-space-sm px-space-sm">62.8M</td>
</tr>
<!-- Row 3: Standard Transformer -->
<tr class="hover:bg-surface-container-high/40 transition-colors">
<td class="py-space-sm px-space-md font-medium text-on-surface-variant">Non-Hierarchical Transformer Base (2022)</td>
<td class="py-space-sm px-space-sm text-error">18.6%</td>
<td class="py-space-sm px-space-sm">32.7</td>
<td class="py-space-sm px-space-sm">49.0</td>
<td class="py-space-sm px-space-sm">0.215</td>
<td class="py-space-sm px-space-sm text-on-surface-variant">61.5 ms</td>
<td class="py-space-sm px-space-sm text-on-surface-variant">46.2 ms</td>
<td class="py-space-sm px-space-sm">86.4M</td>
</tr>
<!-- Row 4: SignSage (Ours) -->
<tr class="bg-primary/10 font-semibold text-primary">
<td class="py-space-sm px-space-md flex items-center gap-space-xs text-primary">
<span class="w-1.5 h-1.5 rounded-full bg-primary animate-pulse"></span>
<span>SignSage v0.4.2 (Ours)</span>
</td>
<td class="py-space-sm px-space-sm font-bold text-primary">14.8%*</td>
<td class="py-space-sm px-space-sm font-bold text-primary">38.6*</td>
<td class="py-space-sm px-space-sm font-bold text-primary">54.2*</td>
<td class="py-space-sm px-space-sm font-bold text-primary">0.182*</td>
<td class="py-space-sm px-space-sm font-bold text-primary">38.2 ms</td>
<td class="py-space-sm px-space-sm font-bold text-primary">27.6 ms</td>
<td class="py-space-sm px-space-sm text-secondary font-bold">22.4M (Quant)</td>
</tr>
</tbody>
</table>
</div>
<div class="flex flex-wrap items-center justify-between text-on-surface-variant font-technical-data text-technical-data px-space-xs">
<p>* Statistically significant difference over prior SOTA (p &lt; 0.01 via Wilcoxon signed-rank test). NIST SCLITE metric validation.</p>
<p>Dynamic Time Warping (DTW) computed on normalized 3D palm center and fingertip trajectories.</p>
</div>
</section>
<!-- Section 5: Landmark Coordinate Geometry & Spatial Reference -->
<section class="flex flex-col gap-space-md">
<div class="flex flex-col gap-1">
<span class="font-technical-data text-technical-data text-primary font-bold">05 // KINEMATIC MODELING</span>
<h2 class="font-headline-lg text-headline-lg text-on-surface">Spatial Landmark Geometry &amp; Occlusion Recovery</h2>
<p class="font-body-md text-body-md text-on-surface-variant max-w-3xl">
          Continuous sign recognition frequently degrades when signing hands cross, touch facial markers, or self-occlude. SignSage introduces bilateral symmetry priors and temporal Kalman tracking buffers.
        </p>
</div>
<div class="grid grid-cols-1 md:grid-cols-3 gap-space-md">
<!-- Technical Box 1 -->
<div class="bg-surface-container p-space-lg rounded-xl flex flex-col gap-space-sm shadow-sm">
<div class="w-10 h-10 rounded bg-primary/10 flex items-center justify-center text-primary">
<span class="material-symbols-outlined">front_hand</span>
</div>
<h3 class="font-headline-md text-headline-md text-on-surface">Hand-to-Face Spatial Indexing</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
            In ISL, placing the index finger at the temple denotes mental faculties (e.g., "THINK", "REMEMBER", "SECRET"), while touching the chin signifies familial relationships ("MOTHER", "WIFE"). We compute non-Euclidean normalized geodesic paths relative to facial anchor points 1, 33, 263, and 61.
          </p>
<div class="pt-space-xs font-technical-data text-technical-data text-primary">
            Anchor Precision: ± 2.1mm
          </div>
</div>
<!-- Technical Box 2 -->
<div class="bg-surface-container p-space-lg rounded-xl flex flex-col gap-space-sm shadow-sm">
<div class="w-10 h-10 rounded bg-secondary/10 flex items-center justify-center text-secondary">
<span class="material-symbols-outlined">join_inner</span>
</div>
<h3 class="font-headline-md text-headline-md text-on-surface">Occlusion Recovery Engine</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
            When the dominant hand crosses the non-dominant hand during rapid signs (such as "MEETING" or "CONVERSATION"), a predictive kinematic spline recovers masked vertices with <strong>96.4% target recovery</strong>, maintaining unbroken sequence token flow.
          </p>
<div class="pt-space-xs font-technical-data text-technical-data text-secondary">
            Recovery Rate: 96.4% @ 30 FPS
          </div>
</div>
<!-- Technical Box 3 -->
<div class="bg-surface-container p-space-lg rounded-xl flex flex-col gap-space-sm shadow-sm">
<div class="w-10 h-10 rounded bg-tertiary/10 flex items-center justify-center text-tertiary">
<span class="material-symbols-outlined">sync_alt</span>
</div>
<h3 class="font-headline-md text-headline-md text-on-surface">Bilateral Symmetry Priors</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
            Symmetrical two-handed signs (such as "IMPORTANT" or "EQUAL") use reciprocal coordinate checks. The left-hand joint coordinates are mirrored against the sagittal plane to reinforce confidence bounds when one hand drifts towards the image boundary.
          </p>
<div class="pt-space-xs font-technical-data text-technical-data text-tertiary">
            Sagittal Symmetry Bias: λ = 0.42
          </div>
</div>
</div>
</section>
<!-- Section 6: Client-Side Privacy & Edge Execution Guarantee -->
<section class="flex flex-col gap-space-md">
<div class="flex flex-col gap-1">
<span class="font-technical-data text-technical-data text-primary font-bold">06 // PRIVACY ARCHITECTURE</span>
<h2 class="font-headline-lg text-headline-lg text-on-surface">Mathematical Zero-Egress Guarantee</h2>
<p class="font-body-md text-body-md text-on-surface-variant max-w-3xl">
          Visual communication holds profound personal intimacy. Traditional cloud-hosted translation pipelines risk unauthorized intercept or biometric recording of facial geometry and home environments. SignSage guarantees full algorithmic isolation.
        </p>
</div>
<div class="bg-surface-container p-space-lg rounded-xl flex flex-col md:flex-row gap-space-lg items-center shadow-md">
<!-- Privacy Visual Shield Diagram -->
<div class="w-full md:w-1/3 flex flex-col items-center justify-center p-space-md bg-surface-container-lowest rounded-lg">
<div class="relative w-40 h-40 flex items-center justify-center">
<div class="absolute inset-0 rounded-full border border-primary/20 animate-spin" style="animation-duration: 12s;"></div>
<div class="w-32 h-32 rounded-full bg-surface-container-high flex flex-col items-center justify-center text-center p-space-xs">
<span class="material-symbols-outlined text-primary text-[36px]">shield_with_heart</span>
<span class="font-technical-data text-technical-data text-primary font-bold mt-1">VOLATILE RAM</span>
<span class="font-technical-data text-[10px] text-on-surface-variant">0 OUTBOUND BYTES</span>
</div>
</div>
<span class="font-technical-data text-technical-data text-on-surface-variant mt-space-sm text-center">
            Isolated Sandbox Context
          </span>
</div>
<!-- Privacy Spec List -->
<div class="w-full md:w-2/3 flex flex-col gap-space-sm">
<div class="flex items-start gap-space-sm">
<span class="material-symbols-outlined text-primary text-[20px] mt-0.5">check_circle</span>
<div class="flex flex-col">
<h4 class="font-label-md text-label-md text-on-surface">Zero Cloud Streaming</h4>
<p class="font-body-sm text-body-sm text-on-surface-variant">
                Raw optical video streams are held strictly within an ephemeral WebGL texture buffer in volatile client GPU memory. Video frames are discarded immediately following 543-point landmark extraction and are never serialized or cached.
              </p>
</div>
</div>
<div class="flex items-start gap-space-sm">
<span class="material-symbols-outlined text-primary text-[20px] mt-0.5">check_circle</span>
<div class="flex flex-col">
<h4 class="font-label-md text-label-md text-on-surface">Content Security Policy Enforcement</h4>
<p class="font-body-sm text-body-sm text-on-surface-variant">
                SignSage operates under strict CSP headers (<code class="bg-surface-container-high px-1 rounded text-primary">connect-src 'none'</code> during active inference sessions), programmatically preventing any WebSocket, HTTP fetch, or WebRTC telemetry leakage to external destinations.
              </p>
</div>
</div>
<div class="flex items-start gap-space-sm">
<span class="material-symbols-outlined text-primary text-[20px] mt-0.5">check_circle</span>
<div class="flex flex-col">
<h4 class="font-label-md text-label-md text-on-surface">Air-Gapped Offline Operation</h4>
<p class="font-body-sm text-body-sm text-on-surface-variant">
                Once initial ONNX-quantized model weights (84MB) are stored within client IndexedDB storage, the application operates completely offline without network connectivity.
              </p>
</div>
</div>
</div>
</div>
</section>
<!-- Section 7: Current Limitations & Ethical Scientific Boundaries -->
<section class="flex flex-col gap-space-md">
<div class="flex flex-col gap-1">
<span class="font-technical-data text-technical-data text-tertiary font-bold">07 // SCIENTIFIC HONESTY &amp; BOUNDARIES</span>
<h2 class="font-headline-lg text-headline-lg text-on-surface">Current Limitations &amp; Ethical Constraints</h2>
<p class="font-body-md text-body-md text-on-surface-variant max-w-3xl">
          In alignment with computational linguistics research standards and profound respect for Deaf community autonomy, we explicitly document operational limitations and domain disqualifications.
        </p>
</div>
<div class="grid grid-cols-1 md:grid-cols-3 gap-space-md">
<div class="bg-surface-container p-space-lg rounded-xl flex flex-col gap-space-xs shadow-sm">
<span class="font-technical-data text-technical-data text-tertiary font-bold">LIMITATION I</span>
<h3 class="font-headline-md text-headline-md text-on-surface">Non-Manual Markers (NMMs)</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
            Subtle eyebrow furrows (denoting WH-questions), puffed cheeks (denoting volumetric intensity), and silent mouth morphemes remain vulnerable under consumer 720p webcams with sensor noise or poor illumination.
          </p>
</div>
<div class="bg-surface-container p-space-lg rounded-xl flex flex-col gap-space-xs shadow-sm">
<span class="font-technical-data text-technical-data text-tertiary font-bold">LIMITATION II</span>
<h3 class="font-headline-md text-headline-md text-on-surface">Temporal Dialect Asynchrony</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
            Substantial regional divergence in temporal markers (e.g., Delhi ISL expressing past-tense via backwards over-the-shoulder motion vs. Chennai ISL using directional downward-swept palm motions) requires manual dialect mode selection.
          </p>
</div>
<div class="bg-surface-container p-space-lg rounded-xl flex flex-col gap-space-xs shadow-sm">
<span class="font-technical-data text-tertiary font-bold">DISQUALIFICATION III</span>
<h3 class="font-headline-md text-headline-md text-on-surface">High-Stakes Ineligibility</h3>
<p class="font-body-sm text-body-sm text-on-surface-variant leading-relaxed">
            This experimental prototype is <strong>strictly not certified</strong> for emergency hospital triage, criminal justice depositions, or high-liability legal contracts. It is engineered to assist and augment, never displace, certified human Deaf interpreters.
          </p>
</div>
</div>
</section>
<!-- Section 8: Artifact Downloads & BibTeX Citation Block -->
<section class="flex flex-col gap-space-md">
<div class="flex flex-col gap-1">
<span class="font-technical-data text-technical-data text-primary font-bold">08 // REPRODUCIBILITY &amp; ARTIFACTS</span>
<h2 class="font-headline-lg text-headline-lg text-on-surface">Monograph Citation &amp; Open Research Assets</h2>
</div>
<!-- Download Cards Grid -->
<div class="grid grid-cols-1 md:grid-cols-4 gap-space-md">
<a class="bg-surface-container hover:bg-surface-container-high transition-colors p-space-md rounded-xl flex flex-col justify-between gap-space-sm shadow-sm group" href="#">
<div class="flex items-center justify-between">
<span class="material-symbols-outlined text-primary text-[24px]">description</span>
<span class="font-technical-data text-[10px] text-on-surface-variant">PDF · 28 PGS</span>
</div>
<div class="flex flex-col">
<span class="font-label-md text-label-md text-on-surface group-hover:text-primary transition-colors">Monograph PDF</span>
<span class="font-body-sm text-body-sm text-on-surface-variant">Full methodology, mathematical proofs, and appendices.</span>
</div>
<span class="font-technical-data text-technical-data text-primary font-semibold flex items-center gap-1">Download (4.2 MB) <span class="material-symbols-outlined text-[14px]">download</span></span>
</a>
<a class="bg-surface-container hover:bg-surface-container-high transition-colors p-space-md rounded-xl flex flex-col justify-between gap-space-sm shadow-sm group" href="#">
<div class="flex items-center justify-between">
<span class="material-symbols-outlined text-secondary text-[24px]">dataset</span>
<span class="font-technical-data text-[10px] text-on-surface-variant">CSV / JSONL</span>
</div>
<div class="flex flex-col">
<span class="font-label-md text-label-md text-on-surface group-hover:text-secondary transition-colors">ISL-CSL-12k Benchmarks</span>
<span class="font-body-sm text-body-sm text-on-surface-variant">Normalized landmark vectors and NIST evaluation scores.</span>
</div>
<span class="font-technical-data text-technical-data text-secondary font-semibold flex items-center gap-1">Download (128 MB) <span class="material-symbols-outlined text-[14px]">download</span></span>
</a>
<a class="bg-surface-container hover:bg-surface-container-high transition-colors p-space-md rounded-xl flex flex-col justify-between gap-space-sm shadow-sm group" href="#">
<div class="flex items-center justify-between">
<span class="material-symbols-outlined text-primary text-[24px]">memory</span>
<span class="font-technical-data text-[10px] text-on-surface-variant">ONNX FP16</span>
</div>
<div class="flex flex-col">
<span class="font-label-md text-label-md text-on-surface group-hover:text-primary transition-colors">Quantized Model Weights</span>
<span class="font-body-sm text-body-sm text-on-surface-variant">Pre-trained edge-compatible ONNX graphs with WebGPU kernels.</span>
</div>
<span class="font-technical-data text-technical-data text-primary font-semibold flex items-center gap-1">Download (84 MB) <span class="material-symbols-outlined text-[14px]">download</span></span>
</a>
<a class="bg-surface-container hover:bg-surface-container-high transition-colors p-space-md rounded-xl flex flex-col justify-between gap-space-sm shadow-sm group" href="#">
<div class="flex items-center justify-between">
<span class="material-symbols-outlined text-tertiary text-[24px]">balance</span>
<span class="font-technical-data text-[10px] text-on-surface-variant">CC-BY 4.0</span>
</div>
<div class="flex flex-col">
<span class="font-label-md text-label-md text-on-surface group-hover:text-tertiary transition-colors">Open Dataset Cards</span>
<span class="font-body-sm text-body-sm text-on-surface-variant">Consent protocols and linguistic transcription standards.</span>
</div>
<span class="font-technical-data text-technical-data text-tertiary font-semibold flex items-center gap-1">Download (1.1 MB) <span class="material-symbols-outlined text-[14px]">download</span></span>
</a>
</div>
<!-- BibTeX Card -->
<div class="bg-surface-container p-space-lg rounded-xl flex flex-col gap-space-sm shadow-md">
<div class="flex items-center justify-between">
<span class="font-technical-data text-technical-data text-on-surface font-semibold flex items-center gap-space-xs">
<span class="material-symbols-outlined text-primary text-[18px]">format_quote</span>
            BibTeX Citation
          </span>
<button class="px-space-sm py-1 rounded bg-surface-container-high hover:bg-primary hover:text-on-primary font-technical-data text-technical-data text-on-surface-variant transition-colors flex items-center gap-1" id="copy-bibtex-btn">
<span class="material-symbols-outlined text-[14px]">content_copy</span>
<span id="copy-label">Copy Citation</span>
</button>
</div>
<pre class="bg-surface-container-lowest p-space-md rounded-lg font-technical-data text-technical-data text-on-surface-variant overflow-x-auto leading-relaxed select-all">@article{sharma2024signsage,
  title={Bidirectional Continuous Indian Sign Language Translation via Edge-Native Kinematic Modeling and Topic-Comment Syntactic Alignment},
  author={Sharma, A. and Venkatesh, R. and Narang, P.},
  journal={M.Tech Research Monograph Series},
  volume={2024},
  number={04},
  pages={1--28},
  year={2024},
  doi={10.1145/signsage.2024.04.2},
  archivePrefix={arXiv},
  eprint={2409.ISL01},
  primaryClass={cs.CV}
}</pre>
</div>
</section>
</div>
<script>
    const copyBtn = document.getElementById('copy-bibtex-btn');
    const copyLabel = document.getElementById('copy-label');
    if (copyBtn && copyLabel) {
      copyBtn.addEventListener('click', () => {
        const bibtexText = \`@article{sharma2024signsage,\n  title={Bidirectional Continuous Indian Sign Language Translation via Edge-Native Kinematic Modeling and Topic-Comment Syntactic Alignment},\n  author={Sharma, A. and Venkatesh, R. and Narang, P.},\n  journal={M.Tech Research Monograph Series},\n  volume={2024},\n  number={04},\n  pages={1--28},\n  year={2024},\n  doi={10.1145/signsage.2024.04.2},\n  archivePrefix={arXiv},\n  eprint={2409.ISL01},\n  primaryClass={cs.CV}\n}\`;
        navigator.clipboard.writeText(bibtexText).then(() => {
          copyLabel.innerText = 'Copied!';
          setTimeout(() => {
            copyLabel.innerText = 'Copy Citation';
          }, 2000);
        });
      });
    }
  </script>
</div></main><footer class="w-full bg-surface-container-lowest border-t border-outline-variant/30 py-space-md"><div class="w-full px-margin-desktop flex flex-col md:flex-row items-center justify-between gap-space-sm text-center md:text-left"><p class="font-technical-data text-technical-data text-on-surface-variant">SignSage is an M.Tech experimental communication prototype. Video data is processed strictly in local client memory and never transmitted or retained.</p><div class="flex items-center gap-space-md font-technical-data text-technical-data text-on-surface-variant"><span>Precision: 30 FPS</span><span>ISL Core v2.1</span></div></div></footer>`;
