// SignSage — dev-only avatar validation page (/dev/avatar).
//
// Loads the REAL signsage-human.glb and exposes bone/morph inspection plus
// interactive finger, wrist and facial-blend-shape controls. Not linked from
// navigation; registered as a route only when import.meta.env.DEV is true.
// Does not touch the production interpreter avatar.

import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Hand,
  Loader2,
  RotateCcw,
  ScanFace,
  Bone,
  Info,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { HumanAvatarScene, type SceneStats } from "./humanAvatarScene";
import {
  FINGERS,
  categorizeBones,
  categorizeMorphs,
  inspectHands,
  type HandSide,
} from "./rigMap";

const MODEL_URL = "/models/signsage-human.glb";
const FINGER_IDS = FINGERS.map((f) => f.finger);

type Phase = "loading" | "ready" | "error";

function curlKey(side: HandSide, finger: string) {
  return `${side}:${finger}`;
}

function zeroCurls(): Record<string, number> {
  const out: Record<string, number> = {};
  for (const side of ["L", "R"] as HandSide[]) for (const f of FINGER_IDS) out[curlKey(side, f)] = 0;
  return out;
}

export default function AvatarTestPage() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const sceneRef = useRef<HumanAvatarScene | null>(null);
  const lastFramesRef = useRef(0);

  const [phase, setPhase] = useState<Phase>("loading");
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<SceneStats | null>(null);
  const [fps, setFps] = useState<number | null>(null);
  const [boneNames, setBoneNames] = useState<string[]>([]);
  const [morphNames, setMorphNames] = useState<string[]>([]);
  const [curls, setCurls] = useState<Record<string, number>>(zeroCurls);
  const [wrist, setWrist] = useState<Record<HandSide, number>>({ L: 0, R: 0 });
  const [morphValues, setMorphValues] = useState<Record<string, number>>({});

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const scene = new HumanAvatarScene(canvas);
    sceneRef.current = scene;

    scene
      .load(MODEL_URL)
      .then(() => {
        setBoneNames(scene.getBoneNames());
        setMorphNames(scene.getMorphTargetNames());
        setStats(scene.getStats());
        setPhase("ready");
      })
      .catch((err: unknown) => {
        console.error("[avatar-test] failed to load GLB", err);
        setError(err instanceof Error ? err.message : String(err));
        setPhase("error");
      });

    const frames = scene.getFramesRendered.bind(scene);
    const id = window.setInterval(() => {
      const rendered = frames();
      setFps(rendered - lastFramesRef.current);
      lastFramesRef.current = rendered;
      setStats(scene.getStats());
    }, 1000);

    return () => {
      window.clearInterval(id);
      scene.dispose();
      sceneRef.current = null;
    };
  }, []);

  // Connectivity is checked against the REAL loaded hierarchy, not name guesses.
  const [hands, setHands] = useState<ReturnType<typeof inspectHands> | null>(null);
  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || phase !== "ready") return;
    setHands(inspectHands(scene.getBones()));
  }, [phase]);

  const boneGroups = useMemo(() => categorizeBones(boneNames), [boneNames]);
  const morphGroups = useMemo(() => categorizeMorphs(morphNames), [morphNames]);

  const setCurl = (side: HandSide, finger: (typeof FINGER_IDS)[number], value: number) => {
    setCurls((prev) => ({ ...prev, [curlKey(side, finger)]: value }));
    sceneRef.current?.setFingerCurl(side, finger, value / 100);
  };

  const setAllCurls = (value: number) => {
    setCurls(() => {
      const next = zeroCurls();
      for (const k of Object.keys(next)) next[k] = value;
      return next;
    });
    sceneRef.current?.setAllFingerCurls("both", value / 100);
  };

  const setWristTwist = (side: HandSide, degrees: number) => {
    setWrist((prev) => ({ ...prev, [side]: degrees }));
    sceneRef.current?.setWristTwist(side, degrees);
  };

  const [armRaise, setArmRaise] = useState(0);
  const setArms = (value: number) => {
    setArmRaise(value);
    sceneRef.current?.setArmPose(value / 100);
  };

  const resetPose = () => {
    setCurls(zeroCurls());
    setWrist({ L: 0, R: 0 });
    setArmRaise(0);
    sceneRef.current?.resetPose();
  };

  const setMorph = (name: string, value: number) => {
    setMorphValues((prev) => ({ ...prev, [name]: value }));
    sceneRef.current?.setMorphTarget(name, value);
  };

  const resetFace = () => {
    setMorphValues({});
    sceneRef.current?.resetFace();
  };

  const blinkTest = () => {
    const scene = sceneRef.current;
    if (!scene) return;
    const close = { eyeBlinkLeft: 1, eyeBlinkRight: 1 };
    const open = { eyeBlinkLeft: 0, eyeBlinkRight: 0 };
    for (const [name, v] of Object.entries(close)) setMorph(name, v);
    window.setTimeout(() => {
      for (const [name, v] of Object.entries(open)) setMorph(name, v);
    }, 220);
    window.setTimeout(() => {
      for (const [name, v] of Object.entries(close)) setMorph(name, v);
    }, 440);
    window.setTimeout(() => {
      for (const [name, v] of Object.entries(open)) setMorph(name, v);
    }, 660);
  };

  const morphValue = (name: string) => morphValues[name] ?? 0;

  return (
    <div className="min-h-screen bg-[#0b0f14] text-slate-100">
      {/* Dev-only banner */}
      <div className="flex items-center justify-between border-b border-amber-400/20 bg-amber-400/10 px-4 py-2 text-xs text-amber-200">
        <span className="flex items-center gap-2">
          <Info aria-hidden="true" className="h-4 w-4" />
          Development-only avatar validation page — not linked in navigation, not part of the production interpreter.
        </span>
        <Badge variant="outline" className="border-amber-400/40 text-amber-200">
          DEV
        </Badge>
      </div>

      <main className="mx-auto grid w-full max-w-[1600px] gap-4 p-4 lg:grid-cols-[minmax(0,1fr)_420px]">
        {/* Viewport */}
        <section className="relative overflow-hidden rounded-xl border border-white/10 bg-[#11161d]">
          <div className="relative h-[62vh] min-h-[420px] w-full lg:h-[calc(100vh-5.5rem)]">
            <canvas ref={canvasRef} className="block h-full w-full touch-none outline-none" aria-label="3D human avatar test viewport" />

            {phase === "loading" && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-[#11161d]/80">
                <Loader2 aria-hidden="true" className="h-8 w-8 animate-spin text-emerald-400" />
                <p className="text-sm text-slate-300">Loading signsage-human.glb (6.4 MB)…</p>
              </div>
            )}
            {phase === "error" && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-[#11161d]/90 p-6 text-center">
                <AlertTriangle aria-hidden="true" className="h-8 w-8 text-red-400" />
                <p className="text-sm font-semibold text-red-300">Failed to load the avatar</p>
                <p className="max-w-md break-words font-mono text-xs text-red-200/80">{error}</p>
              </div>
            )}

            {phase === "ready" && (
              <>
                <div className="absolute left-3 top-3 flex gap-2">
                  <Button size="sm" variant="secondary" onClick={() => sceneRef.current?.resetCamera()}>
                    <RotateCcw aria-hidden="true" />
                    Reset camera
                  </Button>
                </div>
                <div className="pointer-events-none absolute bottom-3 left-3 flex flex-wrap gap-2 text-[11px] text-slate-300">
                  {stats && (
                    <>
                      <StatChip label="Bones" value={String(stats.boneCount)} />
                      <StatChip label="Skin joints" value={String(stats.skinnedMeshJoints)} />
                      <StatChip label="Triangles" value={stats.triangleCount.toLocaleString()} />
                      <StatChip label="Draw calls" value={String(stats.drawCalls)} />
                      <StatChip label="Pixel ratio" value={stats.pixelRatio.toFixed(2)} />
                      <StatChip label="Renders/s" value={fps === null ? "—" : String(fps)} />
                    </>
                  )}
                </div>
                <p className="pointer-events-none absolute bottom-3 right-3 text-[11px] text-slate-500">
                  Drag to orbit · scroll to zoom · right-drag to pan
                </p>
              </>
            )}
          </div>
        </section>

        {/* Control / inspection panel */}
        <aside className="flex max-h-none flex-col gap-4 overflow-y-auto lg:max-h-[calc(100vh-5.5rem)]">
          {/* Skeleton */}
          <Card className="border-white/10 bg-white/[0.03] text-slate-100">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-base">
                <Bone aria-hidden="true" className="h-5 w-5 text-emerald-400" />
                Skeleton inspection
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-xs">
              {phase !== "ready" ? (
                <p className="text-slate-400">Waiting for model…</p>
              ) : (
                <>
                  <p className="text-[10px] leading-4 text-slate-500">
                    Bone names below are shown as stored in the live scene — three.js strips reserved characters
                    (<code>. : [ ]</code>) from node names at import, so GLB <code>wrist.L</code> appears as{" "}
                    <code>wristL</code>.
                  </p>
                  <div className="grid grid-cols-4 gap-2 text-center">
                    <Metric label="Scene bones" value={boneNames.length} />
                    <Metric label="Skin joints" value={stats?.skinnedMeshJoints ?? 0} />
                    <Metric label="Meshes" value={stats?.meshCount ?? 0} />
                    <Metric label="Morph targets" value={morphNames.length} />
                  </div>

                  {hands?.map((hand) => (
                    <div key={hand.side} className="rounded-lg border border-white/10 p-2">
                      <p className="mb-1 flex items-center justify-between font-semibold text-slate-200">
                        {hand.side === "L" ? "Left hand" : "Right hand"}
                        <span className={hand.wrist ? "text-emerald-400" : "text-red-400"}>
                          wrist: {hand.wrist ?? "MISSING"}
                        </span>
                      </p>
                      <ul className="space-y-1">
                        {hand.fingers.map((f) => (
                          <li key={f.finger} className="flex items-start gap-2">
                            {f.connected && f.missing.length === 0 ? (
                              <CheckCircle2 aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" />
                            ) : (
                              <AlertTriangle aria-hidden="true" className="mt-0.5 h-4 w-4 shrink-0 text-amber-400" />
                            )}
                            <span className="min-w-0">
                              <span className="font-medium text-slate-200">{f.label}</span>
                              <span className="block break-words font-mono text-[10px] leading-4 text-slate-400">
                                {f.present.join(" → ") || "—"}
                                {f.missing.length > 0 && (
                                  <span className="text-red-400"> · missing: {f.missing.join(", ")}</span>
                                )}
                                {!f.connected && <span className="text-amber-400"> · chain not parented as expected</span>}
                              </span>
                            </span>
                          </li>
                        ))}
                      </ul>
                    </div>
                  ))}

                  <details className="rounded-lg border border-white/10 p-2">
                    <summary className="cursor-pointer font-semibold text-slate-200">
                      Categorised bones ({boneNames.length})
                    </summary>
                    <div className="mt-2 space-y-1.5">
                      {boneGroups.map((g) => (
                        <div key={g.label}>
                          <p className="text-[11px] font-semibold text-slate-300">
                            {g.label} <span className="text-slate-500">({g.bones.length})</span>
                          </p>
                          <p className="break-words font-mono text-[10px] leading-4 text-slate-500">
                            {g.bones.join(", ") || "—"}
                          </p>
                        </div>
                      ))}
                    </div>
                  </details>
                </>
              )}
            </CardContent>
          </Card>

          {/* Finger articulation */}
          <Card className="border-white/10 bg-white/[0.03] text-slate-100">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center gap-2 text-base">
                <Hand aria-hidden="true" className="h-5 w-5 text-emerald-400" />
                Finger articulation
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-xs">
              <p className="text-slate-400">
                Drives the actual imported bones (<code>fingerN-*</code> / <code>metacarpalN-*</code>) — no prerecorded
                animation, no visual effects.
              </p>
              <div className="flex flex-wrap gap-2">
                <Button size="sm" onClick={() => setAllCurls(100)} disabled={phase !== "ready"}>
                  Close both hands
                </Button>
                <Button size="sm" variant="outline" onClick={() => setAllCurls(0)} disabled={phase !== "ready"}>
                  Open both hands
                </Button>
                <Button size="sm" variant="ghost" onClick={resetPose} disabled={phase !== "ready"}>
                  Reset pose
                </Button>
              </div>
              <label className="grid grid-cols-[4.5rem_1fr_2.5rem] items-center gap-2 border-b border-white/10 pb-2">
                <span className="text-slate-300">Arms raised</span>
                <input
                  type="range"
                  min={0}
                  max={100}
                  step={1}
                  value={armRaise}
                  aria-label="Raise both arms"
                  onChange={(e) => setArms(Number(e.target.value))}
                  className="h-1.5 w-full cursor-pointer accent-emerald-500"
                />
                <span className="text-right tabular-nums text-slate-400">{armRaise}%</span>
              </label>
              {(["L", "R"] as HandSide[]).map((side) => (
                <fieldset key={side} className="rounded-lg border border-white/10 p-2" disabled={phase !== "ready"}>
                  <legend className="px-1 font-semibold text-slate-200">{side === "L" ? "Left hand" : "Right hand"}</legend>
                  <div className="space-y-1.5">
                    {FINGERS.map((f) => {
                      const value = curls[curlKey(side, f.finger)] ?? 0;
                      return (
                        <label key={f.finger} className="grid grid-cols-[3.5rem_1fr_2.5rem] items-center gap-2">
                          <span className="text-slate-300">{f.label}</span>
                          <input
                            type="range"
                            min={0}
                            max={100}
                            step={1}
                            value={value}
                            aria-label={`${side === "L" ? "Left" : "Right"} ${f.label} flexion`}
                            onChange={(e) => setCurl(side, f.finger, Number(e.target.value))}
                            className="h-1.5 w-full cursor-pointer accent-emerald-500"
                          />
                          <span className="text-right tabular-nums text-slate-400">{value}%</span>
                        </label>
                      );
                    })}
                    <label className="grid grid-cols-[3.5rem_1fr_3rem] items-center gap-2 border-t border-white/10 pt-1.5">
                      <span className="text-slate-300">Wrist</span>
                      <input
                        type="range"
                        min={-75}
                        max={75}
                        step={1}
                        value={wrist[side]}
                        aria-label={`${side === "L" ? "Left" : "Right"} wrist twist`}
                        onChange={(e) => setWristTwist(side, Number(e.target.value))}
                        className="h-1.5 w-full cursor-pointer accent-emerald-500"
                      />
                      <span className="text-right tabular-nums text-slate-400">{wrist[side]}°</span>
                    </label>
                  </div>
                </fieldset>
              ))}
            </CardContent>
          </Card>

          {/* Facial morph targets */}
          <Card className="border-white/10 bg-white/[0.03] text-slate-100">
            <CardHeader className="pb-2">
              <CardTitle className="flex items-center justify-between text-base">
                <span className="flex items-center gap-2">
                  <ScanFace aria-hidden="true" className="h-5 w-5 text-emerald-400" />
                  Facial blend shapes
                </span>
                <span className="text-xs font-normal text-slate-400">{morphNames.length} targets</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3 text-xs">
              <div className="flex flex-wrap gap-2">
                <Button size="sm" variant="outline" onClick={blinkTest} disabled={phase !== "ready"}>
                  Blink test
                </Button>
                <Button size="sm" variant="ghost" onClick={resetFace} disabled={phase !== "ready"}>
                  Reset face
                </Button>
              </div>
              {phase === "ready" && morphGroups.map((group) => (
                <details key={group.label} className="rounded-lg border border-white/10 p-2" open>
                  <summary className="cursor-pointer font-semibold text-slate-200">
                    {group.label} <span className="text-slate-500">({group.targets.length})</span>
                  </summary>
                  <div className="mt-2 space-y-1.5">
                    {group.targets.map((name) => (
                      <label key={name} className="grid grid-cols-[7.5rem_1fr_2.5rem] items-center gap-2">
                        <span className="truncate font-mono text-[10px] text-slate-300" title={name}>
                          {name}
                        </span>
                        <input
                          type="range"
                          min={0}
                          max={100}
                          step={1}
                          value={Math.round(morphValue(name) * 100)}
                          aria-label={`Morph target ${name}`}
                          onChange={(e) => setMorph(name, Number(e.target.value) / 100)}
                          className="h-1.5 w-full cursor-pointer accent-emerald-500"
                        />
                        <span className="text-right tabular-nums text-slate-400">
                          {Math.round(morphValue(name) * 100)}%
                        </span>
                      </label>
                    ))}
                  </div>
                </details>
              ))}
            </CardContent>
          </Card>

          <Separator className="bg-white/10" />
          <p className="px-1 pb-4 text-[11px] leading-4 text-slate-500">
            Model: <code>/models/signsage-human.glb</code> · Ara 3D Gen · MakeHuman rig · 164 joints · no animations in
            export. Orbit with left-drag, pan with right-drag, zoom with scroll.
          </p>
        </aside>
      </main>
    </div>
  );
}

function StatChip({ label, value }: { label: string; value: string }) {
  return (
    <span className="rounded-md border border-white/10 bg-black/50 px-2 py-1 backdrop-blur-sm">
      <span className="text-slate-500">{label}: </span>
      <span className="tabular-nums text-slate-200">{value}</span>
    </span>
  );
}

function Metric({ label, value }: { label: string; value: number }) {
  return (
    <div className="rounded-lg border border-white/10 p-2">
      <p className="text-lg font-semibold tabular-nums text-emerald-400">{value}</p>
      <p className="text-[10px] uppercase tracking-wide text-slate-500">{label}</p>
    </div>
  );
}
