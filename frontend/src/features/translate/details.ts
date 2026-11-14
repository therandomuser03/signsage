export interface CaptureDetails {
  kind: "word" | "phrase";
  framesCaptured: number;
  effectiveFps: number;
  leftHandFraction: number;
  rightHandFraction: number;
  roundTripMs: number;
  predictedLabel?: string | null;
  confidence?: number;
  threshold?: number;
  accepted?: boolean;
  top3?: Array<{ label: string; probability: number }>;
  modelVersion?: string;
  status?: string;
  glossSequence?: Array<{ gloss: string; confidence: number }>;
  sentence?: string;
  nWindows?: number;
}

export function formatDetailsCompact(d: CaptureDetails): Record<string, unknown> {
  const common = {
    kind: d.kind,
    frames_captured: d.framesCaptured,
    effective_fps: Number(d.effectiveFps.toFixed(1)),
    left_hand_fraction: Number(d.leftHandFraction.toFixed(3)),
    right_hand_fraction: Number(d.rightHandFraction.toFixed(3)),
    request_round_trip_ms: Number(d.roundTripMs.toFixed(1)),
    model_version: d.modelVersion ?? null,
    status: d.status ?? null,
  };
  if (d.kind === "phrase") {
    return {
      ...common,
      gloss_sequence: d.glossSequence ?? [],
      sentence: d.sentence ?? null,
      n_windows: d.nWindows ?? null,
      accepted: d.accepted ?? null,
      threshold: d.threshold ?? null,
    };
  }
  return {
    ...common,
    predicted_label: d.predictedLabel ?? null,
    confidence: d.confidence !== undefined ? Number(d.confidence.toFixed(3)) : null,
    threshold: d.threshold !== undefined ? Number(d.threshold.toFixed(2)) : null,
    accepted: d.accepted ?? false,
    top3: d.top3 ?? [],
  };
}

export function detailsToJson(d: CaptureDetails): string {
  return JSON.stringify(formatDetailsCompact(d));
}
