// SignSage — Text → ISL avatar types
// One VRM avatar + a library of small GLB animation clips, addressed by gloss ID.

/** A single entry in the sign library: one gloss maps to one animation clip. */
export interface SignManifestEntry {
  /** Canonical gloss ID, e.g. "HELLO", "THANK_YOU". Always upper snake case. */
  id: string;
  /** Human-readable label for debugging/UI captions. */
  label: string;
  /** Path to the GLB file containing this sign's AnimationClip (no mesh needed). */
  animationUrl: string;
  /** Optional: seconds. If omitted, read from the loaded THREE.AnimationClip. */
  durationSeconds?: number;
}

/** The full library, keyed by gloss ID for O(1) lookup. */
export type SignManifest = Record<string, SignManifestEntry>;

/** Result of turning raw English text into a sequence the avatar can perform. */
export interface GlossSequence {
  /** Original input text, kept for logging/debugging. */
  sourceText: string;
  /** Ordered gloss IDs to play. Unknown words are dropped, not guessed. */
  glossIds: string[];
  /** Words from sourceText that had no matching sign in the manifest. */
  unmatchedWords: string[];
}

/** Playback state exposed to the UI by the useSignAnimation hook. */
export interface SignPlaybackState {
  isPlaying: boolean;
  currentIndex: number; // -1 when idle
  currentGlossId: string | null;
  queue: string[];
}
