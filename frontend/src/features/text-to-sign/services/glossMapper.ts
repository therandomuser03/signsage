// SignSage — English → gloss mapping
//
// MVP: naive per-word dictionary lookup against the sign manifest. This is
// intentionally dumb (no grammar reordering, no multi-word signs) so the
// avatar pipeline can be built and tested before any NLP work happens.
//
// Swap `localGlossMapper` for `remoteGlossMapper` once the backend's
// Groq/Ollama-assisted gloss endpoint (see backend/app/services/gloss_service.py)
// is ready — the GlossSequence shape doesn't change.

import type { ApiClient } from "../../../services/api/client";
import type { GlossSequence, SignManifest } from "../types";

function normalizeWord(word: string): string {
  return word.trim().toUpperCase().replace(/[^A-Z']/g, "");
}

/**
 * Client-side, zero-latency mapping. Greedy longest-match so multi-word signs
 * (e.g. THANK_YOU) match "thank you" instead of degrading to just YOU.
 * Unmatched words are dropped, never guessed.
 */
export function localGlossMapper(text: string, manifest: SignManifest): GlossSequence {
  const words = text.split(/\s+/).map(normalizeWord).filter(Boolean);
  const originals = text.split(/\s+/).filter(Boolean);
  const maxSpan = Math.max(1, ...Object.keys(manifest).map((id) => id.split("_").length));

  const glossIds: string[] = [];
  const unmatchedWords: string[] = [];

  let i = 0;
  while (i < words.length) {
    let matched = false;
    for (let span = Math.min(maxSpan, words.length - i); span >= 1; span--) {
      const candidate = words.slice(i, i + span).join("_");
      if (manifest[candidate]) {
        glossIds.push(candidate);
        i += span;
        matched = true;
        break;
      }
    }
    if (!matched) {
      unmatchedWords.push(originals[i] ?? words[i]);
      i += 1;
    }
  }

  return { sourceText: text, glossIds, unmatchedWords };
}

/** Backend-assisted mapping (vocabulary lives server-side too; same GlossSequence shape). */
export async function remoteGlossMapper(
  text: string,
  manifest: SignManifest,
  api: Pick<ApiClient, "avatarGloss">,
  saveHistory = true,
): Promise<GlossSequence> {
  const res = await api.avatarGloss({ text, save_history: saveHistory });
  // Defence in depth: never try to play a clip the manifest doesn't have.
  const glossIds = res.gloss.filter((id) => manifest[id]);
  const missingClips = res.gloss.filter((id) => !manifest[id]);
  return { sourceText: text, glossIds, unmatchedWords: [...res.unmatched_words, ...missingClips] };
}
