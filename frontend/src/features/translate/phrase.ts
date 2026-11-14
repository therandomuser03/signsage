import type { CapabilitiesResponse } from "../../types/api";
import { config } from "../../app/config";

/** "Sign a phrase" (POST /api/v1/translation/sentence) shows only when
 * word mode is enabled by capabilities AND VITE_PHRASE_MODE === "true".
 * Default = hidden (unset/false). */
export function showSignPhraseButton(
  capabilities: CapabilitiesResponse | null,
  phraseMode: string = config.phraseMode,
): boolean {
  return phraseMode === "true" && capabilities?.word_recognition_enabled === true;
}
