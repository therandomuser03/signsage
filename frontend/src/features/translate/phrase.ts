import type { CapabilitiesResponse } from "../../types/api";
import { config } from "../../app/config";

/** "Sign a phrase" (POST /api/v1/translation/sentence) shows by default when
 * word mode is enabled. It is hidden only when VITE_PHRASE_MODE is exactly "false". */
export function showSignPhraseButton(
  capabilities: CapabilitiesResponse | null,
  phraseMode: string = config.phraseMode,
): boolean {
  return phraseMode !== "false" && capabilities?.word_recognition_enabled === true;
}
