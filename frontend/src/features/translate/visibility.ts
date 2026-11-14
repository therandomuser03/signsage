import { config } from "../../app/config";
import type { CapabilitiesResponse } from "../../types/api";

/** Word recognition card (trained model) renders only when the backend
 * enables it. Any other value (false/null/missing capabilities) hides it. */
export function showWordRecognitionCard(
  capabilities: CapabilitiesResponse | null,
): boolean {
  return capabilities?.word_recognition_enabled === true;
}

/** Legacy six-word video-upload pilot: hidden by default, visible only when
 * VITE_LEGACY_UPLOAD is exactly "true". Default = hidden. */
export function showLegacyUpload(
  legacyUpload: string = config.legacyUpload,
): boolean {
  return legacyUpload === "true";
}
