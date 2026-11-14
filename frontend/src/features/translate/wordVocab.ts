import type { CapabilitiesResponse } from "../../types/api";

export type InterpreterTab = "word" | "rule";

/** The rest (no-sign) class is reported inside the same vocab list as the
 * words but is not a signable word. Splits it out, case-insensitively. */
export function splitWordVocab(vocab: string[] | null | undefined): {
  words: string[];
  restClass: boolean;
} {
  const list = vocab ?? [];
  const words = list.filter((v) => v.trim().toLowerCase() !== "rest");
  const restClass = words.length !== list.length;
  return { words, restClass };
}

/** "8 words plus a rest (no-sign) class" derived from the capabilities list. */
export function wordVocabSummary(vocab: string[] | null | undefined): string {
  const { words, restClass } = splitWordVocab(vocab);
  const noun = words.length === 1 ? "word" : "words";
  return restClass
    ? `${words.length} ${noun} plus a rest (no-sign) class`
    : `${words.length} ${noun}`;
}

/** Default tab for the interpreter page: Word recognition when the backend
 * enables it, Rule-based preview otherwise (incl. capabilities unavailable). */
export function defaultInterpreterTab(
  capabilities: CapabilitiesResponse | null,
): InterpreterTab {
  return capabilities?.word_recognition_enabled === true ? "word" : "rule";
}

/** Detail line for the disabled-Word-card. The only reason the capabilities
 * payload carries is sign_to_text_unavailable_reason; use it when present. */
export function wordDisabledDetail(
  capabilities: CapabilitiesResponse | null,
): string | null {
  if (capabilities === null) {
    return "Capabilities could not be loaded — the backend may be offline or still waking up.";
  }
  return capabilities.sign_to_text_unavailable_reason ?? null;
}
