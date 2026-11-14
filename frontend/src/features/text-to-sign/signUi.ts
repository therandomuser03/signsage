/** Pure helpers for the Sign Generator page (Part D). Kept DOM-free for tests. */

/** Example prompts shown above the input; clicking one fills the input and
 * runs Generate with exactly this text. */
export const EXAMPLE_CHIPS = [
  "Hello",
  "Thank you",
  "Where is water?",
  "Help me",
  "Yes",
  "No",
] as const;

export type GlossChipState = "past" | "current" | "queued";

/** Visual state of a gloss chip during playback. `currentIndex` comes from
 * the playback `onStep` callback; -1 means not playing. */
export function glossChipState(
  index: number,
  currentIndex: number,
  isPlaying: boolean,
): GlossChipState {
  if (index < currentIndex) return "past";
  if (index === currentIndex && isPlaying) return "current";
  return "queued";
}

/** Display text that maps a vocabulary gloss id back through the text->gloss
 * mapper path (underscores are not word characters, so THANK_YOU would
 * tokenize wrong as-is). */
export function glossIdToText(glossId: string): string {
  return glossId.replaceAll("_", " ").toLowerCase();
}

/** Sorted copy of the available-signs list (vocabulary endpoint order is
 * backend-defined; render alphabetically). */
export function sortVocabulary(ids: string[]): string[] {
  return [...ids].sort((a, b) => a.localeCompare(b));
}

/** Muted-chip label for a word with no sign in the library. */
export function missingSignLabel(word: string): string {
  return `${word} — not in sign library`;
}
