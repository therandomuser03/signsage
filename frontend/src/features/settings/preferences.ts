// SignSage — user preferences.
//
// Only preferences the app actually honors. Both are read by the
// translation pages at action time; nothing here is decorative.

export interface TranslationPrefs {
  /** Sent as save_history on sentence boundaries and gloss requests. */
  saveHistory: boolean;
  /** Mirror the camera preview (selfie view). */
  mirrorPreview: boolean;
}

const KEY = "signsage.prefs";

const DEFAULTS: TranslationPrefs = {
  saveHistory: true,
  mirrorPreview: true,
};

export function loadPrefs(): TranslationPrefs {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return { ...DEFAULTS };
    const parsed = JSON.parse(raw) as Partial<TranslationPrefs>;
    return {
      saveHistory: parsed.saveHistory ?? DEFAULTS.saveHistory,
      mirrorPreview: parsed.mirrorPreview ?? DEFAULTS.mirrorPreview,
    };
  } catch {
    return { ...DEFAULTS };
  }
}

export function savePrefs(prefs: TranslationPrefs): void {
  localStorage.setItem(KEY, JSON.stringify(prefs));
}
