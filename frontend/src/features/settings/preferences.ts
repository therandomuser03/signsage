// SignSage — user preferences.
//
// Only preferences the app actually honors. Read by the translation pages at
// action time; nothing here is decorative.

export type TtsProvider = "auto" | "groq" | "browser";

export interface TranslationPrefs {
  /** Sent as save_history on sentence boundaries and gloss requests. */
  saveHistory: boolean;
  /** Mirror the camera preview (selfie view). */
  mirrorPreview: boolean;
  /** Speak completed interpretations automatically. */
  autoSpeak: boolean;
  /** Audio provider: "auto" = Groq Orpheus when configured, browser voices otherwise;
   *  "groq" = backend Groq Orpheus (falls back to browser voices on failure);
   *  "browser" = Web Speech API only, no backend call. */
  ttsProvider: TtsProvider;
  /** Orpheus voice persona (backend). */
  groqTtsVoice: string;
  /** Browser voice URI (only used by the browser path). */
  browserVoiceURI: string | null;
  /** Browser speech rate (Web Speech API only; Orpheus has no rate control). */
  speechRate: number;
}

const KEY = "signsage.prefs";

const DEFAULTS: TranslationPrefs = {
  saveHistory: true,
  mirrorPreview: false,
  autoSpeak: false,
  ttsProvider: "auto",
  groqTtsVoice: "hannah",
  browserVoiceURI: null,
  speechRate: 0.9,
};

export function coercePrefs(parsed: unknown): TranslationPrefs {
  if (typeof parsed !== "object" || parsed === null) return { ...DEFAULTS };
  const p = parsed as Record<string, unknown>;
  const rate = typeof p.speechRate === "number" && Number.isFinite(p.speechRate)
    ? Math.min(2, Math.max(0.5, p.speechRate))
    : DEFAULTS.speechRate;
  const provider: TtsProvider =
    p.ttsProvider === "groq" || p.ttsProvider === "browser" || p.ttsProvider === "auto"
      ? p.ttsProvider
      : DEFAULTS.ttsProvider;
  return {
    saveHistory: typeof p.saveHistory === "boolean" ? p.saveHistory : DEFAULTS.saveHistory,
    mirrorPreview: typeof p.mirrorPreview === "boolean" ? p.mirrorPreview : DEFAULTS.mirrorPreview,
    autoSpeak: typeof p.autoSpeak === "boolean" ? p.autoSpeak : DEFAULTS.autoSpeak,
    ttsProvider: provider,
    groqTtsVoice: typeof p.groqTtsVoice === "string" && p.groqTtsVoice ? p.groqTtsVoice : DEFAULTS.groqTtsVoice,
    browserVoiceURI: typeof p.browserVoiceURI === "string" ? p.browserVoiceURI : DEFAULTS.browserVoiceURI,
    speechRate: rate,
  };
}

export function loadPrefs(): TranslationPrefs {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return { ...DEFAULTS };
    return coercePrefs(JSON.parse(raw));
  } catch {
    return { ...DEFAULTS };
  }
}

export function savePrefs(prefs: Partial<TranslationPrefs>): TranslationPrefs {
  const merged = { ...loadPrefs(), ...prefs };
  try {
    localStorage.setItem(KEY, JSON.stringify(merged));
  } catch {
    /* private mode — prefs are best-effort */
  }
  return merged;
}
