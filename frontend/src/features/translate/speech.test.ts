import { test } from "node:test";
import assert from "node:assert/strict";
import { pickBrowserVoice, shouldSpeak } from "./speech";
import { coercePrefs, loadPrefs, savePrefs } from "../settings/preferences";

// ---------------------------------------------------------------- prefs coerce

test("coercePrefs: defaults fill missing keys from an old stored shape", () => {
  const merged = coercePrefs({ saveHistory: false, mirrorPreview: true });
  assert.equal(merged.saveHistory, false);
  assert.equal(merged.mirrorPreview, true);
  assert.equal(merged.autoSpeak, false);
  assert.equal(merged.ttsProvider, "auto");
  assert.equal(merged.groqTtsVoice, "hannah");
  assert.equal(merged.speechRate, 0.9);
  assert.equal(merged.browserVoiceURI, null);
});

test("coercePrefs: invalid provider falls back to auto", () => {
  assert.equal(coercePrefs({ ttsProvider: "not-a-provider" }).ttsProvider, "auto");
  assert.equal(coercePrefs({ ttsProvider: "groq" }).ttsProvider, "groq");
  assert.equal(coercePrefs({ ttsProvider: "browser" }).ttsProvider, "browser");
});

test("coercePrefs: speechRate is clamped to the supported range", () => {
  assert.equal(coercePrefs({ speechRate: 5 }).speechRate, 2);
  assert.equal(coercePrefs({ speechRate: 0.1 }).speechRate, 0.5);
  assert.equal(coercePrefs({ speechRate: 1.25 }).speechRate, 1.25);
  assert.equal(coercePrefs({ speechRate: Number.NaN }).speechRate, 0.9);
});

test("loadPrefs/savePrefs round-trip through localStorage", () => {
  const store = new Map<string, string>();
  const storage: Storage = {
    get length() { return store.size; },
    clear: () => store.clear(),
    getItem: (k: string) => store.get(k) ?? null,
    key: (i: number) => [...store.keys()][i] ?? null,
    setItem: (k: string, v: string) => void store.set(k, String(v)),
    removeItem: (k: string) => void store.delete(k),
  };
  (globalThis as unknown as { localStorage: Storage }).localStorage = storage;
  savePrefs({ autoSpeak: true, ttsProvider: "groq", groqTtsVoice: "troy", speechRate: 1.25 });
  const prefs = loadPrefs();
  assert.equal(prefs.autoSpeak, true);
  assert.equal(prefs.ttsProvider, "groq");
  assert.equal(prefs.groqTtsVoice, "troy");
  assert.equal(prefs.speechRate, 1.25);
  delete (globalThis as unknown as { localStorage?: Storage }).localStorage;
});

// ------------------------------------------------------------- browser voices

const fakeVoice = (voiceURI: string, lang: string, name = voiceURI): SpeechSynthesisVoice =>
  ({ voiceURI, lang, name, default: false, localService: true } as SpeechSynthesisVoice);

test("pickBrowserVoice honors a stored preference", () => {
  const voices = [fakeVoice("a", "en-US"), fakeVoice("b", "en-IN")];
  assert.equal(pickBrowserVoice(voices, "b")?.voiceURI, "b");
});

test("pickBrowserVoice prefers en-IN over other English locales", () => {
  const voices = [fakeVoice("us", "en-US"), fakeVoice("in", "en-IN")];
  assert.equal(pickBrowserVoice(voices, null)?.voiceURI, "in");
});

test("pickBrowserVoice falls back to any voice when URI is stale", () => {
  const voices = [fakeVoice("x", "fr-FR")];
  assert.equal(pickBrowserVoice(voices, "missing")?.voiceURI, "x");
  assert.equal(pickBrowserVoice([], null), null);
});

// --------------------------------------------------------------- auto-speak

test("shouldSpeak dedupes identical repeated sentences but not new ones", () => {
  assert.equal(shouldSpeak(null, "Hello."), true);
  assert.equal(shouldSpeak("Hello.", "Hello."), false);
  assert.equal(shouldSpeak("Hello.", "Hello, water."), true);
  assert.equal(shouldSpeak("Hello.", ""), false);
  assert.equal(shouldSpeak("Hello.", "   "), false);
});
