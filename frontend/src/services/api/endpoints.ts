// Every backend path in one place. tests/contract/test_frontend_contract.py
// checks each one exists in the backend's OpenAPI schema.
export const endpoints = {
  health: "/api/v1/health",
  capabilities: "/api/v1/capabilities",
  models: "/api/v1/models",
  me: "/api/v1/auth/me",
  history: "/api/v1/history",
  historySelfcheck: "/api/v1/history/selfcheck",
  historyItem: (id: string) => `/api/v1/history/${encodeURIComponent(id)}`,
  historyBulkDelete: "/api/v1/history/bulk-delete",
  speechSpeak: "/api/v1/speech/speak",
  signToText: "/api/v1/translation/sign-to-text",
  avatarGloss: "/api/v1/text-to-sign",
  avatarVocabulary: "/api/v1/text-to-sign/vocabulary",
  classifyImage: "/api/v1/classification/image",
  videoUpload: "/api/v1/translation/video",
  islimTranslate: "/api/v1/translation/islim/translate",
  islimTranslateVideo: "/api/v1/translation/islim/translate-video",
  word: "/api/v1/translation/word",
  sentence: "/api/v1/translation/sentence",
  compose: "/api/v1/translation/compose",
} as const;

export const WS_PATH = "/ws/translate";
