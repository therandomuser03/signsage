import { test } from "node:test";
import assert from "node:assert/strict";
import { TranslationSocket } from "./client";
import type { ErrorMessage, TranslationTextMessage } from "./protocol";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

class FakeSocket {
  static OPEN = 1;
  url: string;
  readyState = 1;
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onerror: ((e: Event) => void) | null = null;
  onclose: (() => void) | null = null;
  sent: string[] = [];
  constructor(url: string) {
    this.url = url;
    FakeSocket.instances.push(this);
  }
  send(data: string) {
    this.sent.push(data);
  }
  close() {
    this.readyState = 3;
    this.onclose?.();
  }
  serverOpen() {
    this.onopen?.();
  }
  serverMessage(obj: unknown) {
    this.onmessage?.({ data: JSON.stringify(obj) });
  }
  serverClose() {
    this.readyState = 3;
    this.onclose?.();
  }
  static instances: FakeSocket[] = [];
  static reset() {
    FakeSocket.instances = [];
  }
}

const MODEL_UNAVAILABLE = {
  type: "error",
  code: "MODEL_UNAVAILABLE",
  message: "checkpoint has not passed the deployment gate",
};

function makeSocket(hooks: {
  onError?: (m: ErrorMessage) => void;
  onTranslation?: (m: TranslationTextMessage) => void;
} = {}) {
  const errors: ErrorMessage[] = [];
  const translations: TranslationTextMessage[] = [];
  const socket = new TranslationSocket({
    url: "ws://x/ws/translate",
    onTranslation: (m) => {
      translations.push(m);
      hooks.onTranslation?.(m);
    },
    onError: (m) => {
      errors.push(m);
      hooks.onError?.(m);
    },
    maxReconnectAttempts: 5,
    reconnectBaseDelayMs: 5,
    webSocketFactory: (url: string) => new FakeSocket(url) as unknown as WebSocket,
  });
  return { socket, errors, translations };
}

test("MODEL_UNAVAILABLE + normal close: error delivered, no reconnect loop", async () => {
  FakeSocket.reset();
  const { socket, errors } = makeSocket();
  socket.connect();
  FakeSocket.instances[0].serverOpen();
  // Exact backend sequence: error message first, then a normal close.
  FakeSocket.instances[0].serverMessage(MODEL_UNAVAILABLE);
  FakeSocket.instances[0].serverClose();
  await sleep(60); // well past all 5 backoff steps (5+10+20+40+80ms)
  assert.equal(socket.modelUnavailable, true);
  assert.equal(errors.length, 1);
  assert.equal(errors[0].code, "MODEL_UNAVAILABLE");
  assert.equal(FakeSocket.instances.length, 1); // suppressed: no retry
  socket.close();
});

test("transient close without any error still auto-retries", async () => {
  FakeSocket.reset();
  const { socket, errors } = makeSocket();
  socket.connect();
  FakeSocket.instances[0].serverOpen();
  FakeSocket.instances[0].serverClose(); // dropped connection, no verdict
  await sleep(30);
  assert.equal(socket.modelUnavailable, false);
  assert.equal(errors.length, 0);
  assert.equal(FakeSocket.instances.length, 2); // genuine retry preserved
  socket.close();
});

test("non-terminal error (SESSION_ERROR) does not suppress retries", async () => {
  FakeSocket.reset();
  const { socket, errors } = makeSocket();
  socket.connect();
  FakeSocket.instances[0].serverOpen();
  FakeSocket.instances[0].serverMessage({
    type: "error",
    code: "SESSION_ERROR",
    message: "too long",
  });
  FakeSocket.instances[0].serverClose();
  await sleep(30);
  assert.equal(socket.modelUnavailable, false);
  assert.equal(errors.length, 1);
  assert.equal(FakeSocket.instances.length, 2); // retried
  socket.close();
});

test("manual connect() after MODEL_UNAVAILABLE resets and retries work again", async () => {
  FakeSocket.reset();
  const { socket } = makeSocket();
  socket.connect();
  FakeSocket.instances[0].serverMessage(MODEL_UNAVAILABLE);
  FakeSocket.instances[0].serverClose();
  await sleep(40);
  assert.equal(FakeSocket.instances.length, 1);
  socket.connect(); // manual retry (Reconnect button path)
  assert.equal(socket.modelUnavailable, false);
  FakeSocket.instances[1].serverOpen();
  FakeSocket.instances[1].serverClose();
  await sleep(30);
  assert.equal(FakeSocket.instances.length, 3); // auto-retry re-armed
  socket.close();
});

test("malformed frames are ignored without callbacks or throws", () => {
  FakeSocket.reset();
  const { socket, errors, translations } = makeSocket();
  socket.connect();
  FakeSocket.instances[0].serverOpen();
  FakeSocket.instances[0].onmessage?.({ data: "not-json{{{" });
  FakeSocket.instances[0].onmessage?.({ data: JSON.stringify({ type: "nope" }) });
  assert.equal(errors.length, 0);
  assert.equal(translations.length, 0);
  assert.equal(socket.modelUnavailable, false);
  socket.close();
});

test("auth message carries history_session_id when getHistorySessionId is set", async () => {
  FakeSocket.reset();
  const socket = new TranslationSocket({
    url: "ws://x/ws/translate",
    getAccessToken: async () => "tok-1",
    getHistorySessionId: () => "interaction-1",
    onTranslation: () => {},
    webSocketFactory: (url: string) => new FakeSocket(url) as unknown as WebSocket,
  });
  socket.connect();
  const fake = FakeSocket.instances[FakeSocket.instances.length - 1];
  fake.serverOpen();
  await sleep(20);
  const auth = fake.sent.map((s) => JSON.parse(s)).find((m) => m.type === "auth");
  assert.deepEqual(auth, { type: "auth", access_token: "tok-1", history_session_id: "interaction-1" });
  socket.close();
});

test("sentenceBoundary forwards client_key for idempotent history writes", () => {
  FakeSocket.reset();
  const socket = new TranslationSocket({
    url: "ws://x/ws/translate",
    onTranslation: () => {},
    webSocketFactory: (url: string) => new FakeSocket(url) as unknown as WebSocket,
  });
  socket.connect();
  const fake = FakeSocket.instances[FakeSocket.instances.length - 1];
  fake.serverOpen();
  socket.sentenceBoundary(true, "sentence-key-1");
  const boundary = fake.sent.map((s) => JSON.parse(s)).find((m) => m.type === "sentence.boundary");
  assert.deepEqual(boundary, { type: "sentence.boundary", save_history: true, client_key: "sentence-key-1" });
  socket.close();
});
