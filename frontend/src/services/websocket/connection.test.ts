import { test } from "node:test";
import assert from "node:assert/strict";
import { Connection } from "./connection";

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** Controllable WebSocket stand-in. Server events are triggered manually. */
class FakeSocket {
  static OPEN = 1;
  url: string;
  readyState = 1;
  onopen: (() => void) | null = null;
  onmessage: ((e: { data: string }) => void) | null = null;
  onerror: ((e: Event) => void) | null = null;
  onclose: (() => void) | null = null;
  sent: string[] = [];
  closeCalls = 0;
  constructor(url: string) {
    this.url = url;
    FakeSocket.instances.push(this);
  }
  send(data: string) {
    this.sent.push(data);
  }
  close() {
    this.closeCalls += 1;
    this.readyState = 3;
    this.onclose?.();
  }
  serverOpen() {
    this.onopen?.();
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

const factory = (statuses: string[]) => ({
  make: (url: string) => {
    const s = new FakeSocket(url);
    return s as unknown as WebSocket;
  },
  statuses,
});

test("transient server close triggers one bounded reconnect", async () => {
  FakeSocket.reset();
  const seen: string[] = [];
  const c = new Connection({
    url: "ws://x/ws/translate",
    onMessage: () => {},
    onStatus: (s) => seen.push(s),
    maxReconnectAttempts: 2,
    reconnectBaseDelayMs: 5,
    webSocketFactory: factory(seen).make,
  });
  c.connect();
  FakeSocket.instances[0].serverOpen();
  FakeSocket.instances[0].serverClose(); // transient: no terminal verdict
  await sleep(30);
  assert.equal(FakeSocket.instances.length, 2); // reconnected
  assert.deepEqual(seen, ["connecting", "open", "closed", "connecting"]);
  c.close();
});

test("suppressReconnect stops the retry after a terminal verdict", async () => {
  FakeSocket.reset();
  const c = new Connection({
    url: "ws://x/ws/translate",
    onMessage: () => {},
    maxReconnectAttempts: 5,
    reconnectBaseDelayMs: 5,
    webSocketFactory: factory([]).make,
  });
  c.connect();
  FakeSocket.instances[0].serverOpen();
  c.suppressReconnect(); // e.g. MODEL_UNAVAILABLE arrived just before close
  FakeSocket.instances[0].serverClose();
  await sleep(40); // past every backoff step
  assert.equal(FakeSocket.instances.length, 1); // no retry
  assert.equal(c.status, "closed");
  c.close();
});

test("connect() re-arms retries after suppression (manual retry)", async () => {
  FakeSocket.reset();
  const c = new Connection({
    url: "ws://x/ws/translate",
    onMessage: () => {},
    maxReconnectAttempts: 5,
    reconnectBaseDelayMs: 5,
    webSocketFactory: factory([]).make,
  });
  c.connect();
  c.suppressReconnect();
  FakeSocket.instances[0].serverClose();
  await sleep(30);
  assert.equal(FakeSocket.instances.length, 1);
  c.connect(); // manual retry: suppression cleared
  FakeSocket.instances[1].serverOpen();
  FakeSocket.instances[1].serverClose(); // transient again
  await sleep(30);
  assert.equal(FakeSocket.instances.length, 3); // retry resumed
  c.close();
});

test("default (max 0) never reconnects", async () => {
  FakeSocket.reset();
  const c = new Connection({
    url: "ws://x/ws/translate",
    onMessage: () => {},
    webSocketFactory: factory([]).make,
  });
  c.connect();
  FakeSocket.instances[0].serverClose();
  await sleep(20);
  assert.equal(FakeSocket.instances.length, 1);
  c.close();
});
