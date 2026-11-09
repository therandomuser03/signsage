import { test } from "node:test";
import assert from "node:assert/strict";
import { ApiClient } from "./client";
import type { ComposeResponse, HistoryListResponse } from "../../types/api";

test("apiClient history requests carry the Authorization Bearer token", async () => {
  let capturedHeaders: any = null;
  let capturedUrl: string | null = null;

  const fetchImpl = async (_input: any, init?: any): Promise<Response> => {
    capturedUrl = String(_input);
    capturedHeaders = init?.headers ?? null;
    const body: HistoryListResponse = { items: [], total: 0, limit: 20, offset: 0 };
    return new Response(JSON.stringify(body), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  };

  const client = new ApiClient(
    "http://test.local",
    async () => "test-access-token-123",
    fetchImpl,
  );

  await client.listHistory(20, 0);
  assert.equal(capturedUrl, "http://test.local/api/v1/history?limit=20&offset=0");
  assert.equal(capturedHeaders?.Authorization, "Bearer test-access-token-123");

  // Guests (no token) send no Authorization header.
  capturedHeaders = null;
  const guestClient = new ApiClient("http://test.local", async () => null, fetchImpl);
  await guestClient.listHistory();
  assert.equal(Object.keys(capturedHeaders ?? {}).length, 0);
});

test("apiClient compose requests carry the Authorization Bearer token and payload", async () => {
  let capturedBody: any = null;
  let capturedHeaders: any = null;

  const fetchImpl = async (_input: any, init?: any): Promise<Response> => {
    capturedBody = init?.body ? JSON.parse(String(init.body)) : null;
    capturedHeaders = init?.headers ?? null;
    const body: ComposeResponse = { sentence: "Hello, help, me.", method: "fallback" };
    return new Response(JSON.stringify(body), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  };

  const client = new ApiClient(
    "http://test.local",
    async () => "test-access-token-123",
    fetchImpl,
  );

  await client.compose({ words: ["hello", "help", "me"], confidences: [0.9, 0.8, 0.7], save_history: true });
  assert.equal(capturedHeaders?.Authorization, "Bearer test-access-token-123");
  assert.deepEqual(capturedBody, {
    words: ["hello", "help", "me"],
    confidences: [0.9, 0.8, 0.7],
    save_history: true,
  });

  // Guests (no token) send no Authorization header.
  capturedHeaders = null;
  capturedBody = null;
  const guestClient = new ApiClient("http://test.local", async () => null, fetchImpl);
  await guestClient.compose({ words: ["water"], save_history: true });
  assert.equal(capturedHeaders?.Authorization, undefined);
  assert.deepEqual(capturedBody, { words: ["water"], save_history: true });
});

test("apiClient deleteHistoryItem carries the Authorization Bearer token", async () => {
  let capturedUrl: string | null = null;
  let capturedHeaders: any = null;

  const fetchImpl = async (_input: any, init?: any): Promise<Response> => {
    capturedUrl = String(_input);
    capturedHeaders = init?.headers ?? null;
    return new Response(JSON.stringify({ deleted: 1 }), {
      status: 200,
      headers: { "content-type": "application/json" },
    });
  };

  const client = new ApiClient(
    "http://test.local",
    async () => "test-access-token-123",
    fetchImpl,
  );

  await client.deleteHistoryItem("rec-1");
  assert.equal(capturedUrl, "http://test.local/api/v1/history/rec-1");
  assert.equal(capturedHeaders?.Authorization, "Bearer test-access-token-123");

  // Guests (no token) send no Authorization header.
  capturedHeaders = null;
  const guestClient = new ApiClient("http://test.local", async () => null, fetchImpl);
  await guestClient.deleteHistoryItem("rec-1");
  assert.equal(Object.keys(capturedHeaders ?? {}).length, 0);
});
