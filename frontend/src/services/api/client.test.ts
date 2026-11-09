import { test } from "node:test";
import assert from "node:assert/strict";
import { ApiClient } from "./client";
import type { HistoryListResponse } from "../../types/api";

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
