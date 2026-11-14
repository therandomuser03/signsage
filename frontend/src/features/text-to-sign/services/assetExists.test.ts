import { test } from "node:test";
import assert from "node:assert/strict";
import { assetExists } from "./assetExists";

function fetchMock(ct: string | null, status = 200): typeof fetch {
  return async () =>
    new Response(null, { status, headers: ct ? { "content-type": ct } : {} });
}

test("assetExists returns false for 200 text/html (Vercel SPA fallback)", async () => {
  assert.strictEqual(await assetExists("/models/avatar/signsage.vrm", fetchMock("text/html")), false);
  assert.strictEqual(await assetExists("/models/avatar/signs/hello.glb", fetchMock("text/html")), false);
});

test("assetExists returns false for 404", async () => {
  assert.strictEqual(await assetExists("/models/avatar/signsage.vrm", fetchMock(null, 404)), false);
});

test("assetExists returns true for 200 application/octet-stream", async () => {
  assert.strictEqual(
    await assetExists("/models/avatar/signsage.vrm", fetchMock("application/octet-stream")),
    true,
  );
});

test("assetExists returns true for 200 model/gltf-binary", async () => {
  assert.strictEqual(
    await assetExists("/models/avatar/signs/hello.glb", fetchMock("model/gltf-binary")),
    true,
  );
});

test("assetExists returns false for network error", async () => {
  const failingFetch: typeof fetch = async () => {
    throw new TypeError("Failed to fetch");
  };
  assert.strictEqual(await assetExists("/models/avatar/signsage.vrm", failingFetch), false);
});

test("assetExists returns false for 200 text/plain", async () => {
  assert.strictEqual(await assetExists("/models/avatar/signsage.vrm", fetchMock("text/plain")), false);
});
