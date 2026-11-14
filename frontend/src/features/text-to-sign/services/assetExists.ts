// SignSage — avatar asset existence check.
//
// Vercel (frontend/vercel.json) rewrites every path to /index.html, so a
// MISSING file (signsage.vrm, signs/*.glb) returns the HTML page with HTTP 200.
// A plain res.ok check would therefore report the avatar as available and
// loading it would fail. This helper is the single place used by every avatar
// availability check and loader fetch so the behaviour is consistent.
//
// Asset is available only if:
//   - res.ok, AND
//   - content-type is not text/html (SPA fallback page), AND
//   - for VRM / GLB, content-type is not any text/* type.
//
// HEAD is used because content-type is reliable for the SPA rewrite case
// (Vercel serves index.html as text/html). A GET with a small Range header
// is the fallback if HEAD ever proves unreliable.
//
// `fetchImpl` defaults to the global `fetch` and is injected for unit testing
// so tests do not race on a shared global.
export async function assetExists(
  url: string,
  fetchImpl: typeof fetch = globalThis.fetch,
): Promise<boolean> {
  try {
    const res = await fetchImpl(url, { method: "HEAD" });
    if (!res.ok) return false;
    const contentType = res.headers.get("content-type") ?? "";
    // Reject the SPA fallback page and any text/* body masquerading as an asset.
    if (contentType.startsWith("text/")) return false;
    return true;
  } catch {
    // Network error, CORS, DNS failure, etc. -> treat as unavailable.
    return false;
  }
}

/** Decides whether the hand-authored VRM avatar should be used.
 * The VRM avatar is only used when both the VRM file and the manifest are
 * actually available (not an HTML fallback page). Otherwise the procedural
 * avatar is shown. */
export function shouldUseVrm(vrmExists: boolean, manifestAvailable: boolean): boolean {
  return vrmExists && manifestAvailable;
}
