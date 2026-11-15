import type {
  AvatarGlossRequest, AvatarGlossResponse, AvatarVocabularyResponse, CapabilitiesResponse,
  BulkDeleteRequest,
  ComposeRequest, ComposeResponse, DeletedResponse, HealthResponse, HistoryListResponse,
  HistorySelfcheckResponse, HistorySessionDetail,
  ImageClassifyRequest, ImageClassifyResponse, LandmarkFramePayload, MeResponse, ModelsResponse,
  SentenceRecognitionResponse,
  SignToTextRequest, SignToTextResponse, VideoTranslationResponse, WordRecognitionResponse,
} from "../../types/api";
import { endpoints } from "./endpoints";
import { ApiError, NetworkError } from "./errors";

export type AccessTokenProvider = () => Promise<string | null>;
export type RequestOpts = { retries?: number; timeoutMs?: number };

export { ApiError, NetworkError };

export class ApiClient {
  private readonly baseUrl: string;
  private readonly getAccessToken?: AccessTokenProvider;
  private readonly fetchImpl: typeof fetch;

  constructor(
    baseUrl: string,
    getAccessToken?: AccessTokenProvider,
    fetchImpl: typeof fetch = (...args) => fetch(...args),
  ) {
    this.baseUrl = baseUrl;
    this.getAccessToken = getAccessToken;
    this.fetchImpl = fetchImpl;
  }

  private async request<T>(
    method: string,
    path: string,
    body?: unknown,
    opts: { retries?: number; timeoutMs?: number } = {},
  ): Promise<T> {
    const { retries = 0, timeoutMs = 0 } = opts;

    const requestOnce = async (): Promise<Response> => {
      const headers: Record<string, string> = {};
      if (body !== undefined) headers["Content-Type"] = "application/json";
      const token = this.getAccessToken ? await this.getAccessToken() : null;
      if (token) headers["Authorization"] = `Bearer ${token}`;

      const init: RequestInit = { method, headers, body: body === undefined ? undefined : JSON.stringify(body) };
      let promise: Promise<Response> = this.fetchImpl(this.baseUrl + path, init);
      if (timeoutMs > 0) {
        // Race against a timeout so a sleeping/cold backend never hangs the tab.
        promise = Promise.race([
          promise,
          new Promise<never>((_, reject) => setTimeout(() => reject(new NetworkError("Request timed out.")), timeoutMs)),
        ]);
      }
      return promise;
    };

    let res: Response;
    for (let attempt = 0; ; attempt++) {
      try {
        res = await requestOnce();
        break;
      } catch (err) {
        // One retry on network-level failure (cold backend, flaky Wi-Fi),
        // never on a 4xx/5xx response (those are thrown by fromResponse).
        if (attempt >= retries || !(err instanceof NetworkError)) throw err;
        await new Promise((r) => setTimeout(r, 1000));
      }
    }
    if (!res.ok) throw await ApiError.fromResponse(res);
    return (await res.json()) as T;
  }

  health = () => this.request<HealthResponse>("GET", endpoints.health);
  capabilities = (opts?: RequestOpts) =>
    this.request<CapabilitiesResponse>("GET", endpoints.capabilities, undefined, opts);
  models = () => this.request<ModelsResponse>("GET", endpoints.models);
  me = () => this.request<MeResponse>("GET", endpoints.me);

  avatarGloss = (req: AvatarGlossRequest) => this.request<AvatarGlossResponse>("POST", endpoints.avatarGloss, req);
  avatarVocabulary = () => this.request<AvatarVocabularyResponse>("GET", endpoints.avatarVocabulary);
  signToText = (req: SignToTextRequest) => this.request<SignToTextResponse>("POST", endpoints.signToText, req);

  classifyImage = (req: ImageClassifyRequest) =>
    this.request<ImageClassifyResponse>("POST", endpoints.classifyImage, req);

  uploadVideo = async (file: File): Promise<VideoTranslationResponse> => {
    const form = new FormData();
    form.append("file", file);
    const token = this.getAccessToken ? await this.getAccessToken() : null;
    const headers: Record<string, string> = {};
    if (token) headers["Authorization"] = `Bearer ${token}`;
    const res = await this.fetchImpl(this.baseUrl + endpoints.videoUpload, {
      method: "POST",
      headers,
      body: form,
    });
    if (!res.ok) throw await ApiError.fromResponse(res);
    return (await res.json()) as VideoTranslationResponse;
  };

  word = (req: { frames: LandmarkFramePayload[]; true_label?: string }, opts?: RequestOpts) =>
    this.request<WordRecognitionResponse>("POST", endpoints.word, req, opts);

  /** Proxy for the backend /api/v1/speech/speak endpoint (Groq Orpheus when
   *  configured; 503 otherwise — callers fall back to browser speech). */
  speakWav = async (text: string, voice?: string): Promise<Blob> => {
    const path = endpoints.speechSpeak;
    const token = this.getAccessToken ? await this.getAccessToken() : null;
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (token) headers["Authorization"] = `Bearer ${token}`;
    const res = await this.fetchImpl(this.baseUrl + path, {
      method: "POST",
      headers,
      body: JSON.stringify(voice ? { text, voice } : { text }),
    });
    if (!res.ok) {
      let code: string | undefined;
      try {
        const body = (await res.json()) as { code?: string };
        code = body?.code;
      } catch {
        /* ignore non-JSON error bodies */
      }
      throw new ApiError(
        res.status,
        code ?? (res.status === 422 ? "VALIDATION_ERROR" : "HTTP_ERROR"),
        "Speech service unavailable.",
      );
    }
    return res.blob();
  };

  sentence = (
    req: { frames: LandmarkFramePayload[]; save_history?: boolean; client_session_key?: string; client_key?: string },
    opts?: RequestOpts,
  ) =>
    this.request<SentenceRecognitionResponse>("POST", endpoints.sentence, req, opts);

  compose = (req: ComposeRequest, opts?: RequestOpts) =>
    this.request<ComposeResponse>("POST", endpoints.compose, req, opts);

  listHistory = (
    limit = 20,
    offset = 0,
    opts: { direction?: "sign_to_text" | "text_to_sign"; q?: string; sort?: "newest" | "oldest" } = {},
  ) => {
    const params = new URLSearchParams({ limit: String(limit), offset: String(offset) });
    if (opts.direction) params.set("direction", opts.direction);
    if (opts.q) params.set("q", opts.q);
    if (opts.sort) params.set("sort", opts.sort);
    return this.request<HistoryListResponse>("GET", `${endpoints.history}?${params.toString()}`);
  };
  getHistorySession = (id: string) =>
    this.request<HistorySessionDetail>("GET", endpoints.historyItem(id));
  deleteHistoryItem = (id: string) => this.request<DeletedResponse>("DELETE", endpoints.historyItem(id));
  deleteHistoryItems = (ids: string[]) =>
    this.request<DeletedResponse>("POST", endpoints.historyBulkDelete, { ids } satisfies BulkDeleteRequest);
  clearHistory = () => this.request<DeletedResponse>("DELETE", endpoints.history);
  historySelfcheck = () => this.request<HistorySelfcheckResponse>("GET", endpoints.historySelfcheck);
}
