import type {
  AvatarGlossRequest, AvatarGlossResponse, AvatarVocabularyResponse, CapabilitiesResponse,
  ComposeRequest, ComposeResponse, DeletedResponse, HealthResponse, HistoryListResponse,
  ImageClassifyRequest, ImageClassifyResponse, LandmarkFramePayload, MeResponse, ModelsResponse,
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

  compose = (req: ComposeRequest, opts?: RequestOpts) =>
    this.request<ComposeResponse>("POST", endpoints.compose, req, opts);

  listHistory = (limit = 20, offset = 0) =>
    this.request<HistoryListResponse>("GET", `${endpoints.history}?limit=${limit}&offset=${offset}`);
  deleteHistoryItem = (id: string) => this.request<DeletedResponse>("DELETE", endpoints.historyItem(id));
  clearHistory = () => this.request<DeletedResponse>("DELETE", endpoints.history);
}
