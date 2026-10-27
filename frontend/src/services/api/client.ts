import type {
  AvatarGlossRequest, AvatarGlossResponse, AvatarVocabularyResponse, CapabilitiesResponse,
  DeletedResponse, HealthResponse, HistoryListResponse, ImageClassifyRequest, ImageClassifyResponse,
  MeResponse, ModelsResponse, SignToTextRequest, SignToTextResponse, VideoTranslationResponse,
} from "../../types/api";
import { endpoints } from "./endpoints";
import { ApiError, NetworkError } from "./errors";

export type AccessTokenProvider = () => Promise<string | null>;

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

  private async request<T>(method: string, path: string, body?: unknown): Promise<T> {
    const headers: Record<string, string> = {};
    if (body !== undefined) headers["Content-Type"] = "application/json";
    const token = this.getAccessToken ? await this.getAccessToken() : null;
    if (token) headers["Authorization"] = `Bearer ${token}`;

    let res: Response;
    try {
      res = await this.fetchImpl(this.baseUrl + path, {
        method,
        headers,
        body: body === undefined ? undefined : JSON.stringify(body),
      });
    } catch {
      throw new NetworkError();
    }
    if (!res.ok) throw await ApiError.fromResponse(res);
    return (await res.json()) as T;
  }

  health = () => this.request<HealthResponse>("GET", endpoints.health);
  capabilities = () => this.request<CapabilitiesResponse>("GET", endpoints.capabilities);
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

  listHistory = (limit = 20, offset = 0) =>
    this.request<HistoryListResponse>("GET", `${endpoints.history}?limit=${limit}&offset=${offset}`);
  deleteHistoryItem = (id: string) => this.request<DeletedResponse>("DELETE", endpoints.historyItem(id));
  clearHistory = () => this.request<DeletedResponse>("DELETE", endpoints.history);
}
