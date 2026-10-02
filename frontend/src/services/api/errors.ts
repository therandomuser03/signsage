import type { ErrorBody } from "../../types/api";

/** Any non-2xx response. `code` is the backend's stable error code (e.g. UNAUTHORIZED, MODEL_NOT_PROMOTED). */
export class ApiError extends Error {
  public readonly status: number;
  public readonly code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.status = status;
    this.code = code;
    this.name = "ApiError";
  }

  static async fromResponse(res: Response): Promise<ApiError> {
    let body: Partial<ErrorBody> & { detail?: unknown } = {};
    try {
      body = await res.json();
    } catch {
      /* non-JSON error body */
    }
    if (typeof body.code === "string") {
      return new ApiError(res.status, body.code, body.message ?? res.statusText);
    }
    // FastAPI request-validation errors look like {"detail": [...]}
    return new ApiError(res.status, res.status === 422 ? "VALIDATION_ERROR" : "HTTP_ERROR", res.statusText || `HTTP ${res.status}`);
  }

  get isAuthError(): boolean {
    return this.status === 401;
  }
}

/** The request never got a response (offline, CORS, server down). */
export class NetworkError extends Error {
  constructor(message = "Could not reach the SignSage server.") {
    super(message);
    this.name = "NetworkError";
  }
}
