import { Connection, type ConnectionStatus } from "./connection";
import {
  COORDINATES_PER_LANDMARK, LANDMARK_COUNT, parseServerMessage,
  type ClientMessage, type ErrorMessage, type TranslationTextMessage,
} from "./protocol";

export interface TranslationSocketOptions {
  url: string;
  /** Optional. If it returns a token, the socket authenticates right after opening (enables history). */
  getAccessToken?: () => Promise<string | null>;
  onTranslation: (message: TranslationTextMessage) => void;
  onError?: (message: ErrorMessage) => void;
  onAuthenticated?: (userId: string) => void;
  onStatus?: (status: ConnectionStatus) => void;
  webSocketFactory?: (url: string) => WebSocket;
  /** 0 (default) = never auto-reconnect. The server keeps per-connection
   * state, so a reconnect starts a fresh backend session. */
  maxReconnectAttempts?: number;
  reconnectBaseDelayMs?: number;
}

/**
 * Live Sign->Text client.
 *   connect() -> (auth) -> sendFrame() x N -> sentenceBoundary() -> onTranslation
 * Frames are the 543 MediaPipe Holistic landmarks in the exact order of the
 * model's landmark_contract.json. Send raw coordinates: the backend normalizes.
 */
export class TranslationSocket {
  private readonly connection: Connection;
  private readonly opts: TranslationSocketOptions;

  constructor(opts: TranslationSocketOptions) {
    this.opts = opts;
    this.connection = new Connection({
      url: opts.url,
      webSocketFactory: opts.webSocketFactory,
      maxReconnectAttempts: opts.maxReconnectAttempts,
      reconnectBaseDelayMs: opts.reconnectBaseDelayMs,
      onStatus: opts.onStatus,
      onOpen: () => void this.authenticate(),
      onMessage: (data) => this.handle(data),
    });
  }

  get status(): ConnectionStatus {
    return this.connection.status;
  }

  connect(): void {
    this.connection.connect();
  }

  close(): void {
    this.connection.close();
  }

  sendFrame(landmarks: number[][]): boolean {
    if (landmarks.length !== LANDMARK_COUNT || landmarks.some((p) => p.length !== COORDINATES_PER_LANDMARK)) {
      throw new Error(`Expected ${LANDMARK_COUNT} landmarks of ${COORDINATES_PER_LANDMARK} coordinates.`);
    }
    return this.post({ type: "landmarks.frame", landmarks });
  }

  sentenceBoundary(saveHistory = true): boolean {
    return this.post({ type: "sentence.boundary", save_history: saveHistory });
  }

  private post(message: ClientMessage): boolean {
    return this.connection.send(JSON.stringify(message));
  }

  private async authenticate(): Promise<void> {
    const token = this.opts.getAccessToken ? await this.opts.getAccessToken() : null;
    if (token) this.post({ type: "auth", access_token: token });
  }

  private handle(data: string): void {
    const message = parseServerMessage(data);
    if (!message) return;
    if (message.type === "translation.text") this.opts.onTranslation(message);
    else if (message.type === "auth.ok") this.opts.onAuthenticated?.(message.user_id);
    else this.opts.onError?.(message);
  }
}
