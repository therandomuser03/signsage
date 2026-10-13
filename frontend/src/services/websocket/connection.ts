export type ConnectionStatus = "idle" | "connecting" | "open" | "closed";

export interface ConnectionOptions {
  url: string;
  onMessage: (data: string) => void;
  onOpen?: () => void;
  onStatus?: (status: ConnectionStatus) => void;
  onError?: (event: Event) => void;
  /** 0 (default) = never auto-reconnect. The server keeps per-connection state, so a reconnect starts a fresh session. */
  maxReconnectAttempts?: number;
  reconnectBaseDelayMs?: number;
  webSocketFactory?: (url: string) => WebSocket;
}

/** Raw socket lifecycle: connect, send, close, optional exponential-backoff reconnect. */
export class Connection {
  status: ConnectionStatus = "idle";
  private ws: WebSocket | null = null;
  private attempts = 0;
  private closedByUser = false;
  private reconnectSuppressed = false;
  private timer: ReturnType<typeof setTimeout> | null = null;
  private readonly opts: ConnectionOptions;

  constructor(opts: ConnectionOptions) {
    this.opts = opts;
  }

  private setStatus(status: ConnectionStatus) {
    this.status = status;
    this.opts.onStatus?.(status);
  }

  connect(): void {
    this.closedByUser = false;
    this.reconnectSuppressed = false;
    this.setStatus("connecting");
    const ws = (this.opts.webSocketFactory ?? ((u) => new WebSocket(u)))(this.opts.url);
    this.ws = ws;
    ws.onopen = () => {
      this.attempts = 0;
      this.setStatus("open");
      this.opts.onOpen?.();
    };
    ws.onmessage = (event) => {
      if (typeof event.data === "string") this.opts.onMessage(event.data);
    };
    ws.onerror = (event) => this.opts.onError?.(event);
    ws.onclose = () => {
      this.setStatus("closed");
      this.maybeReconnect();
    };
  }

  private maybeReconnect() {
    const max = this.opts.maxReconnectAttempts ?? 0;
    // A terminal server verdict (e.g. MODEL_UNAVAILABLE) suppresses retries:
    // reconnecting cannot change the deployment gate, it only flashes status.
    if (this.closedByUser || this.reconnectSuppressed || this.attempts >= max) return;
    const delay = (this.opts.reconnectBaseDelayMs ?? 500) * 2 ** this.attempts;
    this.attempts += 1;
    this.timer = setTimeout(() => this.connect(), delay);
  }

  /** Stop any further automatic reconnects (terminal server verdict).
   * Only another connect() re-arms retries. Does not touch the socket. */
  suppressReconnect(): void {
    this.reconnectSuppressed = true;
    if (this.timer) clearTimeout(this.timer);
    this.timer = null;
  }

  /** false if the socket isn't open (nothing is queued: stale landmark frames are worthless). */
  send(data: string): boolean {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
    this.ws.send(data);
    return true;
  }

  close(): void {
    this.closedByUser = true;
    if (this.timer) clearTimeout(this.timer);
    this.ws?.close();
  }
}
