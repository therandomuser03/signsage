# ponytail: flat FastAPI entry — split into routers only when >200 lines
from fastapi import FastAPI, WebSocket

app = FastAPI(title="SignSage")

@app.get("/health")
def health(): return {"status": "ok"}

# ponytail: WebSocket lives here until it justifies its own module
@app.websocket("/ws")
async def ws(ws: WebSocket):
    await ws.accept()
    await ws.close()
