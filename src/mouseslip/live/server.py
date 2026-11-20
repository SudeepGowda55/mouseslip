"""The HTTP face of the watcher: state as JSON, updates as server-sent events, and the web page if there is one."""

import asyncio
from contextlib import asynccontextmanager
import json
import logging

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from mouseslip.live.watcher import Watcher
from mouseslip.predict import ROOT

KEEPALIVE_SECONDS = 15
log = logging.getLogger("mouseslip.live")


def create_app(watcher: Watcher) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.task = asyncio.create_task(watcher.run())
        app.state.task.add_done_callback(lambda task: task.cancelled() or log.error("the watcher stopped: %r", task.exception()))
        yield
        app.state.task.cancel()
        await asyncio.gather(app.state.task, return_exceptions=True)

    app = FastAPI(title="Mouse Slip live", lifespan=lifespan)

    @app.get("/api/health")
    def health() -> dict:
        task = app.state.task
        error = repr(task.exception()) if task.done() and not task.cancelled() else None
        return {"ok": not task.done(), "model": watcher.predictor.name, "games": len(watcher.games), "error": error}

    @app.get("/api/state")
    def state() -> dict:
        return watcher.state()

    @app.get("/api/results")
    def results() -> dict:
        """The offline test-set results, one entry per file in results/ (the baselines and each trained model)."""
        return {path.stem: json.loads(path.read_text()) for path in sorted((ROOT / "results").glob("*.json"))}

    @app.get("/api/events")
    async def events() -> StreamingResponse:
        queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        watcher.subscribers.add(queue)

        async def lines():
            try:
                yield ": connected\n\n"
                while True:
                    try:
                        yield f"data: {json.dumps(await asyncio.wait_for(queue.get(), KEEPALIVE_SECONDS))}\n\n"
                    except TimeoutError:
                        yield ": keep-alive\n\n"
            finally:
                watcher.subscribers.discard(queue)

        return StreamingResponse(lines(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    if (ROOT / "web").is_dir():  # mounted last so it cannot shadow /api
        app.mount("/", StaticFiles(directory=ROOT / "web", html=True))
    return app
