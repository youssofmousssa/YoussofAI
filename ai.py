"""
youssof_backend.py

FastAPI backend that exposes a unified, themed API for multiple Groq models and
external image/video endpoints as requested.

Features:
- Endpoints with "Youssof" friendly names:
    /youssof/models              -> lists friendly names + capabilities
    /youssof/chat                -> Llama 3.3 70B (default conversational)
    /youssof/deep_reasoning      -> GPT OSS 120B (long-reasoning)
    /youssof/scout_tool_call     -> Llama 4 Scout (tooling/function-call style)
    /youssof/vision              -> Llama 4 Maverick (image understanding)
    /youssof/image_gen           -> external image gen API (sii3.moayman.top)
    /youssof/video_gen           -> external video gen API (image-to-video & text-to-video)
    /youssof/tts                 -> PlayAI TTS via Groq audio.speech.create
    /youssof/moderate           -> Llama Guard moderation model
    (No STT endpoint per request.)

- Streaming support for model responses (Server-Sent-Events style)
- File streaming for TTS output (returns WAV)
- Uses environment variable GROQ_API_KEY (defaults to the provided key if not set)
- Async HTTP calls via httpx for external APIs
- Clean error handling and typed Pydantic request/response models

Dependencies:
    pip install fastapi uvicorn httpx aiofiles pydantic groq

Run:
    GROQ_API_KEY="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" \
        uvicorn youssof_backend:app --host 0.0.0.0 --port 8000

Note: This file includes the API key as the default for convenience (per your demo note).
You should still prefer setting GROQ_API_KEY in env vars for production.

"""

import os
import json
import tempfile
import asyncio
from pathlib import Path
from typing import Optional, List, Any, Dict, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from pydantic import BaseModel
import httpx
import aiofiles

# Import the Groq client - assumes it's installed in the environment.
# If not present, pip install groq
try:
    from groq import Groq
except Exception as e:
    raise RuntimeError(
        "groq SDK not available. Install with `pip install groq` "
        "(or if you are in a locked environment, adapt this code)."
    ) from e

# ---------------------------
# Configuration & Constants
# ---------------------------

DEFAULT_GROQ_KEY = os.environ.get(
    "GROQ_API_KEY",
    "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",  # user-provided demo key
)

GROQ_CLIENT = Groq(api_key=DEFAULT_GROQ_KEY)

# Friendly mapping of "Youssof" names to actual model IDs and a short description.
YOUSSOF_MODELS = {
    "youssof_chat": {
        "display_name": "Youssof Chat",
        "model": "llama-3.3-70b-versatile",
        "description": "Default general chat + multilingual assistant (Llama 3.3 70B).",
        "streamable": True,
        "max_tokens": 1024,
    },
    "youssof_deep_reasoning": {
        "display_name": "Youssof Deep Reasoning",
        "model": "openai/gpt-oss-120b",
        "description": "Heavy-duty deep reasoning model (GPT OSS 120B).",
        "streamable": True,
        "max_tokens": 8192,
    },
    "youssof_scout": {
        "display_name": "Youssof Scout",
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "description": "Tool / function-calling style assistant.",
        "streamable": True,
        "max_tokens": 1024,
    },
    "youssof_maverick": {
        "display_name": "Youssof Maverick (Vision)",
        "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "description": "Image understanding and vision-enabled assistant.",
        "streamable": True,
        "max_tokens": 1024,
    },
    "youssof_guard": {
        "display_name": "Youssof Guard (Moderation)",
        "model": "meta-llama/llama-guard-4-12b",
        "description": "Safety and moderation checks.",
        "streamable": False,
        "max_tokens": 1024,
    },
    "youssof_tts": {
        "display_name": "Youssof TTS (PlayAI)",
        "model": "playai-tts",
        "description": "Text to speech via PlayAI TTS.",
        "streamable": False,
        "max_tokens": None,
    },
}

# External image/video endpoints (as provided by the user).
IMAGE_GEN_BASE = "https://sii3.moayman.top/api/flux-pro.php"
IMG_TO_VIDEO_BASE = "https://sii3.moayman.top/api/img-to-video.php"
TEXT_TO_VIDEO_BASE = "https://sii3.moayman.top/api/text-to-video.php"

# HTTP client for external requests
HTTPX_TIMEOUT = 60  # seconds
httpx_client = httpx.AsyncClient(timeout=HTTPX_TIMEOUT)


# ---------------------------
# Pydantic models (requests)
# ---------------------------

class ChatMessage(BaseModel):
    role: str  # "user" or "assistant" or "system"
    content: Any  # string OR structured (for vision: may include list with image_url)


class ChatRequest(BaseModel):
    model_key: Optional[str] = "youssof_chat"
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    max_completion_tokens: Optional[int] = None
    top_p: Optional[float] = 1.0
    stream: Optional[bool] = True


class SimpleTextRequest(BaseModel):
    model_key: Optional[str] = "youssof_chat"
    text: str
    temperature: Optional[float] = 1.0
    max_completion_tokens: Optional[int] = None
    stream: Optional[bool] = False


class ImageGenRequest(BaseModel):
    prompt: str
    n: Optional[int] = 1


class VideoGenRequest(BaseModel):
    prompt: str
    image_url: Optional[str] = None  # for image-to-video


class TTSRequest(BaseModel):
    input: str
    model_key: Optional[str] = "youssof_tts"
    voice: Optional[str] = "Aaliyah-PlayAI"
    response_format: Optional[str] = "wav"


class ModerateRequest(BaseModel):
    text: str


# ---------------------------
# FastAPI app
# ---------------------------

app = FastAPI(
    title="Youssof Multi-Model Gateway",
    description="One-stop backend to access Groq models and external image/video APIs. "
                "Models are presented with friendly Youssof branding.",
    version="1.0.0",
)


# ---------------------------
# Helper utilities
# ---------------------------

def get_model_entry(model_key: str):
    entry = YOUSSOF_MODELS.get(model_key)
    if not entry:
        raise HTTPException(status_code=400, detail=f"Unknown model_key: {model_key}")
    return entry


async def stream_groq_chat_response(
    model_id: str,
    messages: List[Dict[str, Any]],
    temperature: float = 1.0,
    max_completion_tokens: Optional[int] = None,
    top_p: float = 1.0,
) -> AsyncGenerator[bytes, None]:
    """
    Streams Groq chat predictions as bytes (SSE compatible).
    Each chunk will be yielded as a small JSON payload line prefixed by 'data:'.
    Note: Groq client's streaming interface is used as in your examples (sync iterator).
    The Groq SDK's streaming may be synchronous; here we adapt it to async by running
    in a thread. If the SDK supports async iteration, adapt accordingly.
    """
    loop = asyncio.get_running_loop()

    def run_and_yield():
        # Blocking call to Groq streaming generator — runs in thread
        try:
            completion = GROQ_CLIENT.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_completion_tokens or 1024,
                top_p=top_p,
                stream=True,
                stop=None,
            )
        except Exception as e:
            # send an error data chunk
            return [{"error": str(e)}]

        # The SDK yields chunks; we will collect and return a list of JSON-serializable chunks
        collected = []
        try:
            for chunk in completion:
                # Each chunk may have structure: chunk.choices[0].delta.content
                try:
                    token = chunk.choices[0].delta.get("content") if hasattr(chunk.choices[0].delta, "get") else getattr(chunk.choices[0].delta, "content", None)
                except Exception:
                    # fallback generic string
                    token = None
                # If token is None, try other fields
                if token is None:
                    token = getattr(chunk.choices[0].delta, "content", None) or getattr(chunk.choices[0], "content", None) or str(chunk)
                collected.append({"chunk": token})
        except Exception as e:
            collected.append({"error": str(e)})
        return collected

    # Run sync generator in threadpool
    chunks = await loop.run_in_executor(None, run_and_yield)
    # Yield SSE-style text/event-stream data lines
    for obj in chunks:
        payload = json.dumps(obj, ensure_ascii=False)
        yield f"data: {payload}\n\n".encode("utf-8")
        await asyncio.sleep(0.01)  # gentle pacing


# ---------------------------
# Endpoints
# ---------------------------

@app.get("/youssof/models")
async def list_models():
    """Return friendly model names and metadata, excluding hidden models."""
    result = []
    for key, entry in YOUSSOF_MODELS.items():
        if entry.get("hidden"):  # Skip models marked as hidden
            continue
        result.append({
            "model_key": key,
            "display_name": entry["display_name"],
            "description": entry["description"],
            "streamable": entry["streamable"],
            "model_id": entry["model"],
            console_note = {
            "console_url": "https://youssofai.onrender.com/",
            "note": "DONATE US TELEGRAM : t.me/youssofxmoussa",
        })



    return {"models": result, "console": console_note}


@app.post("/youssof/chat")
async def youssof_chat(req: ChatRequest):
    """
    Streamed chat endpoint for Youssof Chat (Llama 3.3 70B by default).
    Accepts messages array. Returns Server-Sent Events stream when stream=True.
    """
    entry = get_model_entry(req.model_key or "youssof_chat")
    model_id = entry["model"]
    max_tokens = req.max_completion_tokens or entry.get("max_tokens")

    # Convert Pydantic messages to plain dict structure expected by Groq SDK
    messages = []
    for m in req.messages:
        messages.append({"role": m.role, "content": m.content})

    if req.stream:
        generator = stream_groq_chat_response(
            model_id=model_id,
            messages=messages,
            temperature=req.temperature,
            max_completion_tokens=max_tokens,
            top_p=req.top_p,
        )
        return StreamingResponse(generator, media_type="text/event-stream")
    else:
        # Non-streaming path: call Groq synchronously (in thread) and return final content
        loop = asyncio.get_event_loop()

        def blocking_call():
            completion = GROQ_CLIENT.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=req.temperature,
                max_completion_tokens=max_tokens or 1024,
                top_p=req.top_p,
                stream=False,
                stop=None,
            )
            # attempt to extract text
            try:
                # structure differs by SDK; attempt common fields
                text = completion.choices[0].message.content
            except Exception:
                text = str(completion)
            return text

        content = await loop.run_in_executor(None, blocking_call)
        return {"model": entry["display_name"], "response": content}


@app.post("/youssof/deep_reasoning")
async def youssof_deep_reasoning(req: SimpleTextRequest):
    """
    Entrypoint for the deep reasoning model (GPT OSS 120B).
    Streams when requested.
    """
    entry = get_model_entry("youssof_deep_reasoning")
    messages = [
        {"role": "user", "content": req.text}
    ]
    if req.stream:
        generator = stream_groq_chat_response(
            model_id=entry["model"],
            messages=messages,
            temperature=req.temperature,
            max_completion_tokens=req.max_completion_tokens or entry["max_tokens"],
        )
        return StreamingResponse(generator, media_type="text/event-stream")
    else:
        # sync call in thread
        loop = asyncio.get_event_loop()

        def blocking():
            comp = GROQ_CLIENT.chat.completions.create(
                model=entry["model"],
                messages=messages,
                temperature=req.temperature,
                max_completion_tokens=req.max_completion_tokens or entry["max_tokens"],
                stream=False,
            )
            try:
                return comp.choices[0].message.content
            except Exception:
                return str(comp)
        result = await loop.run_in_executor(None, blocking)
        return {"model": entry["display_name"], "response": result}


@app.post("/youssof/scout_tool_call")
async def youssof_scout(req: ChatRequest):
    """
    Tool-calling / function-use endpoint (Scout). Accepts message structures that may
    include images or specialized content in message.content.
    """
    entry = get_model_entry("youssof_scout")
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    if req.stream:
        return StreamingResponse(
            stream_groq_chat_response(
                model_id=entry["model"],
                messages=messages,
                temperature=req.temperature,
                max_completion_tokens=req.max_completion_tokens or entry["max_tokens"],
            ),
            media_type="text/event-stream",
        )
    else:
        # blocking sync call
        loop = asyncio.get_event_loop()

        def blocking():
            comp = GROQ_CLIENT.chat.completions.create(
                model=entry["model"],
                messages=messages,
                temperature=req.temperature,
                max_completion_tokens=req.max_completion_tokens or entry["max_tokens"],
                stream=False,
            )
            try:
                return comp.choices[0].message.content
            except Exception:
                return str(comp)
        result = await loop.run_in_executor(None, blocking)
        return {"model": entry["display_name"], "response": result}


@app.post("/youssof/vision")
async def youssof_vision(req: ChatRequest):
    """
    Vision-capable endpoint (Maverick). Expects one message where content may be a list
    including an 'image_url' entry as in your example.
    Example content:
    [
        {"type": "text", "text": "Who is he?"},
        {"type": "image_url", "image_url": {"url": "https://..."}}
    ]
    """
    entry = get_model_entry("youssof_maverick")
    # Prepare messages: pass through content as-is (the model expects list content for vision)
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    if req.stream:
        return StreamingResponse(
            stream_groq_chat_response(
                model_id=entry["model"],
                messages=messages,
                temperature=req.temperature,
                max_completion_tokens=req.max_completion_tokens or entry["max_tokens"],
            ),
            media_type="text/event-stream",
        )
    else:
        loop = asyncio.get_event_loop()

        def blocking():
            comp = GROQ_CLIENT.chat.completions.create(
                model=entry["model"],
                messages=messages,
                temperature=req.temperature,
                max_completion_tokens=req.max_completion_tokens or entry["max_tokens"],
                stream=False,
            )
            try:
                return comp.choices[0].message.content
            except Exception:
                return str(comp)
        result = await loop.run_in_executor(None, blocking)
        return {"model": entry["display_name"], "response": result}


@app.post("/youssof/image_gen")
async def youssof_image_gen(req: ImageGenRequest):
    """
    Calls the external image-generation API (sii3.moayman.top).
    Returns the JSON that the service returns (with image URLs).
    """
    params = {"text": req.prompt}
    # if n > 1, the remote API might not support it; just call once per requested image
    try:
        r = await httpx_client.post(IMAGE_GEN_BASE, data=params)
        r.raise_for_status()
        return JSONResponse(content=r.json())
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Image service error: {str(e)}")


@app.post("/youssof/video_gen")
async def youssof_video_gen(req: VideoGenRequest):
    """
    Two modes:
    - If image_url provided -> image-to-video API
    - Otherwise -> text-to-video API
    """
    if req.image_url:
        params = {"text": req.prompt, "link": req.image_url}
        target = IMG_TO_VIDEO_BASE
    else:
        params = {"text": req.prompt}
        target = TEXT_TO_VIDEO_BASE

    try:
        r = await httpx_client.post(target, data=params)
        r.raise_for_status()
        return JSONResponse(content=r.json())
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Video service error: {str(e)}")


@app.post("/youssof/tts")
async def youssof_tts(req: TTSRequest):
    """
    Uses Groq audio.speech.create to generate TTS audio and returns a WAV file.
    """
    entry = get_model_entry(req.model_key or "youssof_tts")
    # Create a temporary file to stream to
    tmp_dir = tempfile.gettempdir()
    filename = Path(tmp_dir) / f"youssof_tts_{int(asyncio.get_event_loop().time() * 1000)}.wav"

    # Blocking call to Groq audio API (stream_to_file)
    def blocking_tts():
        try:
            response = GROQ_CLIENT.audio.speech.create(
                model=entry["model"],
                voice=req.voice,
                response_format=req.response_format,
                input=req.input,
            )
            # response is assumed to have stream_to_file method like your sample
            response.stream_to_file(str(filename))
            return str(filename)
        except Exception as e:
            raise RuntimeError(f"TTS generation failed: {e}")

    loop = asyncio.get_event_loop()
    try:
        filepath = await loop.run_in_executor(None, blocking_tts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Return file as response
    return FileResponse(path=filepath, media_type="audio/wav", filename=Path(filepath).name)


@app.post("/youssof/moderate")
async def youssof_moderate(req: ModerateRequest):
    """
    Run a moderation check via Llama Guard.
    Returns the model's moderation classification.
    """
    entry = get_model_entry("youssof_guard")
    messages = [
        {"role": "user", "content": req.text}
    ]

    loop = asyncio.get_event_loop()

    def blocking():
        comp = GROQ_CLIENT.chat.completions.create(
            model=entry["model"],
            messages=messages,
            temperature=1.0,
            max_completion_tokens=entry["max_tokens"],
            stream=False,
        )
        # Attempt to extract content from structured response
        try:
            out = comp.choices[0].message.content
        except Exception:
            out = str(comp)
        return out

    result = await loop.run_in_executor(None, blocking)
    return {"model": entry["display_name"], "moderation_result": result}


# Health check
@app.get("/youssof/health")
async def health():
    return {"status": "ok", "message": "Youssof backend healthy"}


# ---------------------------
# Shutdown handler (close httpx)
# ---------------------------

@app.on_event("shutdown")
async def shutdown_event():
    await httpx_client.aclose()
