""" youssof_backend_fixed.py

Rewritten & fixed version of the user's FastAPI backend.

Changes & fixes applied:

More robust handling when Groq SDK is missing (server still starts; endpoints return clear errors).

Improved external HTTP calls to DarkAI (image, video, edit): explicit Content-Type header, safe JSON parsing and fallback to returning raw text with status codes.

Better logging for debugging (includes response text when errors occur).

Minor corrections to model listing (returns model id and model name correctly).

Ensured httpx AsyncClient is created with sensible defaults and closed at shutdown.

Defensive checks and clearer error messages across endpoints.

Preserved all original endpoints and functionality but hardened for reliability.


Run: GROQ_API_KEY="<your_key>" uvicorn youssof_backend_fixed:app --host 0.0.0.0 --port 8000

Note: This file assumes you have FastAPI, httpx, pydantic, and (optionally) the Groq SDK installed. """

import os import json import tempfile import asyncio import logging from pathlib import Path from typing import Optional, List, Any, Dict, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request from fastapi.responses import StreamingResponse, JSONResponse, FileResponse, PlainTextResponse from fastapi.middleware.cors import CORSMiddleware from pydantic import BaseModel import httpx

Setup logging

logging.basicConfig(level=logging.INFO) logger = logging.getLogger("youssof_backend")

---------------------------

Try import Groq SDK (optional)

---------------------------

GROQ_CLIENT = None GROQ_AVAILABLE = False try: from groq import Groq GROQ_AVAILABLE = True except Exception: logger.warning("groq SDK not available. Endpoints that require Groq will return helpful errors.\nInstall with: pip install groq")

---------------------------

Configuration & Constants

---------------------------

DEFAULT_GROQ_KEY = os.environ.get( "GROQ_API_KEY", "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", )

if GROQ_AVAILABLE: GROQ_CLIENT = Groq(api_key=DEFAULT_GROQ_KEY)

Friendly mapping of "Youssof" names to actual model IDs and a short description.

YOUSSOF_MODELS = { "youssof_chat": { "display_name": "Youssof Chat", "model": "llama-3.3-70b-versatile", "description": "Default general chat + multilingual assistant (Youssof chat v1).", "streamable": True, "max_tokens": 1024, }, "youssof_deep_reasoning": { "display_name": "Youssof Deep Reasoning", "model": "openai/gpt-oss-120b", "description": "Heavy-duty deep reasoning model (Youssof Reasoning v1).", "streamable": True, "max_tokens": 8192, }, "youssof_scout": { "display_name": "Youssof Scout", "model": "meta-llama/llama-4-scout-17b-16e-instruct", "description": "Tool / function-calling style assistant.", "streamable": True, "max_tokens": 1024, }, "youssof_maverick": { "display_name": "Youssof Maverick (Vision)", "model": "meta-llama/llama-4-maverick-17b-128e-instruct", "description": "Image understanding and vision-enabled assistant.", "streamable": True, "max_tokens": 1024, }, "youssof_guard": { "display_name": "Youssof Guard (Moderation)", "model": "meta-llama/llama-guard-4-12b", "description": "Safety and moderation checks.", "streamable": False, "max_tokens": 1024, }, "youssof_tts": { "display_name": "Youssof TTS (PlayAI)", "model": "playai-tts", "description": "Text to speech via YoussofAI TTS v1.", "streamable": False, "max_tokens": None, }, }

External DarkAI / sii3 endpoints

IMAGE_GEN_BASE = "https://sii3.moayman.top/api/flux-pro.php" VIDEO_BASE = "https://sii3.moayman.top/api/veo3.php" EDIT_BASE = "https://sii3.moayman.top/api/gpt-img.php"

HTTP client for external requests

HTTPX_TIMEOUT = 60  # seconds httpx_client = httpx.AsyncClient(timeout=HTTPX_TIMEOUT)

---------------------------

Pydantic models (requests)

---------------------------

class ChatMessage(BaseModel): role: str  # "user" or "assistant" or "system" content: Any  # string OR structured (for vision: may include list with image_url)

class ChatRequest(BaseModel): model_key: Optional[str] = "youssof_chat" messages: List[ChatMessage] temperature: Optional[float] = 1.0 max_completion_tokens: Optional[int] = None top_p: Optional[float] = 1.0 stream: Optional[bool] = True

class SimpleTextRequest(BaseModel): model_key: Optional[str] = "youssof_chat" text: str temperature: Optional[float] = 1.0 max_completion_tokens: Optional[int] = None stream: Optional[bool] = False

class ImageGenRequest(BaseModel): prompt: str n: Optional[int] = 1

class VideoGenRequest(BaseModel): prompt: str image_url: Optional[str] = None  # for image-to-video

class EditRequest(BaseModel): prompt: str image_url: str  # link to source image to edit

class TTSRequest(BaseModel): input: str model_key: Optional[str] = "youssof_tts" voice: Optional[str] = "Aaliyah-PlayAI" response_format: Optional[str] = "wav"

class ModerateRequest(BaseModel): text: str

---------------------------

FastAPI app

---------------------------

app = FastAPI( title="Youssof Multi-Model Gateway", description="One-stop backend to access Groq models and external image/video APIs.", version="1.0.1", )

Enable CORS (allow all origins). In production, restrict allow_origins to your domain(s).

app.add_middleware( CORSMiddleware, allow_origins=[""], allow_credentials=True, allow_methods=[""], allow_headers=["*"], )

---------------------------

Helper utilities

---------------------------

def get_model_entry(model_key: str): entry = YOUSSOF_MODELS.get(model_key) if not entry: raise HTTPException(status_code=400, detail=f"Unknown model_key: {model_key}") return entry

async def stream_groq_chat_response( model_id: str, messages: List[Dict[str, Any]], temperature: float = 1.0, max_completion_tokens: Optional[int] = None, top_p: float = 1.0, ) -> AsyncGenerator[bytes, None]: """ Streams Groq chat predictions as bytes (SSE compatible). """ if not GROQ_AVAILABLE: yield b"data: {"error": "Groq SDK not installed on server"}\n\n" return

loop = asyncio.get_running_loop()

def run_and_yield():
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
        logger.exception("Groq streaming creation failed")
        return [{"error": str(e)}]

    collected = []
    try:
        for chunk in completion:
            try:
                delta = getattr(chunk.choices[0], "delta", None)
                token = None
                if isinstance(delta, dict):
                    token = delta.get("content")
                else:
                    token = getattr(delta, "content", None)
            except Exception:
                token = None

            if token is None:
                token = getattr(chunk.choices[0], "message", None)
                if token:
                    token = getattr(token, "content", None)
            if token is None:
                token = str(chunk)
            collected.append({"chunk": token})
    except Exception as e:
        logger.exception("Error while iterating Groq stream")
        collected.append({"error": str(e)})
    return collected

chunks = await loop.run_in_executor(None, run_and_yield)
for obj in chunks:
    payload = json.dumps(obj, ensure_ascii=False)
    yield f"data: {payload}\n\n".encode("utf-8")
    await asyncio.sleep(0.01)

---------------------------

Endpoints

---------------------------

@app.get("/youssof/models") async def list_models(): """Return friendly model names and metadata, excluding hidden models.""" result = [] for key, entry in YOUSSOF_MODELS.items(): if entry.get("hidden"): continue result.append({ "model_key": key, "display_name": entry["display_name"], "description": entry["description"], "streamable": entry["streamable"], "model_id": entry["model"], })

console_note = {
    "console_url": "https://youssofai.onrender.com/",
    "note": "DONATE US TELEGRAM : t.me/youssofxmoussa",
}

return {"models": result, "console": console_note}

@app.post("/youssof/chat") async def youssof_chat(req: ChatRequest): """ Streamed chat endpoint for Youssof Chat (Llama 3.3 70B). """ if not GROQ_AVAILABLE: raise HTTPException(status_code=500, detail="Groq SDK not installed on server")

entry = get_model_entry(req.model_key or "youssof_chat")
model_id = entry["model"]
max_tokens = req.max_completion_tokens or entry.get("max_tokens")

messages = [{"role": m.role, "content": m.content} for m in req.messages]

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
    loop = asyncio.get_event_loop()

    def blocking_call():
        try:
            completion = GROQ_CLIENT.chat.completions.create(
                model=model_id,
                messages=messages,
                temperature=req.temperature,
                max_completion_tokens=max_tokens or 1024,
                top_p=req.top_p,
                stream=False,
                stop=None,
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.exception("Groq blocking call failed")
            return str(e)

    content = await loop.run_in_executor(None, blocking_call)
    return {"model": entry["display_name"], "response": content}

@app.post("/youssof/deep_reasoning") async def youssof_deep_reasoning(req: SimpleTextRequest): """ Entrypoint for the deep reasoning model (GPT OSS 120B). """ if not GROQ_AVAILABLE: raise HTTPException(status_code=500, detail="Groq SDK not installed on server")

entry = get_model_entry("youssof_deep_reasoning")
messages = [{"role": "user", "content": req.text}]
if req.stream:
    generator = stream_groq_chat_response(
        model_id=entry["model"],
        messages=messages,
        temperature=req.temperature,
        max_completion_tokens=req.max_completion_tokens or entry["max_tokens"],
    )
    return StreamingResponse(generator, media_type="text/event-stream")
else:
    loop = asyncio.get_event_loop()

    def blocking():
        try:
            comp = GROQ_CLIENT.chat.completions.create(
                model=entry["model"],
                messages=messages,
                temperature=req.temperature,
                max_completion_tokens=req.max_completion_tokens or entry["max_tokens"],
                stream=False,
            )
            return comp.choices[0].message.content
        except Exception as e:
            logger.exception("Groq deep_reasoning failed")
            return str(e)

    result = await loop.run_in_executor(None, blocking)
    return {"model": entry["display_name"], "response": result}

@app.post("/youssof/scout_tool_call") async def youssof_scout(req: ChatRequest): """ Tool-calling / function-use endpoint (Scout). """ if not GROQ_AVAILABLE: raise HTTPException(status_code=500, detail="Groq SDK not installed on server")

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
    loop = asyncio.get_event_loop()

    def blocking():
        try:
            comp = GROQ_CLIENT.chat.completions.create(
                model=entry["model"],
                messages=messages,
                temperature=req.temperature,
                max_completion_tokens=req.max_completion_tokens or entry["max_tokens"],
                stream=False,
            )
            return comp.choices[0].message.content
        except Exception as e:
            logger.exception("Groq scout call failed")
            return str(e)

    result = await loop.run_in_executor(None, blocking)
    return {"model": entry["display_name"], "response": result}

@app.post("/youssof/vision") async def youssof_vision(req: ChatRequest): """ Vision-capable endpoint (Maverick). Pass content list containing image_url as shown in examples. """ if not GROQ_AVAILABLE: raise HTTPException(status_code=500, detail="Groq SDK not installed on server")

entry = get_model_entry("youssof_maverick")
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
        try:
            comp = GROQ_CLIENT.chat.completions.create(
                model=entry["model"],
                messages=messages,
                temperature=req.temperature,
                max_completion_tokens=req.max_completion_tokens or entry["max_tokens"],
                stream=False,
            )
            return comp.choices[0].message.content
        except Exception as e:
            logger.exception("Groq vision call failed")
            return str(e)

    result = await loop.run_in_executor(None, blocking)
    return {"model": entry["display_name"], "response": result}

@app.post("/youssof/image_gen") async def youssof_image_gen(req: ImageGenRequest): """ Calls DarkAI flux-pro image-generation API. Example POST form: text=<prompt> """ params = {"text": req.prompt} try: # Post as form-urlencoded and include header explicitly r = await httpx_client.post(IMAGE_GEN_BASE, data=params, headers={"Content-Type": "application/x-www-form-urlencoded"}) r.raise_for_status() try: return JSONResponse(content=r.json()) except ValueError: # Not JSON — return text with status code logger.warning("Image gen returned non-JSON response: %s", r.text) return PlainTextResponse(content=r.text, status_code=r.status_code) except httpx.HTTPStatusError as e: logger.exception("Image service returned status error") raise HTTPException(status_code=502, detail=f"Image service status {e.response.status_code}: {e.response.text}") except Exception as e: logger.exception("Image service error") raise HTTPException(status_code=502, detail=f"Image service error: {str(e)}")

@app.post("/youssof/video_gen") async def youssof_video_gen(req: VideoGenRequest): """ Unified video endpoint. - If image_url provided -> image-to-video via VIDEO_BASE with params text & link - Otherwise -> text-to-video via VIDEO_BASE with params text """ params = {"text": req.prompt} if req.image_url: # the external API uses 'link' param for the image (per docs) params["link"] = req.image_url

try:
    r = await httpx_client.post(VIDEO_BASE, data=params, headers={"Content-Type": "application/x-www-form-urlencoded"})
    # Always check response status first
    r.raise_for_status()

    # Try JSON parse, fallback to text
    try:
        data = r.json()
        return JSONResponse(content=data)
    except ValueError:
        logger.info("Video service returned non-JSON response — returning raw text (status %s)", r.status_code)
        return PlainTextResponse(content=r.text, status_code=r.status_code)

except httpx.HTTPStatusError as e:
    # External service returned 4xx/5xx
    text = e.response.text if e.response is not None else str(e)
    logger.exception("Video service returned status error: %s", text)
    raise HTTPException(status_code=502, detail=f"Video service status {e.response.status_code}: {text}")
except Exception as e:
    logger.exception("Video service error")
    raise HTTPException(status_code=502, detail=f"Video service error: {str(e)}")

@app.post("/youssof/edit", tags=["youssof-edit"]) async def youssof_edit(req: EditRequest): """ YoussofEdit - edit an existing image via DarkAI gpt-img.php Example POST form: text=<prompt>&link=<image_url> """ params = {"text": req.prompt, "link": req.image_url} try: r = await httpx_client.post(EDIT_BASE, data=params, headers={"Content-Type": "application/x-www-form-urlencoded"}) r.raise_for_status() try: return JSONResponse(content=r.json()) except ValueError: logger.warning("Edit service returned non-JSON response: %s", r.text) return PlainTextResponse(content=r.text, status_code=r.status_code) except httpx.HTTPStatusError as e: logger.exception("Edit service returned status error") raise HTTPException(status_code=502, detail=f"Edit service status {e.response.status_code}: {e.response.text}") except Exception as e: logger.exception("Edit service error") raise HTTPException(status_code=502, detail=f"Edit service error: {str(e)}")

@app.post("/youssof/tts") async def youssof_tts(req: TTSRequest): """ Uses Groq audio.speech.create to generate TTS audio and returns a WAV file. """ if not GROQ_AVAILABLE: raise HTTPException(status_code=500, detail="Groq SDK not installed on server")

entry = get_model_entry(req.model_key or "youssof_tts")
tmp_dir = tempfile.gettempdir()
filename = Path(tmp_dir) / f"youssof_tts_{int(asyncio.get_event_loop().time() * 1000)}.wav"

def blocking_tts():
    try:
        response = GROQ_CLIENT.audio.speech.create(
            model=entry["model"],
            voice=req.voice,
            response_format=req.response_format,
            input=req.input,
        )
        # stream_to_file might be provided by the SDK; if not, try to save bytes
        try:
            response.stream_to_file(str(filename))
        except Exception:
            # fallback: if response has `read` or `content` attribute
            if hasattr(response, "read"):
                with open(filename, "wb") as fh:
                    fh.write(response.read())
            elif hasattr(response, "content"):
                with open(filename, "wb") as fh:
                    fh.write(response.content)
            else:
                raise RuntimeError("TTS response does not support stream_to_file or read/content")
        return str(filename)
    except Exception as e:
        logger.exception("TTS generation failed")
        raise RuntimeError(f"TTS generation failed: {e}")

loop = asyncio.get_event_loop()
try:
    filepath = await loop.run_in_executor(None, blocking_tts)
except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

return FileResponse(path=filepath, media_type="audio/wav", filename=Path(filepath).name)

@app.post("/youssof/moderate") async def youssof_moderate(req: ModerateRequest): """ Run a moderation check via Llama Guard. """ if not GROQ_AVAILABLE: raise HTTPException(status_code=500, detail="Groq SDK not installed on server")

entry = get_model_entry("youssof_guard")
messages = [{"role": "user", "content": req.text}]

loop = asyncio.get_event_loop()

def blocking():
    try:
        comp = GROQ_CLIENT.chat.completions.create(
            model=entry["model"],
            messages=messages,
            temperature=1.0,
            max_completion_tokens=entry["max_tokens"],
            stream=False,
        )
        return comp.choices[0].message.content
    except Exception as e:
        logger.exception("Moderation call failed")
        return str(e)

result = await loop.run_in_executor(None, blocking)
return {"model": entry["display_name"], "moderation_result": result}

Health check

@app.get("/youssof/health") async def health(): return {"status": "ok", "message": "Youssof backend healthy"}

---------------------------

Shutdown handler (close httpx)

---------------------------

@app.on_event("shutdown") async def shutdown_event(): try: await httpx_client.aclose() except Exception: logger.exception("Error closing httpx client")

---------------------------

Optional: quick root for convenience

---------------------------

@app.get("/") async def root(): return {"service": "youssof_backend_fixed", "status": "running"}

