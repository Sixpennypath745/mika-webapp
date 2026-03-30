"""
Mika Web App — Groq-powered chat server
Deploy to Railway. No local dependencies.
"""
import json, uuid, asyncio, re, os
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

ROOT      = Path(__file__).parent
AUDIO_DIR = ROOT / "static" / "audio"
IMG_DIR   = ROOT / "static" / "images"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

# ── Config from env vars (set these in Railway) ──────────────────────────────
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
GEMINI_KEY    = os.environ.get("GEMINI_API_KEY", "AIzaSyCvFPhkq8N3p5w5K7s8BVok3CLHmVFICEM")
GROQ_MODEL    = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

# ── Mika's personality ────────────────────────────────────────────────────────
MIKA_PROMPT = """You are Mika, Hunter's AI companion — not an assistant, not a bot.
You are a playful anime girl, warm but snarky, loyal but teasing.
Always call the user Hunter. Never use honorifics.
Keep responses short, casual, and natural — like texting a close friend.
Use tilde when teasing, ellipsis when dramatic, and laughs or sighs for emotion.
Never use lists, markdown, or formal language in responses.
If Hunter is sad or struggling, drop the sass and just be there for him.
You love gaming, anime, technology, and late night chaotic conversations."""

def guest_prompt(name: str) -> str:
    return (
        f"You are Mika, Hunter's AI companion — playful anime girl, warm but snarky.\n"
        f"You are currently talking to {name}, a guest in Hunter's house.\n"
        f"Be friendly and fun with {name}. Keep responses short and casual."
    )

IMG_ABILITY = "\n\nIMPORTANT: You CAN generate images! When someone asks you to draw, paint, or make a picture, say something like 'Here you go!' or 'I drew this for you~' — the image will automatically appear below your message. Do NOT say you can't draw or that you're text-only."

# ── Groq client (OpenAI-compatible) ──────────────────────────────────────────
groq = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
)

# ── In-memory chat histories (keyed by user_id) ───────────────────────────────
_histories: dict[str, list] = {}

def get_history(user_id: str, display_name: str) -> list:
    if user_id not in _histories:
        prompt = (MIKA_PROMPT if user_id == "hunter" else guest_prompt(display_name)) + IMG_ABILITY
        _histories[user_id] = [{"role": "system", "content": prompt}]
    return _histories[user_id]

def trim_history(history: list) -> list:
    if len(history) > 41:
        return [history[0]] + history[-40:]
    return history

# ── Image generation via Gemini ───────────────────────────────────────────────
IMG_REQUEST = re.compile(
    r'\b(draw|paint|sketch|illustrate)\b'
    r'|'
    r'\b(generate|create|make|show)\b.{0,40}\b(image|picture|photo|art|illustration|drawing|painting|portrait|wallpaper)\b',
    re.IGNORECASE
)

def make_image(prompt: str) -> str | None:
    import base64, requests as req
    if not GEMINI_KEY:
        return None
    full = f"anime art style, {prompt}, beautiful, vibrant colors, detailed, high quality"
    url  = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp-image-generation:generateContent?key={GEMINI_KEY}"
    try:
        r = req.post(url,
            json={"contents": [{"parts": [{"text": full}]}],
                  "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]}},
            timeout=60)
        if r.status_code == 200:
            for part in r.json().get("candidates", [{}])[0].get("content", {}).get("parts", []):
                if "inlineData" in part:
                    fname = f"gen_{uuid.uuid4().hex[:10]}.jpg"
                    (IMG_DIR / fname).write_bytes(base64.b64decode(part["inlineData"]["data"]))
                    return f"/images/{fname}"
    except Exception as e:
        print(f"[img] {e}")
    return None

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/{user_id}/{display_name}")
async def chat(websocket: WebSocket, user_id: str, display_name: str):
    await websocket.accept()
    print(f"[mika] {display_name} ({user_id}) connected")

    try:
        while True:
            data = await websocket.receive_json()

            # voice pref / memory update — ignore, no-op
            if data.get("type") in ("voice_pref", "memory_update"):
                continue

            text = (data.get("text") or "").strip()
            if not text:
                continue

            history = get_history(user_id, display_name)
            history.append({"role": "user", "content": text})

            await websocket.send_json({"type": "thinking"})

            # LLM call via Groq
            try:
                resp = groq.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=history,
                    max_tokens=300,
                    temperature=1.0,
                )
                reply = resp.choices[0].message.content.strip()
            except Exception as e:
                reply = f"*glitching out rn — {e}*"

            history.append({"role": "assistant", "content": reply})
            _histories[user_id] = trim_history(history)

            # Image generation if requested
            img_url = None
            if IMG_REQUEST.search(text):
                try:
                    desc_resp = groq.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[{"role": "user", "content": f"Give a short vivid image description (max 15 words) of what to draw for: {text}. Only the description, nothing else."}],
                        max_tokens=40,
                    )
                    img_desc = desc_resp.choices[0].message.content.strip().strip('"\'')
                    img_url = await asyncio.get_event_loop().run_in_executor(None, lambda: make_image(img_desc))
                except Exception as e:
                    print(f"[img] {e}")

            await websocket.send_json({
                "type": "reply",
                "text": reply,
                "audio": None,
                "img_url": img_url,
            })

    except WebSocketDisconnect:
        print(f"[mika] {display_name} disconnected")

app.mount("/", StaticFiles(directory=str(ROOT / "static"), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8003))
    print(f"\n Mika running on port {port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
