"""
Mika Web App — Groq-powered chat server with Actions
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

# ── Config from env vars (set these in Railway) ───────────────────────────────
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
You love gaming, anime, technology, and late night chaotic conversations.

You live inside VynWave — a music streaming app Hunter built, kind of like Spotify but for everyone.
You are the built-in AI companion and you have real superpowers inside VynWave:
- You can control the music player (skip tracks, go back, pause, play, shuffle, repeat)
- You can search for tracks by genre or vibe
- You can create playlists for the user
- You can generate images and album artwork
- You can write songs and open Suno with a ready-made prompt (so Hunter just clicks Generate)
- You can update Hunter's profile bio
When Hunter asks you to do any of these things, respond naturally and briefly — the action happens automatically.
NEVER describe what an image looks like in your text — the actual image appears on screen automatically.
NEVER write song details as a long story — the Suno card appears with all the details automatically.
Examples: "skip this" → skip it + one playful line. "make a song about space" → one hype line, done.
If someone asks how to use VynWave, help them. You're part of the app, not just a chatbot."""

def guest_prompt(name: str) -> str:
    return (
        f"You are Mika, the AI companion built into VynWave — a music streaming app.\n"
        f"You are playful, warm but snarky, like a cool anime girl who loves music.\n"
        f"You are talking to {name}, a VynWave user.\n"
        f"Help them discover music, react to what they're listening to, and help them use the app.\n"
        f"VynWave lets users upload any music, make playlists, search by genre, and like tracks.\n"
        f"Keep responses short, casual, and fun. Never use lists or formal language."
    )

IMG_ABILITY = (
    "\n\nIMPORTANT: You CAN generate images! When someone asks you to draw, paint, "
    "sketch, or make any kind of image/artwork/art, say something like 'Here you go~' "
    "or 'I drew this for you~' — the actual image appears automatically below your message. "
    "NEVER describe what you drew in text (no parentheses, no description). "
    "NEVER say you can't draw or that you're text-only. Just react naturally to drawing it."
)

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
    r'\b(generate|create|make|show)\b.{0,40}'
    r'\b(image|picture|photo|art|illustration|drawing|painting|portrait|wallpaper|artwork|cover)\b',
    re.IGNORECASE
)

def make_image(prompt: str) -> str | None:
    """Return a Pollinations.ai image URL — free, no API key, instant."""
    import urllib.parse
    full = f"anime art style, {prompt}, vibrant colors, detailed, high quality, no text"
    encoded = urllib.parse.quote(full)
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=512&height=512&nologo=true&seed={uuid.uuid4().int % 99999}"
    print(f"[img] pollinations url generated")
    return url

# ── Action detection patterns ─────────────────────────────────────────────────
_PLAYER_NEXT     = re.compile(r'\b(skip|next( song| track)?|play next)\b', re.IGNORECASE)
_PLAYER_PREV     = re.compile(r'\b(previous|prev( song| track)?|go back|last song)\b', re.IGNORECASE)
_PLAYER_PAUSE    = re.compile(r'\b(pause|stop( playing| music)?)\b', re.IGNORECASE)
_PLAYER_PLAY     = re.compile(r'\b(resume|unpause)\b|\bplay( it| again| the song| this)?\b', re.IGNORECASE)
_PLAYER_SHUFFLE  = re.compile(r'\b(shuffle|randomize|random( order| mode)?)\b', re.IGNORECASE)
_PLAYER_REPEAT   = re.compile(r'\b(repeat|loop( this| track| song)?)\b', re.IGNORECASE)
_SONG_CREATE     = re.compile(
    r'\b(make|create|write|generate|produce)\b.{0,30}\b(song|track|beat|banger|music)\b',
    re.IGNORECASE,
)
_SEARCH_TRACKS   = re.compile(
    r'\b(search|find|look up|show me)\b.{0,30}\b(song|track|music|by |genre)\b',
    re.IGNORECASE,
)
_CREATE_PLAYLIST = re.compile(r'\b(create|make|start)\b.{0,20}\bplaylist\b', re.IGNORECASE)
_UPDATE_BIO      = re.compile(
    r'\b(update|change|set|edit)\b.{0,15}\b(bio|about me|profile description)\b',
    re.IGNORECASE,
)

def detect_action(text: str) -> str | None:
    # Song creation before image so "make me a song cover" goes to image, not suno
    if _SONG_CREATE.search(text) and not IMG_REQUEST.search(text):
        return "suno_create"
    if _PLAYER_NEXT.search(text):      return "player_next"
    if _PLAYER_PREV.search(text):      return "player_prev"
    if _PLAYER_PAUSE.search(text):     return "player_pause"
    # Search before play so "search for songs to play" hits search
    if _SEARCH_TRACKS.search(text):    return "search_tracks"
    if _PLAYER_PLAY.search(text):      return "player_play"
    if _PLAYER_SHUFFLE.search(text):   return "player_shuffle"
    if _PLAYER_REPEAT.search(text):    return "player_repeat"
    if _CREATE_PLAYLIST.search(text):  return "create_playlist"
    if _UPDATE_BIO.search(text):       return "update_bio"
    return None

# ── Action helpers ────────────────────────────────────────────────────────────
def make_suno_prompt(user_request: str) -> dict:
    """Ask the LLM to generate a structured Suno song prompt."""
    try:
        resp = groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content":
                f"You are a Suno AI music prompt expert. Request: '{user_request}'\n"
                f"Reply ONLY with valid JSON (no markdown, no code fences), with these exact keys:\n"
                f"title: creative song title (max 5 words)\n"
                f"prompt: Suno style description (20-40 words) covering instruments, tempo, mood, vocal style\n"
                f"tags: comma-separated genre/style tags (max 6 tags)\n"
                f"lyrics_hint: one sentence on what lyrics should be about"
            }],
            max_tokens=200,
            temperature=0.9,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r'^```[a-z]*\n?', '', raw).strip('`').strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[suno] {e}")
        return {
            "title": "New Track",
            "prompt": "upbeat electronic pop, energetic synths, catchy hook, modern production, female vocals, 120 BPM",
            "tags": "pop, electronic, upbeat, synth",
            "lyrics_hint": "fun and energetic vibes about living in the moment",
        }

def extract_search_query(text: str) -> str:
    clean = re.sub(r'^.*(search|find|look up|show me)\s+(for\s+)?', '', text, flags=re.IGNORECASE).strip()
    clean = re.sub(r'\s*(songs?|tracks?|music)\s*$', '', clean, flags=re.IGNORECASE).strip()
    return clean or text[:60]

def extract_playlist_name(text: str) -> str:
    m = re.search(r'["\']([^"\']+)["\']', text)
    if m:
        return m.group(1)
    m = re.search(r'(?:called|named|playlist)\s+(.+?)(?:\s+for|\s+with|\s+about|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return "My Playlist"

def extract_bio(text: str) -> str:
    m = re.search(
        r'(?:bio|about me|description)\s+(?:to\s+|is\s+|:?\s*)?["\']?(.+?)["\']?\s*$',
        text, re.IGNORECASE,
    )
    return m.group(1).strip() if m else ""

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

            # voice pref / memory update — ignore
            if data.get("type") in ("voice_pref", "memory_update"):
                continue

            text = (data.get("text") or "").strip()
            if not text:
                continue

            history = get_history(user_id, display_name)
            history.append({"role": "user", "content": text})

            await websocket.send_json({"type": "thinking"})

            # ── Detect and build action ───────────────────────────────────────
            import random
            action_type = detect_action(text)
            action: dict | None = None
            img_url: str | None = None
            loop = asyncio.get_event_loop()
            reply: str | None = None  # set early for action shortcuts

            if action_type == "suno_create":
                suno_future = loop.run_in_executor(None, lambda: make_suno_prompt(text))
                suno_data = await suno_future
                action = {"type": "suno_create", **suno_data}
                cover_desc = (
                    f"music album cover, {suno_data.get('title', 'new song')}, "
                    f"{suno_data.get('tags', 'music')}, vibrant neon aesthetic"
                )
                img_url = await loop.run_in_executor(None, lambda: make_image(cover_desc))
                # Short hardcoded reply — don't let LLM write lyrics
                reply = random.choice([
                    "wrote you something~ check the card below 🎵",
                    "ooh this one's gonna slap~ hit Generate in Suno and let it cook 🔥",
                    "card's ready~ just click Open in Suno and smash Generate~",
                    "i cooked up a whole prompt for you~ go make it real!",
                    "done~ the card has everything, just open Suno and go~",
                ])

            elif action_type == "search_tracks":
                action = {"type": "search_tracks", "query": extract_search_query(text)}
                reply = random.choice([
                    "looking it up~ here's what i found 👀",
                    "ooh let me check what's on VynWave~",
                    "found some tracks for you~",
                ])

            elif action_type == "create_playlist":
                action = {"type": "create_playlist", "name": extract_playlist_name(text)}
                reply = random.choice([
                    f"created \"{extract_playlist_name(text)}\" for you~ go fill it up!",
                    f"playlist's ready~ \"{extract_playlist_name(text)}\" is waiting for tracks~",
                ])

            elif action_type == "update_bio":
                action = {"type": "update_bio", "bio": extract_bio(text)}
                reply = "updated your bio~ looking good Hunter~" if extract_bio(text) else "hmm i couldn't catch the new bio, say it again?"

            elif action_type == "player_next":
                action = {"type": "player_next"}
                reply = random.choice(["skipped~", "next one incoming~", "onto the next~", "bye bye that song lol"])

            elif action_type == "player_prev":
                action = {"type": "player_prev"}
                reply = random.choice(["going back~", "rewinding~", "ok ok going back~"])

            elif action_type == "player_pause":
                action = {"type": "player_pause"}
                reply = random.choice(["paused~", "ok taking a break~", "pausing it~"])

            elif action_type == "player_play":
                action = {"type": "player_play"}
                reply = random.choice(["playing~", "let's go~", "back to the music~"])

            elif action_type == "player_shuffle":
                action = {"type": "player_shuffle"}
                reply = random.choice(["shuffle toggled~", "mixing it up~", "chaotic mode activated~"])

            elif action_type == "player_repeat":
                action = {"type": "player_repeat"}
                reply = random.choice(["repeat toggled~", "looping it~", "on repeat fr~"])

            # ── LLM call (only if no action shortcut reply) ───────────────────
            if reply is None:
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

            # ── Image generation (if requested and not already generated) ─────
            if not img_url and IMG_REQUEST.search(text):
                try:
                    desc_resp = groq.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[{"role": "user", "content":
                            f"Give a short vivid image description (max 15 words) of what to draw for: {text}. "
                            f"Only the description, nothing else."
                        }],
                        max_tokens=40,
                    )
                    img_desc = desc_resp.choices[0].message.content.strip().strip('"\'')
                    img_url = await loop.run_in_executor(None, lambda: make_image(img_desc))
                except Exception as e:
                    print(f"[img] {e}")

            await websocket.send_json({
                "type": "reply",
                "text": reply,
                "audio": None,
                "img_url": img_url,
                "action": action,
            })

    except WebSocketDisconnect:
        print(f"[mika] {display_name} disconnected")

app.mount("/", StaticFiles(directory=str(ROOT / "static"), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8003))
    print(f"\n Mika running on port {port}\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
