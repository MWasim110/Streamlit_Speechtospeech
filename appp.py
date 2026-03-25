from __future__ import annotations
import os
import time
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class AppConfig:
    # ASR
    whisper_model: str
    whisper_language: str
    whisper_device: str
    whisper_compute_type: str
    whisper_cpu_threads: int

    # LLM
    groq_api_key: str
    groq_model_id: str
    system_prompt: str

    # TTS
    tts_engine: str          # "gtts" or "piper"
    gtts_lang: str
    piper_model_path: str
    piper_config_path: str
    piper_voices_dir: str    # NEW: folder that contains multiple .onnx voices


def load_config() -> AppConfig:
    def _get_int(name: str, default: int) -> int:
        raw = os.getenv(name, "").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    return AppConfig(
        whisper_model=os.getenv("WHISPER_MODEL", "base.en").strip(),
        whisper_language=os.getenv("WHISPER_LANGUAGE", "en").strip(),
        whisper_device=os.getenv("WHISPER_DEVICE", "cpu").strip(),
        whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip(),
        whisper_cpu_threads=_get_int("WHISPER_CPU_THREADS", 4),

        groq_api_key=os.getenv("GROQ_API_KEY", "").strip(),
        groq_model_id=os.getenv("GROQ_MODEL_ID", "llama-3.1-8b-instant").strip(),
        system_prompt=os.getenv(
            "SYSTEM_PROMPT",
            "You are a helpful voice assistant for students. Keep replies short and clear."
        ).strip(),

        tts_engine=os.getenv("TTS_ENGINE", "gtts").strip().lower(),
        gtts_lang=os.getenv("GTTS_LANG", "en").strip(),
        piper_model_path=os.getenv("PIPER_MODEL_PATH", "").strip(),
        piper_config_path=os.getenv("PIPER_CONFIG_PATH", "").strip(),
        piper_voices_dir=os.getenv("PIPER_VOICES_DIR", "").strip(),
    )


CFG = load_config()


# ─────────────────────────────────────────────
# HELPERS: discover Piper voice files
# ─────────────────────────────────────────────
def discover_piper_voices() -> Dict[str, Tuple[str, str]]:
    """
    Scan PIPER_VOICES_DIR for pairs of <name>.onnx + <name>.onnx.json.
    Returns  { display_name: (model_path, config_path) }
    Also adds the single-voice entry from env vars if set.
    """
    voices: Dict[str, Tuple[str, str]] = {}

    # single voice configured via env vars
    if CFG.piper_model_path and CFG.piper_config_path:
        name = os.path.splitext(os.path.basename(CFG.piper_model_path))[0]
        voices[f"{name} (default)"] = (CFG.piper_model_path, CFG.piper_config_path)

    # scan voices directory
    vdir = CFG.piper_voices_dir
    if vdir and os.path.isdir(vdir):
        for fname in sorted(os.listdir(vdir)):
            if fname.endswith(".onnx"):
                model_path  = os.path.join(vdir, fname)
                config_path = model_path + ".json"
                if os.path.exists(config_path):
                    display = fname.replace(".onnx", "").replace("_", " ").title()
                    voices[display] = (model_path, config_path)

    return voices


# ─────────────────────────────────────────────
# ASR – faster-whisper
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_whisper_model(model_size: str, device: str, compute_type: str, cpu_threads: int):
    from faster_whisper import WhisperModel
    return WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
    )


MIN_AUDIO_BYTES = 4_096   # bytes – anything smaller is treated as silence/noise

def transcribe_wav_bytes(wav_bytes: bytes) -> Tuple[str, float]:
    """
    Returns (transcript, elapsed_seconds).
    Raises ValueError for empty / too-short audio.
    """
    if len(wav_bytes) < MIN_AUDIO_BYTES:
        raise ValueError("Audio clip is too short or empty. Please record again.")

    model = get_whisper_model(
        CFG.whisper_model,
        CFG.whisper_device,
        CFG.whisper_compute_type,
        CFG.whisper_cpu_threads,
    )

    tmp_path = None
    t0 = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(wav_bytes)
            tmp_path = f.name

        segments, _info = model.transcribe(
            tmp_path,
            language=CFG.whisper_language,
            beam_size=1,
            vad_filter=True,
        )

        text = "".join(seg.text for seg in segments).strip()
        elapsed = time.perf_counter() - t0

        if not text:
            raise ValueError(
                "No speech detected. Try speaking louder or in a quieter environment."
            )

        return text, elapsed

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# ─────────────────────────────────────────────
# LLM – Groq
# ─────────────────────────────────────────────
SUPPORTED_LANGUAGES = {
    "English":  {"whisper": "en", "gtts": "en"},
    "Urdu":     {"whisper": "ur", "gtts": "ur"},
    "Arabic":   {"whisper": "ar", "gtts": "ar"},
    "French":   {"whisper": "fr", "gtts": "fr"},
    "Spanish":  {"whisper": "es", "gtts": "es"},
    "German":   {"whisper": "de", "gtts": "de"},
    "Hindi":    {"whisper": "hi", "gtts": "hi"},
    "Chinese":  {"whisper": "zh", "gtts": "zh-CN"},
}


def offline_demo_reply(user_text: str) -> str:
    if not user_text:
        return "I did not catch that. Please try again."
    return (
        "Offline demo mode – no AI model connected yet.\n\n"
        f"You said: {user_text}\n\n"
        "To enable real AI replies, add your GROQ_API_KEY in .env."
    )


def groq_chat_completion(messages: List[Dict[str, str]]) -> str:
    import requests

    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {CFG.groq_api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": CFG.groq_model_id,
        "messages": messages,
        "temperature": 0.4,
        "max_tokens": 250,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def generate_reply(
    user_text: str,
    history: List[Dict[str, str]],
    lang_name: str = "English",
) -> Tuple[str, float]:
    """Returns (reply_text, elapsed_seconds)."""
    t0 = time.perf_counter()

    if not CFG.groq_api_key:
        time.sleep(0.1)          # fake latency for demo
        return offline_demo_reply(user_text), time.perf_counter() - t0

    trimmed = history[-6:] if len(history) > 6 else history

    # Inject language instruction into system prompt
    lang_note = (
        f" Always reply in {lang_name}."
        if lang_name != "English"
        else ""
    )
    sys_prompt = CFG.system_prompt + lang_note

    messages = [{"role": "system", "content": sys_prompt}]
    messages.extend(trimmed)
    messages.append({"role": "user", "content": user_text})

    try:
        reply = groq_chat_completion(messages)
        return reply, time.perf_counter() - t0
    except Exception as e:
        return f"Could not reach Groq: {e}", time.perf_counter() - t0


# ─────────────────────────────────────────────
# TTS – gTTS (online) or Piper (offline)
# ─────────────────────────────────────────────
def tts_to_audio_file(
    text: str,
    gtts_lang_override: str = "",
    piper_voice: Optional[Tuple[str, str]] = None,
) -> Tuple[bytes, str, str, float]:
    """
    Returns (audio_bytes, mime_type, file_name, elapsed_seconds).
    piper_voice = (model_path, config_path) when user picks from dropdown.
    """
    text = (text or "").strip() or "I do not have a response to speak."

    t0 = time.perf_counter()

    if CFG.tts_engine == "piper":
        audio_bytes, mime, fname = piper_tts(text, piper_voice)
    else:
        lang = gtts_lang_override or CFG.gtts_lang
        audio_bytes, mime, fname = gtts_tts(text, lang)

    return audio_bytes, mime, fname, time.perf_counter() - t0


def gtts_tts(text: str, lang: str = "") -> Tuple[bytes, str, str]:
    from gtts import gTTS

    lang = lang or CFG.gtts_lang
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        out_path = f.name

    try:
        gTTS(text=text, lang=lang).save(out_path)
        with open(out_path, "rb") as rf:
            return rf.read(), "audio/mpeg", "reply.mp3"
    finally:
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass


def piper_tts(
    text: str,
    voice_override: Optional[Tuple[str, str]] = None,
) -> Tuple[bytes, str, str]:
    import wave
    from piper import PiperVoice

    model_path, config_path = voice_override or (
        CFG.piper_model_path,
        CFG.piper_config_path,
    )

    if not model_path or not config_path:
        return gtts_tts(
            "Piper is not configured. "
            "Please set PIPER_MODEL_PATH and PIPER_CONFIG_PATH in .env."
        )

    voice = PiperVoice.load(model_path, config_path)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        out_path = f.name

    try:
        with wave.open(out_path, "wb") as wav_file:
            voice.synthesize_wav(text, wav_file)
        with open(out_path, "rb") as rf:
            return rf.read(), "audio/wav", "reply.wav"
    finally:
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass


# ─────────────────────────────────────────────
# FULL PIPELINE (called automatically on record)
# ─────────────────────────────────────────────
def run_pipeline(
    wav_bytes: bytes,
    lang_name: str,
    piper_voice: Optional[Tuple[str, str]],
) -> None:
    """
    Runs ASR → LLM → TTS and appends a turn to st.session_state.chat_history.
    All errors are caught and stored as assistant error messages.
    """
    gtts_lang = SUPPORTED_LANGUAGES.get(lang_name, {}).get("gtts", CFG.gtts_lang)

    # ── 1. ASR ──────────────────────────────
    with st.spinner("🎙️ Transcribing…"):
        try:
            transcript, asr_t = transcribe_wav_bytes(wav_bytes)
        except ValueError as e:
            st.error(str(e))
            return
        except Exception as e:
            st.error(f"ASR error: {e}")
            return

    # ── 2. LLM ──────────────────────────────
    with st.spinner("🤖 Generating reply…"):
        reply_text, llm_t = generate_reply(
            transcript,
            st.session_state.chat_history,
            lang_name=lang_name,
        )

    # ── 3. TTS ──────────────────────────────
    with st.spinner("🔊 Synthesising audio…"):
        audio_bytes, mime, fname, tts_t = tts_to_audio_file(
            reply_text,
            gtts_lang_override=gtts_lang,
            piper_voice=piper_voice,
        )

    # ── store turn ───────────────────────────
    st.session_state.chat_history.append({"role": "user",      "content": transcript})
    st.session_state.chat_history.append({"role": "assistant", "content": reply_text})
    st.session_state.last_audio = (audio_bytes, mime, fname)
    st.session_state.last_latency = {
        "ASR":  round(asr_t, 2),
        "LLM":  round(llm_t, 2),
        "TTS":  round(tts_t, 2),
        "Total": round(asr_t + llm_t + tts_t, 2),
    }


# ─────────────────────────────────────────────
# STREAMLIT PAGE CONFIG & CUSTOM CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VoiceBridge AI",
    page_icon="🎙️",
    layout="centered",
)

st.markdown(
    """
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=Space+Grotesk:wght@600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* ── Background gradient ── */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* ── Header ── */
    .vb-header {
        text-align: center;
        padding: 2rem 0 1rem;
    }
    .vb-header h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2.4rem;
        background: linear-gradient(90deg, #a78bfa, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .vb-header p {
        color: #94a3b8;
        font-size: 0.95rem;
    }

    /* ── Chat bubbles ── */
    [data-testid="stChatMessage"] {
        border-radius: 1rem;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
    }

    /* ── Latency pill ── */
    .latency-pill {
        display: inline-flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 0.4rem;
    }
    .latency-pill span {
        background: rgba(167,139,250,0.15);
        border: 1px solid rgba(167,139,250,0.3);
        color: #c4b5fd;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 500;
    }
    .latency-pill span.total {
        background: rgba(56,189,248,0.15);
        border-color: rgba(56,189,248,0.35);
        color: #7dd3fc;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.85);
    }
    section[data-testid="stSidebar"] * {
        color: #cbd5e1 !important;
    }

    /* ── Divider ── */
    hr {
        border-color: rgba(100,100,160,0.3) !important;
    }

    /* ── Buttons ── */
    div.stButton > button {
        background: linear-gradient(135deg, #7c3aed, #2563eb);
        color: white;
        border: none;
        border-radius: 0.6rem;
        padding: 0.5rem 1.4rem;
        font-weight: 600;
        transition: opacity 0.2s;
    }
    div.stButton > button:hover {
        opacity: 0.85;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
for key, default in {
    "chat_history": [],
    "last_audio":   None,
    "last_latency": None,
    "prev_audio_id": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    # ── Language selector ───────────────────
    st.markdown("**🌐 Language**")
    lang_name = st.selectbox(
        "Select language",
        list(SUPPORTED_LANGUAGES.keys()),
        label_visibility="collapsed",
    )

    st.divider()

    # ── Piper voice selector (only if Piper mode & voices found) ────────
    available_voices = discover_piper_voices()
    selected_piper_voice: Optional[Tuple[str, str]] = None

    if CFG.tts_engine == "piper" and available_voices:
        st.markdown("**🎤 Piper Voice**")
        chosen_voice_name = st.selectbox(
            "Pick a voice",
            list(available_voices.keys()),
            label_visibility="collapsed",
        )
        selected_piper_voice = available_voices[chosen_voice_name]
    elif CFG.tts_engine == "piper":
        st.info("No Piper voices found.\nSet PIPER_VOICES_DIR in .env.")

    st.divider()

    # ── Config display ──────────────────────
    st.markdown("**📋 Active Config**")
    st.code(
        "\n".join([
            f"WHISPER_MODEL     = {CFG.whisper_model}",
            f"WHISPER_LANGUAGE  = {CFG.whisper_language}",
            f"WHISPER_DEVICE    = {CFG.whisper_device}",
            f"WHISPER_COMPUTE   = {CFG.whisper_compute_type}",
            f"WHISPER_THREADS   = {CFG.whisper_cpu_threads}",
            "",
            f"GROQ_API_KEY      = {'✅ SET' if CFG.groq_api_key else '❌ NOT SET'}",
            f"GROQ_MODEL        = {CFG.groq_model_id}",
            "",
            f"TTS_ENGINE        = {CFG.tts_engine}",
            f"GTTS_LANG         = {CFG.gtts_lang}",
            f"PIPER_MODEL_PATH  = {CFG.piper_model_path or '—'}",
            f"PIPER_VOICES_DIR  = {CFG.piper_voices_dir or '—'}",
        ]),
        language="ini",
    )
    st.caption("💡 Tip: Use `tiny.en` for low-RAM machines.")


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown(
    """
    <div class="vb-header">
        <h1>🎙️ VoiceBridge AI</h1>
        <p>Record → Transcribe → AI Reply → Speak &nbsp;|&nbsp; All in one flow</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Clear chat button ───────────────────────
col_l, col_r = st.columns([5, 1])
with col_r:
    if st.button("🗑️ Clear", help="Clear chat history"):
        st.session_state.chat_history = []
        st.session_state.last_audio   = None
        st.session_state.last_latency = None
        st.rerun()

# ── Render chat history ─────────────────────
for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"], avatar="🧑" if turn["role"] == "user" else "🤖"):
        st.markdown(turn["content"])

# ── Latency display (last turn) ────────────
if st.session_state.last_latency:
    lat = st.session_state.last_latency
    st.markdown(
        f"""
        <div class="latency-pill">
            <span>🎙️ ASR {lat['ASR']}s</span>
            <span>🤖 LLM {lat['LLM']}s</span>
            <span>🔊 TTS {lat['TTS']}s</span>
            <span class="total">⚡ Total {lat['Total']}s</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Last audio playback ─────────────────────
if st.session_state.last_audio:
    audio_bytes, mime, fname = st.session_state.last_audio
    st.audio(audio_bytes, format=mime)
    st.download_button(
        "⬇️ Download reply audio",
        data=audio_bytes,
        file_name=fname,
        mime=mime,
    )

st.divider()

# ── Audio recorder ──────────────────────────
st.markdown("**🎤 Record your message:**")
audio_value = st.audio_input("Speak now…", label_visibility="collapsed")

# ── AUTO-RUN: trigger pipeline as soon as a new recording arrives ──
if audio_value is not None:
    # Use object id as a proxy for "new recording"
    current_id = id(audio_value)
    if current_id != st.session_state.prev_audio_id:
        st.session_state.prev_audio_id = current_id
        wav_bytes = audio_value.getvalue()
        run_pipeline(
            wav_bytes,
            lang_name=lang_name,
            piper_voice=selected_piper_voice,
        )
        st.rerun()     # refresh to show updated chat bubbles


# ─────────────────────────────────────────────
# DEBUG SECTION
# ─────────────────────────────────────────────
with st.expander("🛠️ Debug – Chat History JSON"):
    st.json(st.session_state.chat_history)