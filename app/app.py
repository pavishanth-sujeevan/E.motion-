"""
E.motion³ — Flask Backend
=========================
Run:
    py -m pip install flask flask-cors gtts langdetect
    py app.py

Then open index.html in your browser.
Login: username = admin, password = 1234
"""

import os, uuid, re, io, random
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

# ── Optional libs (app still works without them) ──────────────────────────
try:
    from flask_cors import CORS
    HAS_CORS = True
except ImportError:
    HAS_CORS = False

try:
    from gtts import gTTS
    HAS_GTTS = True
except ImportError:
    HAS_GTTS = False

try:
    from langdetect import detect as _ld
    HAS_LANGDETECT = True
except ImportError:
    HAS_LANGDETECT = False

# ─────────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "emotion3_secret"

# ── CORS (allow browser to call Flask from file:// or localhost) ──────────
if HAS_CORS:
    CORS(app, origins="*")
else:
    @app.after_request
    def add_cors(response):
        response.headers["Access-Control-Allow-Origin"]  = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        return response

# ── Audio output folder ───────────────────────────────────────────────────
AUDIO_DIR = Path(__file__).parent / "static" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

EMOTIONS  = ["happy", "sad", "anxious", "angry", "disgust", "neutral"]
LANGUAGES = ["en", "ta", "si"]
GTTS_LANG = {"en": "en", "ta": "ta", "si": "si"}

# ══════════════════════════════════════════════════════════════════════════
# USERS  (add more here or replace with a database later)
# ══════════════════════════════════════════════════════════════════════════
USERS = {
    "admin": {
        "password": "1234",
        "name":     "Admin",
        "email":    "admin@emotion3.ai"
    }
}

# ══════════════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION
# ══════════════════════════════════════════════════════════════════════════
def detect_language(text):
    """Detect language from text. Returns 'en', 'ta', or 'si'."""
    if re.search(r"[\u0B80-\u0BFF]", text):  return "ta"   # Tamil
    if re.search(r"[\u0D80-\u0DFF]", text):  return "si"   # Sinhala
    if HAS_LANGDETECT:
        try:
            code = _ld(text)
            if code == "ta":              return "ta"
            if code in ("si", "sin"):     return "si"
        except Exception:
            pass
    return "en"

# ══════════════════════════════════════════════════════════════════════════
# TER — Text Emotion Recognition
# ══════════════════════════════════════════════════════════════════════════
# Keyword banks for each language
TER_KEYWORDS = {
    "en": {
        "happy":   ["happy","joy","joyful","excited","glad","great","wonderful","amazing",
                    "love","grateful","thankful","fantastic","ecstatic","cheerful","blessed",
                    "delighted","elated","content","pleased","thrilled"],
        "sad":     ["sad","cry","crying","tears","depressed","depression","hopeless","miserable",
                    "grief","lonely","alone","empty","heartbroken","lost","worthless","down",
                    "unhappy","sorrow","gloomy","devastated"],
        "anxious": ["anxious","anxiety","worried","worry","nervous","panic","stress","stressed",
                    "overwhelmed","fear","scared","terrified","dread","uneasy","tense","restless",
                    "apprehensive","afraid","phobia","frightened"],
        "angry":   ["angry","anger","furious","rage","hate","frustrated","frustration","mad",
                    "annoyed","irritated","outraged","bitter","livid","enraged","hostile",
                    "resentful","aggravated","infuriated"],
        "disgust": ["disgusted","disgust","gross","revolting","sick","nauseated","repulsed",
                    "appalled","repelled","horrible","vile","awful","repulsive","yuck",
                    "nasty","foul","hideous","offensive"],
        "neutral": []
    },
    "ta": {
        "happy":   ["மகிழ்ச்சி","சந்தோஷம்","அன்பு","நன்றி","நல்லது","அருமை","பேரானந்தம்","மகிழ்வு"],
        "sad":     ["சோகம்","அழுகை","தனிமை","வலி","கஷ்டம்","மனவேதனை","கண்ணீர்","துக்கம்","வருத்தம்"],
        "anxious": ["பயம்","கவலை","பதட்டம்","மன அழுத்தம்","பீதி","தயக்கம்","கலக்கம்","பதற்றம்"],
        "angry":   ["கோபம்","எரிச்சல்","ஆத்திரம்","வெறுப்பு","சினம்","ஆவேசம்"],
        "disgust": ["அருவருப்பு","வெறுக்கிறேன்","கேவலம்","இழிவு","கொடுமை"],
        "neutral": []
    },
    "si": {
        "happy":   ["සතුට","ප්‍රීතිය","සන්තෝෂ","ආදරය","ස්තූතිය","හොඳ","ප්‍රමෝදය"],
        "sad":     ["දුක","කඳුළු","හිස්","ශෝකය","දොම්නස","හුදකලා","විලාපය","දුකේ"],
        "anxious": ["බිය","කනස්සල්ල","ආතතිය","නොසන්සුන්","කලබල","බියසුලු"],
        "angry":   ["කෝපය","තරහ","රුෂ්ට","ක්‍රෝධය","ගිනි","කෝපපත"],
        "disgust": ["පිළිකුල","ජුගුප්සාව","අප්‍රිය","කළකිරීම"],
        "neutral": []
    }
}

def run_ter(text, lang="en"):
    """
    Text Emotion Recognition.
    ► TO USE YOUR OWN MODEL: replace this function body with your model inference.
    Must return: {"emotion": str, "confidence": float}
    """
    # ── Plug your TER model here ──────────────────────────────────────────
    # Example:
    # import joblib
    # model = joblib.load("models/ter_en.pkl")
    # label = model.predict([text])[0]
    # conf  = model.predict_proba([text])[0].max()
    # return {"emotion": label, "confidence": round(float(conf), 2)}
    # ─────────────────────────────────────────────────────────────────────

    bank  = TER_KEYWORDS.get(lang, TER_KEYWORDS["en"])
    lower = text.lower()
    scores = {}
    for emotion, keywords in bank.items():
        if emotion == "neutral":
            continue
        scores[emotion] = sum(1 for kw in keywords if kw in lower)

    if not scores or max(scores.values()) == 0:
        return {"emotion": "neutral", "confidence": 0.55}

    best  = max(scores, key=scores.get)
    total = sum(scores.values())
    conf  = round(scores[best] / total, 2) if total > 0 else 0.5
    return {"emotion": best, "confidence": conf}


# ══════════════════════════════════════════════════════════════════════════
# SER — Speech Emotion Recognition
# ══════════════════════════════════════════════════════════════════════════
def run_ser(audio_bytes, lang="en"):
    """
    Speech Emotion Recognition.
    ► TO USE YOUR OWN MODEL: replace this function body with your model inference.
    Must return: {"emotion": str, "confidence": float}
    """
    # ── Plug your SER model here ──────────────────────────────────────────
    # Example (PyTorch):
    # import torch, torchaudio
    # model  = torch.load("models/ser_en.pt", map_location="cpu"); model.eval()
    # labels = ["happy","sad","anxious","angry","disgust","neutral"]
    # wav, sr = torchaudio.load(io.BytesIO(audio_bytes))
    # with torch.no_grad(): logits = model(wav)
    # probs = torch.softmax(logits, -1)[0]
    # idx   = probs.argmax().item()
    # return {"emotion": labels[idx], "confidence": round(probs[idx].item(), 2)}
    # ─────────────────────────────────────────────────────────────────────

    # Placeholder: returns neutral until you plug in your model
    return {"emotion": "neutral", "confidence": 0.60}


# ══════════════════════════════════════════════════════════════════════════
# LLM — Emotion-Aware Therapy Response
# ══════════════════════════════════════════════════════════════════════════
LLM_RESPONSES = {
    "en": {
        "happy": [
            "That's wonderful to hear! 😊 It sounds like you're in a really positive space right now. What's been contributing to this happiness? Nurturing what brings us joy is an important part of emotional wellbeing.",
            "Your positive energy is beautiful! 🌟 Moments of happiness are worth savoring. Would you like to explore what you can do to carry this feeling forward, even during harder times?",
            "I love hearing that! 😊 Positive emotions build resilience. What's the one thing that's made you feel most alive recently?",
        ],
        "sad": [
            "I hear you, and I want you to know your feelings are completely valid. 💙 Sadness can feel very heavy. Can you tell me more about what's been weighing on you? You don't have to carry this alone.",
            "Thank you for trusting me with this. 🌧️ When sadness feels overwhelming, it helps to take things one small breath at a time. What do you think triggered this feeling today?",
            "Feeling sad is a natural human experience, though it doesn't make it less painful. 💙 Have you been able to talk to anyone close to you about how you're feeling?",
        ],
        "anxious": [
            "Anxiety can feel so overwhelming — like your mind won't slow down. 🌿 Let's try this together: breathe in slowly for **4 counts**, hold for **4**, and out for **4**. What's the main thing worrying you right now?",
            "I understand that anxious feeling. 💚 The **5-4-3-2-1 grounding technique** can help right now: name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste.",
            "Pre-event anxiety shows you care deeply. 🌟 Let's reframe: **what's the realistic worst case?** Often anxiety inflates risks far beyond reality. You are more prepared than you think.",
        ],
        "angry": [
            "I can feel the intensity in what you've shared. 🔥 Anger often signals that something important to you has been violated. What happened that sparked this feeling?",
            "Your anger is valid. 💪 Before anything else, it's okay to feel this. Anger often covers deeper feelings like hurt or fear. What do you think is underneath yours?",
            "Take a breath. 🍃 Anger carries important information about our values and needs. What specifically crossed a line for you?",
        ],
        "disgust": [
            "What you're feeling makes complete sense. 😔 Disgust is one of our most powerful protective emotions — it tells us when something conflicts with our values. Can you tell me more about what triggered this?",
            "That sounds really unsettling. 💙 Disgust often points to things that feel fundamentally wrong to us. What happened that brought up this feeling?",
            "Feeling repelled by something can be very distressing. 🌿 Your feelings are valid. What feels most wrong right now?",
        ],
        "neutral": [
            "Hello! I'm **E.motion³**, your emotion regulation assistant. 🌱 How are you feeling today? You can type or speak — I'm here to listen in any language.",
            "Thank you for sharing. 💛 I'm here to listen and support you. Would you like to explore how you're feeling a bit more deeply today?",
            "I'm here with you. 🌿 Sometimes it's hard to put feelings into words — that's perfectly okay. Take your time.",
        ],
    },
    "ta": {
        "happy": [
            "அது மிகவும் அழகான செய்தி! 😊 உங்கள் மகிழ்ச்சி என்னையும் மகிழ்விக்கிறது. இந்த நேர்மறையான உணர்வை நீடிக்க என்ன செய்கிறீர்கள்?",
            "உங்கள் மகிழ்ச்சி பொங்குகிறது! 🌟 என்ன உங்களுக்கு இந்த சந்தோஷம் தருகிறது என்று சொல்லுங்கள்.",
        ],
        "sad": [
            "உங்கள் வலியை புரிந்துகொள்கிறேன். 💙 நீங்கள் என்னிடம் பகிர்ந்துகொண்டதற்கு நன்றி. இன்று என்ன நடந்தது என்று சொல்ல முடியுமா?",
            "நீங்கள் தனியாக இல்லை. 🌧️ இந்த உணர்வுகள் நிஜமானவை. சுவாசிக்கவும் — ஒவ்வொரு கணமாக முன்னேறலாம்.",
        ],
        "anxious": [
            "கவலை மிகவும் சுமையாக இருக்கலாம். 🌿 **4 எண்ணிக்கை** வரை சுவாசிக்கவும், 4 நிமிடம் வைத்திருக்கவும், 4 எண்ணிக்கை வரை வெளியே விடவும். என்ன கவலைப்படுத்துகிறது?",
            "பதட்டம் உங்களை பாதுகாக்க முயற்சிக்கிறது. 💚 **5-4-3-2-1 முறை**: 5 பார்க்கக்கூடியவை, 4 தொடக்கூடியவை, 3 கேட்கக்கூடியவை, 2 நுகரக்கூடியவை, 1 சுவைக்கக்கூடியவை.",
        ],
        "angry": [
            "உங்கள் கோபம் புரிகிறது. 🔥 என்ன நடந்தது என்று என்னிடம் சொல்லுங்கள்.",
            "கோபம் செல்லுபடியாகும். 💪 அடியில் வலியோ, பயமோ இருக்கலாம். உங்கள் கோபத்தின் கீழே என்ன இருக்கிறது?",
        ],
        "disgust": [
            "அந்த உணர்வு மிகவும் புரிகிறது. 😔 அருவருப்பு நம் மதிப்புகளுக்கு எதிரான ஒன்று நடக்கும்போது வருகிறது. என்ன நடந்தது என்று சொல்ல முடியுமா?",
            "அது மிகவும் தொந்தரவான உணர்வு. 💙 இப்போது என்ன மிகவும் தவறாக தெரிகிறது?",
        ],
        "neutral": [
            "வணக்கம்! நான் **E.motion³**, உங்கள் உணர்வு நிர்வாக உதவியாளர். 🌱 இன்று நீங்கள் எப்படி உணர்கிறீர்கள்?",
            "நன்றி பகிர்ந்துகொண்டதற்கு. 💛 நீங்கள் என்ன உணர்கிறீர்கள் என்று மேலும் சொல்ல விரும்புகிறீர்களா?",
        ],
    },
    "si": {
        "happy": [
            "ඒක ඇසීමට සතුටයි! 😊 ඔබේ සතුට මටත් දැනෙනවා. මේ ධනාත්මක හැඟීම රැකගන්නේ කොහොමද?",
            "ඔබේ සතුට ඉතා ලස්සනයි! 🌟 ඔබට සතුට ගෙනෙන්නේ කුමක්ද?",
        ],
        "sad": [
            "ඔබේ වේදනාව මට තේරෙනවා. 💙 මා සමඟ බෙදා ගැනීමට ස්තූතියි. අද දවසේ සිදු වූ දේ ගැන කියන්නේ ද?",
            "ඔබ තනිව නැහැ. 🌧️ මේ හැඟීම් සැබෑ ඒවා. හුස්ම ගන්න — එකින් එක ඉදිරියට යන්නට පුළුවන්.",
        ],
        "anxious": [
            "කනස්සල්ල ඉතා අධිකව දැනෙනවා ඇති. 🌿 **4 ගණනකට** හුස්ම ගන්නවා, රඳවා ගන්නවා, පිට කරනවා. ඔබව වැඩිපුරම කරදර කරන්නේ කුමක්ද?",
            "ඔබව ආරක්ෂා කිරීමට කනස්සල්ල උත්සාහ කරනවා. 💚 **5-4-3-2-1**: දකින 5ක්, ස්පර්ශ 4ක්, ඇසෙන 3ක්, ගඳ 2ක්, රස 1ක්.",
        ],
        "angry": [
            "ඔබේ කෝපය මට තේරෙනවා. 🔥 සිදු වූ දේ මට කියන්නේ ද?",
            "ඔබේ කෝපය සාධාරණයි. 💪 යටින් ඇත්තේ කුමක්ද — වේදනාවද? බියද?",
        ],
        "disgust": [
            "ඒ හැඟීම ඉතා තේරෙනවා. 😔 පිළිකුල අපගේ වටිනාකම්වලට එරෙහිව යමක් සිදු වූ විට ඇති වෙනවා. කුමක් සිදු වූවාද?",
            "ඒ ඉතා කලකිරීමක් ඇති කරන හැඟීමකි. 💙 දැන් වැඩිම වරදක් ලෙස දැනෙන්නේ කුමක්ද?",
        ],
        "neutral": [
            "ආයුබෝවන්! මම **E.motion³**, ඔබේ හැඟීම් නියාමන සහායකයා. 🌱 අද ඔබ හැඟෙන්නේ කෙසේද?",
            "බෙදා ගැනීමට ස්තූතියි. 💛 ඔබේ හැඟීම් ගැන වැඩිදුරටත් කතා කිරීමට කැමතිද?",
        ],
    },
}

def run_llm(user_message, emotion, lang, history):
    """
    Emotion-aware therapy response generator.
    ► TO USE YOUR OWN MODEL: replace this function body with your model inference.
    Must return: str (the reply text)
    """
    # ── Plug your LLM here ────────────────────────────────────────────────
    # Example (Ollama):
    # import requests
    # r = requests.post("http://localhost:11434/api/chat", json={
    #     "model": "llama3",
    #     "messages": [
    #         {"role":"system","content":f"You are E.motion³. User feels {emotion}. Reply in {lang}."},
    #         *history[-10:],
    #         {"role":"user","content":user_message}
    #     ],
    #     "stream": False
    # })
    # return r.json()["message"]["content"]
    # ─────────────────────────────────────────────────────────────────────

    bank      = LLM_RESPONSES.get(lang, LLM_RESPONSES["en"])
    responses = bank.get(emotion, bank["neutral"])
    last      = next((m["content"] for m in reversed(history) if m["role"] == "assistant"), "")
    pool      = [r for r in responses if r != last] or responses
    return random.choice(pool)


# ══════════════════════════════════════════════════════════════════════════
# TTS — Text-to-Speech
# ══════════════════════════════════════════════════════════════════════════
def run_tts(text, lang="en"):
    """
    Text-to-Speech synthesis.
    ► TO USE YOUR OWN MODEL: replace this function body.
    Must return: filename (str) saved in AUDIO_DIR, or None to skip audio.
    """
    # ── Plug your TTS model here ──────────────────────────────────────────
    # Example (custom TTS):
    # from your_tts import synthesize
    # filename = f"{uuid.uuid4()}.wav"
    # synthesize(text, str(AUDIO_DIR / filename), language=lang)
    # return filename
    # ─────────────────────────────────────────────────────────────────────

    if not HAS_GTTS:
        return None
    try:
        # Strip markdown bold and emoji for cleaner speech
        clean = re.sub(r"\*+([^*]+)\*+", r"\1", text)
        clean = re.sub(r"[\U0001F300-\U0001FFFF\U00002600-\U000027BF]", "", clean).strip()
        if not clean:
            return None
        filename = f"{uuid.uuid4()}.mp3"
        gTTS(text=clean, lang=GTTS_LANG.get(lang, "en"), slow=False).save(
            str(AUDIO_DIR / filename)
        )
        return filename
    except Exception as e:
        print(f"[TTS error] {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════
# CONVERSATION STORE (in-memory)
# ══════════════════════════════════════════════════════════════════════════
_conversations = {}

def get_history(conv_id):
    return _conversations.setdefault(conv_id, [])

def push_history(conv_id, role, content):
    h = _conversations.setdefault(conv_id, [])
    h.append({"role": role, "content": content})
    if len(h) > 50:
        _conversations[conv_id] = h[-50:]


# ══════════════════════════════════════════════════════════════════════════
# ROUTES — AUTH
# ══════════════════════════════════════════════════════════════════════════
@app.route("/auth/login", methods=["POST", "OPTIONS"])
def login():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data     = request.get_json(force=True)
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    # Match by username or email
    matched_user = None
    matched_uid  = None
    if username in USERS and USERS[username]["password"] == password:
        matched_user = USERS[username]
        matched_uid  = username
    else:
        for uid, u in USERS.items():
            if u.get("email") == username and u["password"] == password:
                matched_user = u
                matched_uid  = uid
                break

    if not matched_user:
        return jsonify({"success": False, "error": "Invalid username or password"}), 401

    return jsonify({
        "success": True,
        "user": {
            "name":  matched_user["name"],
            "email": matched_user["email"],
            "uid":   matched_uid
        }
    })


@app.route("/auth/signup", methods=["POST", "OPTIONS"])
def signup():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data     = request.get_json(force=True)
    name     = data.get("name",     "").strip()
    email    = data.get("email",    "").strip().lower()
    password = data.get("password", "").strip()

    if not name or not email or not password:
        return jsonify({"success": False, "error": "All fields are required"}), 400

    # Check duplicate email
    for u in USERS.values():
        if u.get("email") == email:
            return jsonify({"success": False, "error": "Email already registered"}), 409

    uid = email.split("@")[0] + "_" + uuid.uuid4().hex[:4]
    USERS[uid] = {"password": password, "name": name, "email": email}

    return jsonify({
        "success": True,
        "user": {"name": name, "email": email, "uid": uid}
    })


# ══════════════════════════════════════════════════════════════════════════
# ROUTE — TEXT CHAT  →  TER → LLM  (no TTS)
# ══════════════════════════════════════════════════════════════════════════
@app.route("/chat", methods=["POST", "OPTIONS"])
def chat():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    data    = request.get_json(force=True)
    message = data.get("message", "").strip()
    conv_id = data.get("conv_id",  "default")

    if not message:
        return jsonify({"error": "message is required"}), 400

    # Step 1 — detect language
    lang = detect_language(message)

    # Step 2 — TER (text emotion recognition)
    ter = run_ter(message, lang)

    # Step 3 — LLM (generate therapy response)
    history = get_history(conv_id)
    reply   = run_llm(message, ter["emotion"], lang, history)

    # Step 4 — save to history
    push_history(conv_id, "user",      message)
    push_history(conv_id, "assistant", reply)

    return jsonify({
        "reply":      reply,
        "emotion":    ter["emotion"],
        "confidence": ter["confidence"],
        "lang":       lang,
        "audio_url":  None,
        "pipeline":   f"text → TER({lang}) → LLM"
    })


# ══════════════════════════════════════════════════════════════════════════
# ROUTE — VOICE CHAT  →  SER → LLM → TTS
# ══════════════════════════════════════════════════════════════════════════
@app.route("/speech", methods=["POST", "OPTIONS"])
def speech():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    if "audio" not in request.files:
        return jsonify({"error": "audio file is required"}), 400

    audio_bytes = request.files["audio"].read()
    transcript  = request.form.get("transcript", "").strip()
    conv_id     = request.form.get("conv_id",    "default")

    # Step 1 — detect language from transcript
    lang = detect_language(transcript) if transcript else "en"

    # Step 2 — SER (speech emotion recognition on raw audio)
    ser = run_ser(audio_bytes, lang)

    # Step 3 — LLM
    user_text = transcript or "[Voice message]"
    history   = get_history(conv_id)
    reply     = run_llm(user_text, ser["emotion"], lang, history)

    # Step 4 — save to history
    push_history(conv_id, "user",      user_text)
    push_history(conv_id, "assistant", reply)

    # Step 5 — TTS (generate audio reply)
    tts_file  = run_tts(reply, lang)
    audio_url = f"/audio/{tts_file}" if tts_file else None

    return jsonify({
        "transcript": user_text,
        "reply":      reply,
        "emotion":    ser["emotion"],
        "confidence": ser["confidence"],
        "lang":       lang,
        "audio_url":  audio_url,
        "pipeline":   f"voice → SER({lang}) → LLM → TTS"
    })


# ══════════════════════════════════════════════════════════════════════════
# ROUTE — SERVE AUDIO FILES
# ══════════════════════════════════════════════════════════════════════════
@app.route("/audio/<filename>")
def serve_audio(filename):
    return send_from_directory(str(AUDIO_DIR), filename)


# ══════════════════════════════════════════════════════════════════════════
# ROUTE — HEALTH CHECK
# ══════════════════════════════════════════════════════════════════════════
@app.route("/health")
def health():
    return jsonify({
        "status":     "ok",
        "app":        "E.motion³",
        "emotions":   EMOTIONS,
        "languages":  LANGUAGES,
        "gtts":       HAS_GTTS,
        "langdetect": HAS_LANGDETECT,
        "cors":       HAS_CORS,
        "users":      len(USERS)
    })


# ══════════════════════════════════════════════════════════════════════════
# START
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print()
    print("=" * 50)
    print("  E.motion³ Backend")
    print("  URL:      http://localhost:5000")
    print("  Health:   http://localhost:5000/health")
    print("  Login:    admin / 1234")
    print("=" * 50)
    print()
    print(f"  gtts installed:       {HAS_GTTS}")
    print(f"  langdetect installed: {HAS_LANGDETECT}")
    print(f"  flask-cors installed: {HAS_CORS}")
    print()
    app.run(debug=True, host="0.0.0.0", port=5000)
