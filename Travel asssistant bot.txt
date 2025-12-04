import os
import json
import time
import logging
import re
from copy import deepcopy

import requests
import spacy
from sentence_transformers import SentenceTransformer, util

# --- CONFIG ---
DEEPSEEK_API_KEY = os.environ.get(
    "DEEPSEEK_API_KEY",
    "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"
)
DEEPRTR_URL = "https://openrouter.ai/api/v1/chat/completions"
DS_MODEL = "deepseek/deepseek-chat-v3-0324:free"

# --- Logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")



# --- Models ---
nlp = spacy.load("en_core_web_sm")
sbert_model = SentenceTransformer("all-MiniLM-L6-v2")

# --- Intents & fields ---
INTENTS = {
    "flight_ticket": {
        "name": None,
        "budget": None,
        "travellers": {
            "adults": None,
            "children": None,
            "infants": None,
            "total_travellers": None
        },
        "destination_source": None,
        "destination_ending": None,
        "travel_dates": None,
        "preferences": None,
        "hotel_type": None
    },
    "umrah_package": {
        "total_participants": None,
        "budget": None,
        "travel_dates": None,
        "umrah_duration": None
    },
    "hotel_booking": {
        "number_of_people": None,
        "budget": None,
        "booking_dates": None,
        "stay_duration": None,
        "rooms_required": None
    }
}

# --- Keywords ---
INTENT_KEYWORDS = {
    "flight_ticket": {
        "strong": {"flight", "airline", "ticket", "boarding", "airport", "economy", "business", "fly", "plane"},
        "weak": {"travel", "trip", "journey"}
    },
    "umrah_package": {
        "strong": {"umrah", "pilgrimage", "makkah", "madinah", "mecca", "medina", "ziyarat"},
        "weak": {"holy", "religious", "visit", "travel"}
    },
    "hotel_booking": {
        "strong": {"hotel", "stay", "room", "reservation", "accommodation", "lodging", "suite"},
        "weak": {"apartment", "house", "bnb", "hostel"}
    }
}

# --- Semantic examples ---
INTENT_EXAMPLES = {
    "flight_ticket": [
        "I want to book a flight",
        "Can you help me find airline tickets?",
        "I need to fly to another city",
        "I want to fly",
        "Book a plane ticket for me",
        "I want to travel by air",
        "Find me a flight to New York",
        "I need airline reservations"
    ],
    "hotel_booking": [
        "I want to book a hotel",
        "Find me accommodation",
        "I need a room to stay",
        "Reserve a hotel for me",
        "Book a suite in a hotel",
        "I need lodging for my trip",
        "I want a place to stay"
    ],
    "umrah_package": [
        "I want to go for Umrah",
        "Help me with pilgrimage plans",
        "I want a Umrah travel package",
        "Book an Umrah trip",
        "I need guidance for Umrah",
        "Arrange my Umrah journey"
    ]
}

# ---------- Precompute semantic embeddings ----------
EXAMPLE_EMB = {intent: sbert_model.encode(examples, convert_to_tensor=True)
               for intent, examples in INTENT_EXAMPLES.items()}


# ---------- Utilities ----------
def flatten_fields(fields):
    flat = {}
    for k, v in fields.items():
        if isinstance(v, dict):
            for sub_k, sub_v in flatten_fields(v).items():
                flat[f"{k}.{sub_k}"] = sub_v
        else:
            flat[k] = v
    return flat


def all_fields_filled(fields):
    for k, v in fields.items():
        if isinstance(v, dict):
            if not all_fields_filled(v):
                return False
        elif v is None:
            return False
    return True


def deep_merge(target, source):
    for k, v in source.items():
        if k in target and isinstance(target[k], dict) and isinstance(v, dict):
            deep_merge(target[k], v)
        else:
            target[k] = v


def extract_name_local(text):
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            return ent.text
    m = re.search(r"\b(?:my name is|i am|i'm|name is)\s+([A-Za-z]{2,30})\b", text, re.I)
    if m:
        return m.group(1)
    tokens = [t for t in text.strip().split() if re.match(r"^[A-Za-z\-']+$", t)]
    if len(tokens) == 1 and 1 <= len(tokens[0]) <= 30:
        return tokens[0]
    return None


# ---------- Intent Detection ----------
def detect_intent_keywords(text):
    text_lower = text.lower()
    doc = nlp(text_lower)
    lemmas = {token.lemma_ for token in doc if token.is_alpha}
    detected = {}
    for intent, kws in INTENT_KEYWORDS.items():
        strong_hits = lemmas & kws["strong"]
        weak_hits = lemmas & kws["weak"]
        if strong_hits:
            detected[intent] = 95
        elif weak_hits:
            detected[intent] = 70
    if not detected:
        logging.info("Keyword detection found no matches.")
        return None, 0
    best_intent = max(detected, key=detected.get)
    logging.info(f"Keyword detection: matched intent={best_intent} score={detected[best_intent]}")
    return best_intent, detected[best_intent]


def detect_intent_semantic(text):
    emb_input = sbert_model.encode(text, convert_to_tensor=True)
    best_intent = None
    best_score = 0.0
    for intent, emb_examples in EXAMPLE_EMB.items():
        scores = util.pytorch_cos_sim(emb_input, emb_examples)
        max_score = float(scores.max().item())
        if max_score > best_score:
            best_score = max_score
            best_intent = intent
    logging.info(f"Semantic similarity best_intent={best_intent} score={best_score:.4f}")
    return best_intent, best_score


def detect_intent_with_deepseek(user_input):
    prompt = f"""
Classify the user's message into one of: flight_ticket, umrah_package, hotel_booking
User: "{user_input}"
Return only the intent name.
"""
    content = ask_smart_bot([{"role": "system", "content": prompt}], timeout=20, retries=2)
    if not content:
        logging.error("DeepSeek intent classification returned nothing.")
        return None
    for intent in INTENTS:
        if intent in content:
            logging.info("DeepSeek intent classification returned: %s", intent)
            return intent
    logging.warning("DeepSeek returned unexpected intent text: %s", content[:200])
    return None


# ---------- DeepSeek helpers ----------
def ask_smart_bot(messages, timeout=20, retries=2):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "AlexAssistant"
    }
    payload = {"model": DS_MODEL, "messages": messages}
    for attempt in range(retries + 1):
        try:
            resp = requests.post(DEEPRTR_URL, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            logging.error("DeepSeek HTTP %s - %s", resp.status_code, resp.text[:200])
        except Exception as e:
            logging.error("DeepSeek request failed (attempt %d/%d): %s", attempt + 1, retries + 1, e)
            time.sleep(1 + attempt * 1.5)
    return None


def deepseek_extract_fields(user_text, intent, timeout=20, retries=2):
    if intent not in INTENTS:
        raise ValueError("Unknown intent for DeepSeek extraction: %s" % intent)
    fields_template = INTENTS[intent]
    prompt = f"""
Extract all relevant fields for intent '{intent}' from the user's text.
Return JSON only (no extra text). Fill missing fields with null.
User text: \"\"\"{user_text}\"\"\"
Fields to extract: {json.dumps(fields_template, indent=2)}
"""
    content = ask_smart_bot([{"role": "user", "content": prompt}], timeout=timeout, retries=retries)
    if not content:
        return deepcopy({k: (deepcopy(v) if isinstance(v, dict) else None) for k, v in fields_template.items()})
    try:
        start = content.find("{")
        end = content.rfind("}") + 1
        json_blob = content[start:end]
        data = json.loads(json_blob)
        if "travellers" in data and isinstance(data["travellers"], dict):
            t = data["travellers"]
            try:
                a = int(t.get("adults") or 0)
                c = int(t.get("children") or 0)
                i = int(t.get("infants") or 0)
                total = a + c + i
                if total > 0 and not t.get("total_travellers"):
                    t["total_travellers"] = total
            except Exception:
                pass
        logging.info("DeepSeek extracted fields successfully.")
        return data
    except Exception as e:
        logging.error("Failed to parse DeepSeek JSON: %s", e)
        return deepcopy({k: (deepcopy(v) if isinstance(v, dict) else None) for k, v in fields_template.items()})


# ---------- State manager ----------
class StateManager:
    def __init__(self, intent_name):
        self.intent = intent_name
        self.fields = deepcopy(INTENTS[intent_name])
        self.history = []

    def update_with(self, new_fields):
        deep_merge(self.fields, new_fields)
        snapshot = deepcopy(self.fields)
        self.history.append({"time": time.time(), "snapshot": snapshot})

    def missing_flat(self):
        flat = flatten_fields(self.fields)
        return [k for k, v in flat.items() if v is None]

    def show(self):
        print(json.dumps(self.fields, indent=2))


# ---------- Prompt generator ----------
def generate_prompt_for_next_question(memory, intent):
    flat = flatten_fields(memory)
    missing = [k for k, v in flat.items() if not v]
    known = {k: v for k, v in flat.items() if v}
    missing_text = ", ".join(missing) if missing else "(none)"
    known_text = json.dumps(known, indent=2) if known else "(none)"
    return f"""
You're Alex, a friendly travel assistant.
The user just said: "{known_text}"
Current known information: {known_text}
Missing info: {missing_text}

Generate a friendly, natural response that:
1. Comments on the last input naturally
2. Asks only one missing field next
3. Keep it human-like, casual, friendly
Return only text to show to the user.
"""


# ---------- Main loop ----------
def main():
    print("👋 Hey! I'm Alex, your friendly travel assistant ✈️")
    print("Say 'quit' to exit.\n")

    while True:
        raw = input("🧍 You: ").strip()
        if not raw:
            continue
        if raw.lower() in ("quit", "exit"):
            print("👋 Bye!")
            break

        # --- Keyword detection ---
        intent, conf = detect_intent_keywords(raw)
        used_method = None
        if intent:
            used_method = "Keyword"

        # --- Semantic fallback if weak keyword ---
        if not intent or conf < 95:  # run semantic if only weak words or no match
            sem_intent, sem_score = detect_intent_semantic(raw)
            if sem_score >= 0.6:
                intent = sem_intent
                conf = int(sem_score * 100)
                used_method = "Semantic"
            else:
                ds_intent = detect_intent_with_deepseek(raw)
                if ds_intent:
                    intent = ds_intent
                    conf = 100
                    used_method = "DeepSeek(intent)"
                else:
                    print("❌ I couldn't detect your intent — could you rephrase that?")
                    continue

        if not intent or intent not in INTENTS:
            print("❌ I don't understand, please try again.")
            continue

        print(f"✅ Detected intent: {intent.replace('_',' ')} (via {used_method}, confidence ~= {conf})")

        state = StateManager(intent)
        extracted = deepseek_extract_fields(raw, intent)
        if extracted and not extracted.get("name"):
            name_local = extract_name_local(raw)
            if name_local:
                extracted["name"] = name_local
        state.update_with(extracted)

        # --- Multi-turn: collect missing fields ---
        max_turns = 20
        turn = 0
        while turn < max_turns and not all_fields_filled(state.fields):
            turn += 1
            missing = state.missing_flat()
            if not missing:
                break
            next_slot = missing[0]
            prompt_text = generate_prompt_for_next_question(state.fields, intent)
            phrasing = ask_smart_bot([{"role": "user", "content": prompt_text}], timeout=10, retries=1)
            question = phrasing.strip().splitlines()[-1] if phrasing else f"Could you provide your {next_slot.replace('_',' ')}?"
            print(f"\n🤖 Alex: {question}")
            reply = input("🧍 You: ").strip()
            if not reply:
                print("⚠️ Didn't catch that, please provide it.")
                continue
            new_fields = deepseek_extract_fields(reply, intent)
            if new_fields and not new_fields.get("name"):
                name_local = extract_name_local(reply)
                if name_local:
                    new_fields["name"] = name_local
            state.update_with(new_fields)
            print("📝 Current collected info:")
            state.show()

        if all_fields_filled(state.fields):
            print("\n✅ All required fields collected!")
            state.show()
        else:
            print("\n⚠️ Stopped collecting (max turns or incomplete). Showing current snapshot:")
            state.show()

        break  # stop after one successful run


if __name__ == "__main__":
    main()
