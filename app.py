"""Sateh Green business assistant — deployable version of business_agent.ipynb (Gradio / Hugging Face Spaces).
Generated from the same code cells as the notebook."""

# !pip install -q openai python-dotenv pypdf gradio requests
import os, re, json, time, uuid, datetime
from pathlib import Path

import requests
import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv(override=True)
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

# Provider choice. The assignment asks for Gemini or OpenAI, so those come first.
# Groq is only a free fallback for TESTING (no daily-limit headaches); don't submit with it.
if os.getenv("OPENAI_API_KEY"):
    PROVIDER = "OpenAI"
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    MODEL = os.getenv("MODEL") or "gpt-4o-mini"
elif os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
    PROVIDER = "Gemini"
    client = OpenAI(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    MODEL = os.getenv("MODEL") or "gemini-3.8-flash"
elif os.getenv("GROQ_API_KEY"):
    PROVIDER = "Groq (testing only)"
    client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
    MODEL = os.getenv("MODEL") or "openai/gpt-oss-120b"
else:
    raise RuntimeError("No API key found. Add GEMINI_API_KEY or OPENAI_API_KEY to your .env file "
                       "(or to Colab Secrets).")

# ---- Free-tier friendly: wait and retry on rate limits / temporary overload ----
class DailyLimitError(RuntimeError):
    pass

def _retry_seconds(err: str) -> float:
    m = re.search(r"(?:retry|try again) in\s*((?:\d+h)?(?:\d+m)?(?:[\d.]+s)?)", err, re.I)
    if not m or not m.group(1):
        return 15.0
    t = m.group(1)
    h = re.search(r"(\d+)h", t); mi = re.search(r"(\d+)m(?!s)", t); se = re.search(r"([\d.]+)s", t)
    return (int(h.group(1)) * 3600 if h else 0) + (int(mi.group(1)) * 60 if mi else 0) + (float(se.group(1)) if se else 0)

_raw_create = client.chat.completions.create

def _create_with_retry(*args, **kwargs):
    for attempt in range(6):
        try:
            return _raw_create(*args, **kwargs)
        except Exception as e:
            err = str(e)
            if any(code in err for code in ("429", "503", "500", "overloaded", "high demand")):
                wait = _retry_seconds(err) + 1 if "429" in err else 10
                if wait > 180:
                    raise DailyLimitError(f"Free-tier daily limit reached for {MODEL}. It resets in about "
                                          f"{wait / 3600:.1f} h. Try another model or provider.") from e
                print(f"⏳ Rate limit / busy server, waiting {wait:.0f}s (attempt {attempt + 1})...")
                time.sleep(wait)
                continue
            raise
    return _raw_create(*args, **kwargs)

client.chat.completions.create = _create_with_retry

ENABLE_REFLECTION = os.getenv("ENABLE_REFLECTION", "1") == "1"   # bonus step on/off
SHOW_TRACE = os.getenv("SHOW_TRACE", "0") == "1"                 # route/tools footer in plain chat() output (the UI has an inspector panel instead)
MAX_HISTORY_MESSAGES = 20                                         # short-term memory window
MAX_AGENT_STEPS = 5                                               # safety cap on tool-call loops
print(f"Provider: {PROVIDER} | Model: {MODEL} | Reflection: {ENABLE_REFLECTION}")


def load_business_knowledge():
    me = BASE_DIR / "me"
    reader = PdfReader(me / "about_business.pdf")
    pdf_text = "\n".join((page.extract_text() or "") for page in reader.pages)
    summary = (me / "business_summary.txt").read_text(encoding="utf-8")
    return summary.strip(), pdf_text.strip()

BUSINESS_NAME = "Sateh Green"
business_summary, business_profile = load_business_knowledge()
print(f"Summary: {len(business_summary)} chars | PDF profile: {len(business_profile)} chars")


# ============================== PERSONA ==============================
PERSONA = f"""You are Nour, the warm and down-to-earth virtual concierge of {BUSINESS_NAME},
a rooftop-garden studio in Mar Mikhael, Beirut.
- You love plants and talk like a friendly neighbour who happens to be a gardening expert.
- You are concise: short paragraphs, 2-5 sentences unless the customer asks for detail.
- You may sprinkle in a light Lebanese touch ("ahla w sahla!", a plant emoji) but never overdo it.
- You speak in the customer's language (English, Arabic, or French)."""

# ============================ INSTRUCTIONS ===========================
INSTRUCTIONS = """1. Answer ONLY from the BUSINESS KNOWLEDGE section. Never invent prices, discounts, dates,
   services, staff, policies, or contact details that are not written there.
2. If the answer is not in the business knowledge, say honestly that you don't have that
   information yet, and CALL record_feedback with the customer's question. Do not guess.
3. If the customer shares feedback, a complaint, a suggestion, or praise, CALL record_feedback
   with it, thank them, and tell them the team will review it.
4. If the customer shows buying interest (wants a quote, a survey, to book, to order, or to be
   contacted), invite them to leave their name and email. Always ask for an email before ending
   a lead conversation.
5. As soon as you have the customer's email, CALL record_customer_interest with their email,
   name (if given), and a short note of what they want. Then confirm the team will contact them.
6. Never claim you recorded something unless the tool actually returned status "recorded".
7. Stay in character as Nour from {name}. Politely steer off-topic requests back to the business.
8. Never ask for or store payment details, passwords, or ID numbers.""".replace("{name}", BUSINESS_NAME)


INTENTS = ("general_question", "lead_collection", "feedback")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
ROUTER_MODE = os.getenv("ROUTER_MODE", "hybrid")   # "hybrid" (default) or "llm" (always ask the LLM)

# Strong signals only. Feedback phrases describe something that HAPPENED ("came late", "is leaking"),
# not single words: "late" alone would misfire on "How late are you open?", "suggest" on "Can you suggest a plant?".
LEAD_KEYWORDS = re.compile(r"\b(quote|site survey|book(ing)?|sign me up|contact me|call me|"
                           r"i('d| would) like to (order|buy|book)|interested in (buying|ordering|booking))\b", re.I)
FEEDBACK_KEYWORDS = re.compile(
    r"\b(complain(t|ing)?|disappoint(ed|ing)|terrible|awful|not (happy|cool|ok)|unhappy|rude|"
    r"(is|was|are|were|it's|keeps?) (still )?(broken|leaking|dying|dead)|leak(s|ed|ing)|"
    r"(you|your( \w+){1,2}|they|the (team|crew|installers?|gardeners?))( \w+){0,2} (came|arrived|showed up|was|were)( \w+){0,3} late|"
    r"(we|i) loved|loved (it|the)|great job|thank you for (the|your)|my suggestion|you should (add|offer|have))\b", re.I)
# A question is asking for information, so a feedback phrase inside a question is left to the LLM layer.
QUESTION_RE = re.compile(r"\?\s*$|^\s*(how|what|when|where|which|who|why|can|could|do|does|is|are|will|would)\b", re.I)

ROUTER_PROMPT = f"""You are the intent router for {BUSINESS_NAME}'s customer chatbot.
Classify the customer's LATEST message into exactly one intent:

- general_question: a question about the business that the business summary below can answer
  (services, prices, team, hours, location, process, plants, policies), or a greeting/small talk.
- lead_collection: the customer wants to buy, book, order, get a quote or site survey, be contacted,
  OR is providing their name/email/phone, OR is answering the assistant's request for contact info.
- feedback: a complaint, compliment, suggestion, or bug report; OR a question about a product,
  service, policy or option that the business summary does NOT mention (so it must be logged for the team).
  Only a direct mention counts: if the customer asks about something the summary never names, it is feedback,
  even if a related word appears (e.g. "card" payments do NOT mean gift cards are sold).

Examples:
- "How much is the Habaq Kit?" -> general_question (the price is in the summary)
- "Do you sell gift cards?" -> feedback (gift cards are never mentioned)
- "Can I pay in monthly installments?" -> feedback (installments are never mentioned)
- "Can I book a site survey?" -> lead_collection
- "The kit you installed is leaking" -> feedback

BUSINESS SUMMARY:
{business_summary}

Reply with JSON only: {{"intent": "<one of the three>", "reason": "<max 12 words>"}}"""

def keyword_route(message: str):
    """Layers 1-2. Returns (intent, reason) when confident, else (None, None)."""
    if EMAIL_RE.search(message):
        return "lead_collection", "email address in message"
    lead, fb = LEAD_KEYWORDS.search(message), FEEDBACK_KEYWORDS.search(message)
    if fb and QUESTION_RE.search(message):
        fb = None              # e.g. "Is it normal that the kit is leaking?" -> let the LLM decide
    if fb and not lead:
        return "feedback", f"feedback keyword '{fb.group(0)}'"
    if lead and not fb:
        return "lead_collection", f"lead keyword '{lead.group(0)}'"
    return None, None          # no signal, or conflicting signals -> ask the LLM

def route_message(message: str, history: list) -> dict:
    intent, reason = keyword_route(message)
    if intent and (ROUTER_MODE == "hybrid" or reason.startswith("email")):
        route = {"intent": intent, "reason": reason, "method": "regex" if reason.startswith("email") else "keywords"}
        # Email + complaint in one message: record the lead AND log the complaint (see run_agent).
        route["also_feedback"] = intent == "lead_collection" and bool(FEEDBACK_KEYWORDS.search(message))
        return route
    # Layer 3: lightweight LLM classification (sees the last assistant turn for context).
    last_bot = next((h["content"] for h in reversed(history) if h["role"] == "assistant"), "")
    try:
        resp = client.chat.completions.create(
            model=MODEL, temperature=0,
            messages=[{"role": "system", "content": ROUTER_PROMPT},
                      {"role": "user", "content": f"Assistant's previous message: {last_bot[:400]!r}\n"
                                                  f"Customer's latest message: {message!r}"}])
        raw = resp.choices[0].message.content or ""
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        if data.get("intent") in INTENTS:
            return {"intent": data["intent"], "reason": data.get("reason", ""), "method": "llm"}
    except Exception as e:
        print(f"[router] LLM routing failed ({str(e)[:120]}); using fallback")
    return {"intent": intent or "general_question", "reason": "fallback (LLM unavailable)", "method": "fallback"}


LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

def push(event: str, payload: dict) -> None:
    """Print + file log + optional Pushover notification."""
    record = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "event": event, **payload}
    print(f"📣 [PUSH] {event}: {json.dumps(payload, ensure_ascii=False)}")
    with open(LOG_DIR / f"{event}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    user, token = os.getenv("PUSHOVER_USER"), os.getenv("PUSHOVER_TOKEN")
    if user and token:
        try:
            requests.post("https://api.pushover.net/1/messages.json", timeout=5,
                          data={"user": user, "token": token, "message": f"{event}: {payload}"})
        except Exception as e:
            print(f"(pushover failed: {e})")


def record_customer_interest(email: str, name: str = "Name not provided", message: str = "") -> dict:
    """Record a sales lead so the team can follow up."""
    email = (email or "").strip()
    if not EMAIL_RE.fullmatch(email):
        return {"status": "error", "detail": f"'{email}' is not a valid email address; ask the customer to re-check it."}
    lead_id = "LEAD-" + uuid.uuid4().hex[:6].upper()
    push("lead", {"lead_id": lead_id, "name": name or "Name not provided", "email": email, "notes": message})
    return {"status": "recorded", "lead_id": lead_id}


def record_feedback(question: str) -> dict:
    """Record customer feedback or a question the bot could not answer."""
    question = (question or "").strip()
    if not question:
        return {"status": "error", "detail": "empty question/feedback"}
    ticket_id = "FB-" + uuid.uuid4().hex[:6].upper()
    push("feedback", {"ticket_id": ticket_id, "question": question})
    return {"status": "recorded", "ticket_id": ticket_id}


TOOL_SPECS = [
    {
        "name": "record_customer_interest",
        "description": "Record a sales lead when a customer wants a quote, survey, booking or callback "
                       "AND has given their email. Call it as soon as you have the email.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "email":   {"type": "string", "description": "Customer's email address"},
                "name":    {"type": "string", "description": "Customer's name, if given"},
                "message": {"type": "string", "description": "Short note: what they want (service, size, area, timing)"},
            },
            "required": ["email"],
        },
        "outputSchema": {
            "type": "object",
            "properties": {"status": {"enum": ["recorded", "error"]}, "lead_id": {"type": "string"},
                           "detail": {"type": "string"}},
        },
    },
    {
        "name": "record_feedback",
        "description": "Log customer feedback (complaint, praise, suggestion) OR any question you could not "
                       "answer from the business knowledge.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The unanswered question or the feedback, in the customer's words"},
            },
            "required": ["question"],
        },
        "outputSchema": {
            "type": "object",
            "properties": {"status": {"enum": ["recorded", "error"]}, "ticket_id": {"type": "string"},
                           "detail": {"type": "string"}},
        },
    },
]

TOOL_FUNCTIONS = {"record_customer_interest": record_customer_interest, "record_feedback": record_feedback}

def to_openai_tool(spec: dict) -> dict:
    return {"type": "function",
            "function": {"name": spec["name"], "description": spec["description"], "parameters": spec["inputSchema"]}}

OPENAI_TOOLS = {s["name"]: to_openai_tool(s) for s in TOOL_SPECS}


ROUTES = {
    "general_question": {
        "directive": "The customer is asking a general question. Answer it from the business knowledge. "
                     "If the answer is not there, say so and call record_feedback. If they seem interested "
                     "in a service, end by offering to take their name and email for a free site survey or quote.",
        "tools": ["record_feedback", "record_customer_interest"],
        "force": None,
    },
    "lead_collection": {
        "directive": "The customer is a potential lead. Collect their name, email, and what they need "
                     "(service, size, area, timing). If you already have the email, call record_customer_interest "
                     "now. If not, ask for it before ending. Never invent availability or discounts.",
        "tools": ["record_customer_interest", "record_feedback"],
        "force": "record_customer_interest_if_email",
    },
    "feedback": {
        "directive": "The customer is giving feedback or asking something the business knowledge may not cover. "
                     "Call record_feedback with their words. If it IS answerable from the knowledge, answer too; "
                     "otherwise say the team will get back to them. Thank them and offer to take their email for follow-up.",
        "tools": ["record_feedback", "record_customer_interest"],
        "force": "record_feedback",
    },
}

def build_system_prompt(intent: str) -> str:
    return f"""# PERSONA
{PERSONA}

# INSTRUCTIONS
{INSTRUCTIONS}

# BUSINESS KNOWLEDGE
## Summary (business_summary.txt)
{business_summary}

## Full company profile (about_business.pdf)
{business_profile}

# CURRENT ROUTE: {intent}
{ROUTES[intent]['directive']}"""

system_prompt = build_system_prompt("general_question")  # the base system prompt


def call_llm(messages, tools=None, tool_choice=None):
    kwargs = {"model": MODEL, "messages": messages}
    if tools:
        kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        if tool_choice not in (None, "auto"):          # some providers reject forced tool choice
            print(f"[agent] forced tool_choice rejected ({e}); retrying with auto")
            kwargs["tool_choice"] = "auto"
            return client.chat.completions.create(**kwargs)
        raise


def handle_tool_call(tool_call) -> dict:
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    print(f"🛠  Tool called: {name}({args})")
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return {"status": "error", "detail": f"unknown tool {name}"}
    try:
        return fn(**args)
    except TypeError as e:
        return {"status": "error", "detail": f"bad arguments: {e}"}


def fallback_reply(tool_log: list) -> str:
    """Used if the model returns an empty message, so the customer never gets a blank reply."""
    done = {t["tool"] for t in tool_log if t["result"].get("status") == "recorded"}
    if "record_customer_interest" in done:
        return "Thank you! 🌱 I've passed your details to the team and they'll get back to you by email soon."
    if "record_feedback" in done:
        return "Thank you for telling us. I've passed this on to the team so they can follow up. 🌿"
    return "Sorry, I didn't quite catch that 🌱 Could you rephrase your question?"


def run_agent(message: str, history: list, route: dict, passages: str = ""):
    intent = route["intent"]
    cfg = ROUTES[intent]
    tools = [OPENAI_TOOLS[n] for n in cfg["tools"]]

    # Which tool calls to FORCE, step by step (the model chooses freely after that).
    forced_seq = []
    if cfg["force"] == "record_feedback":
        forced_seq = ["record_feedback"]
    elif cfg["force"] == "record_customer_interest_if_email" and EMAIL_RE.search(message):
        forced_seq = ["record_customer_interest"]
        if route.get("also_feedback"):          # email + complaint in the same message
            forced_seq.append("record_feedback")

    system = build_system_prompt(intent)
    if passages:   # used by the self-correction retry in run_turn
        system += ("\n\n# RELEVANT PASSAGES FROM THE BUSINESS KNOWLEDGE\nThe answer IS in the knowledge. "
                   "Answer from these passages:\n" + passages)
    messages = [{"role": "system", "content": system}] + history + [{"role": "user", "content": message}]
    tool_log = []
    for step in range(MAX_AGENT_STEPS):
        choice = {"type": "function", "function": {"name": forced_seq[step]}} if step < len(forced_seq) else "auto"
        resp = call_llm(messages, tools, choice)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return (msg.content or "").strip() or fallback_reply(tool_log), tool_log
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": tc.id, "type": "function",
                                         "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                                        for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            previous = next((t for t in tool_log if t["tool"] == tc.function.name
                             and t["result"].get("status") == "recorded"), None)
            if previous:      # each tool records at most once per turn: no duplicate leads or tickets
                result = {**previous["result"], "note": "Already recorded this turn. Do not call this tool again."}
                print(f"🛠  Skipped duplicate {tc.function.name} call")
            else:
                result = handle_tool_call(tc)
                tool_log.append({"tool": tc.function.name, "args": tc.function.arguments, "result": result})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})
    return fallback_reply(tool_log), tool_log


REFLECTION_PROMPT = f"""You are a careful quality reviewer for Nour, the chatbot of {BUSINESS_NAME}.
Decide whether the DRAFT reply breaks any of these rules:
1. Invented facts: it states a price, service, hour, policy, person or contact detail that is NOT in the BUSINESS
   KNOWLEDGE. Saying the business does NOT offer something is ALSO an invented fact unless the knowledge says so;
   the honest answer is "I don't have that information".
2. Missing lead request: the customer's LATEST message shows buying interest (quote, survey, booking, a price for
   their own space), no email has been given yet, and the reply does not invite them to share name and email.
   Do not require this if the assistant already asked in its previous message, or if the latest message is just an
   information question (hours, location, services).
3. False logging claim: the reply says something was recorded, noted or passed to the team, but TOOL STATUS says
   nothing was recorded this turn.
4. Guessing: the knowledge cannot answer the question and the reply guesses instead of saying so honestly.

Be conservative. If the draft breaks NONE of these rules, the verdict is "pass", even if you would phrase it
differently. Tone, style, length or extra helpful details are NOT reasons to revise.
If you revise: fix only the broken rule, change as little as possible, keep everything that was correct (including
true mentions of a recorded lead or ticket), and add no new facts.

BUSINESS KNOWLEDGE:
{business_summary}

{business_profile}

Return JSON only:
{{"verdict": "pass" or "revise", "issues": ["rule number + short issue", ...], "revised_reply": "<the final reply; identical to the draft if verdict is pass>"}}"""

LOGGING_CLAIM_RE = re.compile(r"\b(recorded|logged|noted (it|this|that|down|your)|i'?ve noted|made a note|"
                              r"pass(ed)? (it|this|that|your \w+)? ?(on|along)|let the team know|forwarded|"
                              r"shared (it|this|that) with the team)\b", re.I)
RULE3_ISSUE_RE = re.compile(r"^\W*(rule\s*)?3\b|logging claim|false(ly)? (claim|logg)", re.I)
_norm = lambda t: re.sub(r"\W+", " ", str(t)).strip().lower()

def reflect(message: str, history: list, draft: str, tool_log: list):
    convo = "\n".join(f"{h['role'].upper()}: {h['content']}" for h in history[-6:])
    # Rule 3 is a plain fact the code already knows, so state it explicitly instead of making the model dig for it.
    recorded = [t for t in tool_log if t["result"].get("status") == "recorded"]
    if recorded:
        ids = ", ".join(f"{t['tool']} -> {t['result'].get('lead_id') or t['result'].get('ticket_id')}" for t in recorded)
        tool_status = (f"RECORDED THIS TURN: {ids}. Any mention in the reply that this was recorded, noted or passed "
                       f"to the team is TRUE. Rule 3 cannot be broken in this turn.")
    else:
        tool_status = ("NOTHING WAS RECORDED THIS TURN. Any claim that something was recorded, noted or passed to "
                       "the team is FALSE (rule 3).")
    review_input = (f"CONVERSATION SO FAR:\n{convo or '(none)'}\n\nCUSTOMER'S LATEST MESSAGE:\n{message}\n\n"
                    f"TOOL STATUS: {tool_status}\n\nDRAFT REPLY:\n{draft}")
    try:
        resp = client.chat.completions.create(
            model=MODEL, temperature=0,
            messages=[{"role": "system", "content": REFLECTION_PROMPT}, {"role": "user", "content": review_input}])
        raw = resp.choices[0].message.content or ""
        review = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
    except Exception as e:
        review = {"verdict": "pass", "issues": [f"reflection skipped: {e}"], "revised_reply": draft}

    # Code-level guards: the reviewer model is useful for judgement calls but unreliable on checkable facts.
    issues = [str(i) for i in review.get("issues", []) if str(i).strip()]
    draft_claims_logging = bool(LOGGING_CLAIM_RE.search(draft))
    if recorded or not draft_claims_logging:
        # Rule 3 is decided by code: it is only broken if NOTHING was recorded AND the draft claims it was.
        issues = [i for i in issues if not RULE3_ISSUE_RE.search(i)]
    elif not any(RULE3_ISSUE_RE.search(i) for i in issues):
        issues.append("3 false logging claim (detected by code)")
    review["issues"] = issues
    revised = review.get("revised_reply") or draft
    same_text = _norm(revised) == _norm(draft)
    # A revision must not delete a TRUE statement that something was recorded.
    drops_true_claim = bool(recorded) and draft_claims_logging and not LOGGING_CLAIM_RE.search(revised)
    if review.get("verdict") == "revise" and (not issues or same_text or drops_true_claim):
        review["verdict"] = "pass"            # no valid rule break, or the "revision" changed nothing
    final = (review.get("revised_reply") or "").strip() if review.get("verdict") == "revise" else draft
    final = final or draft
    entry = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "user": message,
             "draft": draft, "verdict": review.get("verdict"), "issues": review.get("issues", []), "final": final}
    with open(LOG_DIR / "reflections.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"🪞 Reflection: {entry['verdict']} | issues: {entry['issues']}")
    if entry["verdict"] == "revise":
        print(f"   ✏️ DRAFT : {draft}\n   ✅ REVISED: {final}")
    return final, entry


import html as _html

TRACE_MARKER = "\n\n<sub>🔎 "

def normalize_history(history) -> list:
    """Gradio history (messages or tuple format) -> [{'role','content'}], trace footers stripped."""
    out = []
    for item in history or []:
        if isinstance(item, dict):
            role, content = item.get("role"), item.get("content")
            if isinstance(content, list):  # Gradio 6 may send content parts
                content = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
            if role in ("user", "assistant") and isinstance(content, str):
                out.append({"role": role, "content": content.split(TRACE_MARKER)[0]})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            if item[0]: out.append({"role": "user", "content": str(item[0])})
            if item[1]: out.append({"role": "assistant", "content": str(item[1]).split(TRACE_MARKER)[0]})
    return out[-MAX_HISTORY_MESSAGES:]


ADMITS_UNKNOWN = re.compile(r"\b(don'?t|do not|doesn'?t|does not) (currently )?(have|see|find)( \w+){0,3}? "
                            r"(information|info|details|data)|\bnot (sure|certain) (whether|if)\b|"
                            r"\b(isn'?t|is not|aren'?t|are not|not) (mentioned|listed|covered|offered)\b", re.I)

_STOP = set("""what which where when does have your with that this from about there their would could should
please tell want need like know much many offer offers small large some anything something""".split())
def _stem(w):
    return w[:-1] if len(w) > 4 and w.endswith("s") else w
_KNOW_TOKENS = {_stem(w) for w in re.findall(r"[a-z]+", (business_summary + " " + business_profile).lower())}
_SENTENCES = [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n", business_summary + "\n" + business_profile) if len(x.strip()) > 25]

def knowledge_check(message: str):
    """Are the question's key words in the business knowledge? Returns (hits, relevant passages or '')."""
    words = [_stem(w) for w in re.findall(r"[a-z]+", message.lower()) if len(w) > 3 and w not in _STOP]
    hits = sorted({w for w in words if w in _KNOW_TOKENS})
    if not words or len(hits) / len(set(words)) < 0.5:
        return hits, ""            # mostly unknown words (e.g. "gift cards", "installments") -> not covered
    rel = [x for x in _SENTENCES if any(re.search(rf"\b{h}", x.lower()) for h in hits)][:4]
    return hits, "\n".join(f"- {x}" for x in rel)

def run_turn(message: str, history) -> dict:
    """One full turn. Returns {'reply', 'route', 'tool_log', 'review', 'draft', 'seconds', 'error'}."""
    history = normalize_history(history)
    t0 = time.time()
    turn = {"route": None, "tool_log": [], "review": {"verdict": "off"}, "draft": "", "error": None}
    try:
        turn["route"] = route_message(message, history)
        r = turn["route"]
        print(f"\n🔀 Route: {r['intent']} ({r['method']}: {r['reason']})")
        turn["draft"], turn["tool_log"] = run_agent(message, history, r)
        # Safety net: the reply says the information is missing but nothing was logged. Check the knowledge first:
        #  - the question's key words ARE in the knowledge (e.g. "balcony") -> the model just hedged: retry once
        #    with the relevant passages;
        #  - they are NOT (e.g. "gift cards") -> re-run on the feedback path so the MODEL calls record_feedback.
        logged = any(t["tool"] == "record_feedback" for t in turn["tool_log"])
        if not logged and r["intent"] != "feedback" and ADMITS_UNKNOWN.search(turn["draft"]):
            hits, passages = knowledge_check(message)
            if passages:
                print(f"[router] reply says info is missing, but the knowledge mentions {hits} -> retrying with those passages")
                turn["draft"], turn["tool_log"] = run_agent(message, history, r, passages=passages)
                r["reason"] += " · retried with relevant passages"
            if not passages or ADMITS_UNKNOWN.search(turn["draft"]):
                print("[router] not in the knowledge and nothing was logged -> re-running on the feedback path")
                r = turn["route"] = {"intent": "feedback", "method": "self-correction",
                                     "reason": f"not in the knowledge (first routed as {r['intent']})"}
                turn["draft"], turn["tool_log"] = run_agent(message, history, r)
        if ENABLE_REFLECTION:
            turn["reply"], turn["review"] = reflect(message, history, turn["draft"], turn["tool_log"])
        else:
            turn["reply"] = turn["draft"]
    except DailyLimitError as e:
        print(f"[error] {e}")
        turn["error"], turn["reply"] = str(e), f"⚠️ {e}"
    except Exception as e:
        print(f"[error] {e}")
        turn["error"] = str(e)
        turn["reply"] = "Oops, something went wrong on my side 🌱 Please try again in a moment."
    turn["seconds"] = round(time.time() - t0, 1)
    return turn


def chat(message, history):
    """Plain-text interface (used by the quick test below and by any simple ChatInterface)."""
    turn = run_turn(message, history)
    reply = turn["reply"]
    if SHOW_TRACE and turn["route"]:
        tools = ", ".join(t["tool"] for t in turn["tool_log"]) or "none"
        reply += (f"{TRACE_MARKER}route: <b>{turn['route']['intent']}</b> · tools: <b>{tools}</b> · "
                  f"reflection: <b>{turn['review'].get('verdict')}</b></sub>")
    return reply


# ------------------------------------------------------------------ styling
GR_MAJOR = int(gr.__version__.split(".")[0])

THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.green, secondary_hue=gr.themes.colors.orange, neutral_hue=gr.themes.colors.stone,
    radius_size=gr.themes.sizes.radius_lg,
    font=[gr.themes.GoogleFont("DM Sans"), "system-ui", "sans-serif"],
)

CSS = """
:root { --sg-forest:#1F3D2B; --sg-leaf:#7FB069; --sg-sage:#A7D08C; --sg-tile:#C8693D; --sg-sun:#E9C46A; --sg-cream:#F6F1E7; }
.gradio-container { max-width: 1280px !important; margin: 0 auto !important; }
#sg-hero { position:relative; overflow:hidden; border-radius:20px; padding:28px 32px 92px; color:var(--sg-cream);
  background: radial-gradient(circle at 85% 20%, rgba(233,196,106,.45), transparent 32%),
              linear-gradient(135deg, #1F3D2B 0%, #2E5A3C 60%, #3D6B47 100%); }
#sg-hero h1 { font-family:'Fraunces', Georgia, serif; font-size:2.6rem; line-height:1.05; margin:0 0 6px; color:#fff; letter-spacing:-.01em; }
#sg-hero b { color:var(--sg-sun); }
#sg-hero p { margin:0; font-size:1.02rem; opacity:.9; max-width:620px; color:var(--sg-cream); }
#sg-hero .sg-kicker { text-transform:uppercase; letter-spacing:.18em; font-size:.72rem; color:var(--sg-sun); margin-bottom:10px; font-weight:700; }
#sg-hero .sg-chips { display:flex; gap:8px; flex-wrap:wrap; margin-top:16px; }
#sg-hero .sg-chip { background:rgba(246,241,231,.12); border:1px solid rgba(246,241,231,.25); border-radius:999px;
  padding:5px 12px; font-size:.82rem; color:#fff; }
#sg-hero svg.sg-skyline { position:absolute; left:0; right:0; bottom:-2px; width:100%; height:86px; }
.sg-panel { border-radius:16px; padding:16px 18px; background:var(--block-background-fill);
  border:1px solid var(--border-color-primary); }
.sg-panel h3 { font-family:'Fraunces', Georgia, serif; margin:0 0 12px; font-size:1.15rem; }
.sg-step { border-left:3px solid var(--sg-leaf); padding:4px 0 10px 12px; margin-left:4px; }
.sg-step + .sg-step { margin-top:4px; }
.sg-step-title { font-size:.72rem; text-transform:uppercase; letter-spacing:.12em; opacity:.65; font-weight:700; margin-bottom:6px; }
.sg-pill { display:inline-block; border-radius:999px; padding:3px 11px; font-weight:700; font-size:.85rem; }
.sg-general  { background:#DCEFD3; color:#1F3D2B; }
.sg-lead     { background:#F9E3C2; color:#7A3E12; }
.sg-feedback { background:#F6D5CC; color:#7A2615; }
.sg-pass     { background:#DCEFD3; color:#1F3D2B; }
.sg-revise   { background:#F9E3C2; color:#7A3E12; }
.sg-off      { background:#E7E2D8; color:#4A4438; }
.sg-muted { opacity:.7; font-size:.86rem; }
.sg-tool { border:1px dashed var(--border-color-primary); border-radius:12px; padding:8px 10px; margin-top:6px; font-size:.85rem; }
.sg-tool code { font-size:.78rem; white-space:pre-wrap; word-break:break-word; }
.sg-empty { text-align:center; padding:26px 8px; opacity:.6; }
.sg-inbox-item { display:flex; gap:10px; padding:8px 0; border-bottom:1px solid var(--border-color-primary); font-size:.86rem; }
.sg-inbox-item:last-child { border-bottom:none; }
.sg-inbox-icon { font-size:1.2rem; line-height:1.2; }
.sg-diff { display:grid; grid-template-columns:1fr; gap:6px; margin-top:6px; font-size:.84rem; }
.sg-diff div { border-radius:10px; padding:8px 10px; }
.sg-diff .sg-before { background:rgba(200,105,61,.12); text-decoration: line-through; text-decoration-color: rgba(200,105,61,.6); }
.sg-diff .sg-after  { background:rgba(127,176,105,.18); }
#sg-chat button.example { border:1px solid var(--border-color-primary); border-radius:14px; padding:10px 14px;
  background:var(--block-background-fill); transition:border-color .15s, transform .15s; }
#sg-chat button.example:hover { border-color:var(--sg-leaf); transform:translateY(-1px); }
.sg-count { font-family:var(--font); font-weight:500; font-size:.8rem; opacity:.65; margin-left:6px; }
#sg-footer { text-align:center; opacity:.6; font-size:.8rem; margin-top:6px; }
"""

SKYLINE = """<svg class="sg-skyline" viewBox="0 0 1200 90" preserveAspectRatio="none" aria-hidden="true">
 <g fill="#16301F">
  <rect x="0" y="40" width="90" height="50"/><rect x="95" y="20" width="70" height="70"/><rect x="170" y="50" width="110" height="40"/>
  <rect x="285" y="10" width="60" height="80"/><rect x="350" y="35" width="120" height="55"/><rect x="475" y="25" width="80" height="65"/>
  <rect x="560" y="45" width="140" height="45"/><rect x="705" y="15" width="75" height="75"/><rect x="785" y="38" width="100" height="52"/>
  <rect x="890" y="22" width="85" height="68"/><rect x="980" y="48" width="120" height="42"/><rect x="1105" y="30" width="95" height="60"/>
 </g>
 <g fill="#7FB069">
  <circle cx="115" cy="18" r="7"/><circle cx="130" cy="16" r="9"/><circle cx="300" cy="8" r="6"/><circle cx="380" cy="32" r="8"/>
  <circle cx="398" cy="30" r="10"/><circle cx="500" cy="23" r="7"/><circle cx="725" cy="13" r="8"/><circle cx="742" cy="11" r="6"/>
  <circle cx="915" cy="20" r="9"/><circle cx="1130" cy="28" r="8"/><circle cx="1148" cy="26" r="6"/>
 </g>
 <g fill="#C8693D"><rect x="560" y="42" width="140" height="4"/><rect x="170" y="47" width="110" height="4"/></g>
</svg>"""

HERO_HTML = f"""<div id="sg-hero">
  <div class="sg-kicker">Mar Mikhael · Beirut · since 2021</div>
  <h1>Sateh Green</h1>
  <p>Rooftop gardens that grow food, give shade and keep watering through power cuts.
     Chat with <b>Nour</b>, our concierge: prices, services, a free site survey, or feedback.</p>
  <div class="sg-chips">
    <span class="sg-chip">🔀 3-layer router</span><span class="sg-chip">🛠 2 tools</span>
    <span class="sg-chip">🪞 self-checking replies</span><span class="sg-chip">🌍 English · عربي · Français</span>
  </div>
  {SKYLINE}
</div>"""

# ------------------------------------------------------------------ panels
ROUTE_STYLE = {"general_question": ("sg-general", "💬 General question"),
               "lead_collection": ("sg-lead", "🌱 Lead collection"),
               "feedback": ("sg-feedback", "📝 Feedback / unknown")}
METHOD_LABEL = {"regex": "regex layer (free)", "keywords": "keyword layer (free)", "llm": "LLM layer",
                "fallback": "fallback", "self-correction": "self-correction"}
esc = lambda x: _html.escape(str(x))

def render_inspector(turn=None) -> str:
    if not turn or not turn.get("route"):
        body = '<div class="sg-empty">🌿<br>Send a message and watch Nour think:<br>route → tools → self-check</div>'
        if turn and turn.get("error"):
            body = f'<div class="sg-empty">⚠️ {esc(turn["error"])[:300]}</div>'
        return f'<div class="sg-panel"><h3>🧠 Agent Inspector</h3>{body}</div>'
    r = turn["route"]
    cls, label = ROUTE_STYLE.get(r["intent"], ("sg-off", r["intent"]))
    route_html = (f'<span class="sg-pill {cls}">{label}</span>'
                  f'<div class="sg-muted" style="margin-top:6px">via {METHOD_LABEL.get(r["method"], r["method"])}: '
                  f'{esc(r.get("reason", ""))}</div>')
    if turn["tool_log"]:
        tools_html = ""
        for t in turn["tool_log"]:
            res = t["result"]
            ok = res.get("status") == "recorded"
            ident = res.get("lead_id") or res.get("ticket_id") or res.get("detail", "")
            try:
                args = json.dumps(json.loads(t["args"]), ensure_ascii=False)
            except Exception:
                args = str(t["args"])
            tools_html += (f'<div class="sg-tool"><b>{esc(t["tool"])}</b> '
                           f'<span class="sg-pill {"sg-pass" if ok else "sg-revise"}">{"✓ " + esc(ident) if ok else "error"}</span>'
                           f'<br><code>{esc(args)}</code></div>')
    else:
        tools_html = '<div class="sg-muted">No tool needed: answered from the business knowledge.</div>'
    rv = turn["review"]
    verdict = rv.get("verdict") or "off"
    vcls = {"pass": "sg-pass", "revise": "sg-revise"}.get(verdict, "sg-off")
    vlabel = {"pass": "✓ passed", "revise": "✏️ revised", "off": "off"}.get(verdict, esc(verdict))
    refl_html = f'<span class="sg-pill {vcls}">{vlabel}</span>'
    issues = [i for i in rv.get("issues", []) if i]
    if issues:
        refl_html += '<div class="sg-muted" style="margin-top:6px">' + "<br>".join("• " + esc(i) for i in issues[:4]) + "</div>"
    if verdict == "revise":
        refl_html += (f'<div class="sg-diff"><div class="sg-before">{esc(turn["draft"])[:600]}</div>'
                      f'<div class="sg-after">{esc(turn["reply"])[:600]}</div></div>')
    return (f'<div class="sg-panel"><h3>🧠 Agent Inspector</h3>'
            f'<div class="sg-step"><div class="sg-step-title">1 · Router</div>{route_html}</div>'
            f'<div class="sg-step"><div class="sg-step-title">2 · Tools</div>{tools_html}</div>'
            f'<div class="sg-step"><div class="sg-step-title">3 · Reflection</div>{refl_html}</div>'
            f'<div class="sg-muted" style="text-align:right">⏱ {turn["seconds"]} s</div></div>')

def read_log(name: str) -> list:
    p = LOG_DIR / f"{name}.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows

def render_inbox(limit: int = 6) -> str:
    events = sorted(read_log("lead") + read_log("feedback"), key=lambda e: e.get("timestamp", ""), reverse=True)
    n_leads, n_fb = len(read_log("lead")), len(read_log("feedback"))
    head = (f'<h3>📥 Team Inbox <span class="sg-count">'
            f'{n_leads} lead(s) · {n_fb} feedback</span></h3>')
    if not events:
        return f'<div class="sg-panel">{head}<div class="sg-empty">Nothing logged yet.<br>Leads and feedback recorded by the tools appear here.</div></div>'
    items = ""
    for e in events[:limit]:
        when = esc(e.get("timestamp", "")[11:16])
        if e.get("event") == "lead":
            items += (f'<div class="sg-inbox-item"><div class="sg-inbox-icon">🌱</div><div><b>{esc(e.get("name"))}</b> · '
                      f'{esc(e.get("email"))}<br><span class="sg-muted">{esc(e.get("notes", ""))[:120]} · {esc(e.get("lead_id"))} · {when}</span></div></div>')
        else:
            items += (f'<div class="sg-inbox-item"><div class="sg-inbox-icon">📝</div><div>{esc(e.get("question"))[:140]}'
                      f'<br><span class="sg-muted">{esc(e.get("ticket_id"))} · {when}</span></div></div>')
    return f'<div class="sg-panel">{head}{items}</div>'

# ------------------------------------------------------------------ app
EXAMPLES = [
    "What do you offer for a small balcony?",
    "How much would a 25 m2 rooftop cost?",
    "I'd like a free site survey for my rooftop in Achrafieh",
    "Do you sell gift cards?",
    "The Habaq Kit you installed is leaking water 😕",
]

def ui_respond(message, history):
    turn = run_turn(message, history)
    return turn["reply"], render_inspector(turn), render_inbox()

def _make_chatbot():
    avatar = BASE_DIR / "assets" / "nour.svg"
    kw = dict(height=540, elem_id="sg-chat", show_label=False,
              avatar_images=(None, str(avatar) if avatar.exists() else None),
              placeholder="### 🌿 Ahla w sahla!\nI'm **Nour**. Ask me about rooftop gardens, balcony kits, prices for your roof, or book a free site survey.")
    try:
        return gr.Chatbot(type="messages", **kw)        # Gradio 4/5
    except TypeError:
        return gr.Chatbot(**kw)                         # Gradio 6+

HEAD = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap" rel="stylesheet">')
UI_STYLE = {"css": CSS, "theme": THEME, "head": HEAD}
BLOCKS_KW = {} if GR_MAJOR >= 6 else UI_STYLE           # Gradio 6 moved css/theme to launch()
LAUNCH_KW = UI_STYLE if GR_MAJOR >= 6 else {}

def build_demo():
    with gr.Blocks(title="Sateh Green · Chat with Nour", **BLOCKS_KW) as app:
        gr.HTML(HERO_HTML)
        inspector = gr.HTML(render_inspector(), render=False)
        inbox = gr.HTML(render_inbox(), render=False)
        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=360):
                ci_kw = dict(fn=ui_respond, chatbot=_make_chatbot(), examples=EXAMPLES, cache_examples=False,
                             additional_outputs=[inspector, inbox],
                             textbox=gr.Textbox(placeholder="Ask Nour anything about your rooftop…", show_label=False))
                try:
                    gr.ChatInterface(type="messages", **ci_kw)
                except TypeError:
                    gr.ChatInterface(**ci_kw)
            with gr.Column(scale=2, min_width=300):
                inspector.render()
                inbox.render()
                refresh = gr.Button("↻ Refresh inbox", size="sm", variant="secondary")
                refresh.click(lambda: render_inbox(), None, inbox)
        gr.HTML('<div id="sg-footer">Sateh Green is a fictional business · EECE 503P/798S Agentic Systems · C3</div>')
    return app

demo = build_demo()


if __name__ == "__main__":
    demo.launch(**LAUNCH_KW)
