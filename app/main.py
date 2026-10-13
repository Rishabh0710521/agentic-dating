"""FastAPI app: intake -> analysis -> agent dates -> rankings.  Run: uvicorn app.main:app"""
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import dating
from .analyzer import full_analysis
from .personas import PERSONAS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
STATIC = ROOT / "static"
DATA.mkdir(exist_ok=True)
CUSTOM_FILE = DATA / "custom_people.json"
CLAUDE_DATES_FILE = DATA / "claude_dates.json"

try:  # optional .env
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

app = FastAPI(title="Agentic Dating")
LOCK = threading.Lock()
PEOPLE = {}          # id -> person dict (with analysis)
DATES = {}           # (a, b) sorted -> date dict


def claude_available():
    return bool(os.getenv("ANTHROPIC_API_KEY"))


# ------------------------------------------------------------------ storage
def _with_analysis(p):
    p = dict(p)
    p["analysis"] = full_analysis(p)
    return p


def _load():
    for p in PERSONAS:
        PEOPLE[p["id"]] = _with_analysis(p)
    if CUSTOM_FILE.exists():
        for p in json.loads(CUSTOM_FILE.read_text()):
            PEOPLE[p["id"]] = _with_analysis(p)
    if CLAUDE_DATES_FILE.exists():
        for d in json.loads(CLAUDE_DATES_FILE.read_text()):
            DATES[(d["a"], d["b"])] = d


def _save_custom():
    custom = [{k: v for k, v in p.items() if k != "analysis"} for p in PEOPLE.values() if not p.get("fictional")]
    CUSTOM_FILE.write_text(json.dumps(custom, indent=1))


def _save_claude_dates():
    ds = [d for d in DATES.values() if d["engine"] == "claude"]
    CLAUDE_DATES_FILE.write_text(json.dumps(ds))


_load()


def people_list():
    return list(PEOPLE.values())


def get_date(a, b, engine="scripted", force=False):
    key = (a, b) if a < b else (b, a)
    if key in DATES and not force:
        if engine != "claude" or DATES[key]["engine"] == "claude":
            return DATES[key]
    d = dating.run_date(PEOPLE[key[0]], PEOPLE[key[1]], engine if claude_available() else "scripted")
    with LOCK:
        DATES[key] = d
        if d["engine"] == "claude":
            _save_claude_dates()
    return d


def ensure_all_dates():
    ps = people_list()
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            get_date(ps[i]["id"], ps[j]["id"])


def public_person(p):
    return {k: v for k, v in p.items()}


# ------------------------------------------------------------------ intake
LI_RX = re.compile(r"^https?://([a-z]{2,3}\.)?(www\.)?linkedin\.com/in/[\w\-%\.]+/?(\?.*)?$", re.I)
IG_RX = re.compile(r"^https?://(www\.)?instagram\.com/[\w.]+/?(\?.*)?$", re.I)


def _labelled(text, labels):
    """split 'Label: value' lines. returns dict label->[values], and the unlabelled lines"""
    out, rest = {}, []
    for line in [l.strip() for l in text.splitlines() if l.strip()]:
        m = re.match(r"^(" + "|".join(labels) + r")\s*:\s*(.+)$", line, re.I)
        if m:
            out.setdefault(m.group(1).lower(), []).append(m.group(2).strip())
        else:
            rest.append(line)
    return out, rest


def parse_linkedin(text):
    lab, rest = _labelled(text, ["headline", "about", "experience", "skills"])
    headline = (lab.get("headline") or [None])[0] or (rest.pop(0) if rest else "")
    about = " ".join(lab.get("about", []) + rest)
    skills = [s.strip() for v in lab.get("skills", []) for s in v.split(",") if s.strip()]
    return {"headline": headline, "about": about, "experience": lab.get("experience", []), "skills": skills}


def parse_instagram(text, handle):
    lab, rest = _labelled(text, ["bio", "caption"])
    bio = (lab.get("bio") or [None])[0] or (rest.pop(0) if rest else "")
    caps = lab.get("caption", []) + rest
    return {"handle": handle, "bio": bio, "captions": caps}


class NewPerson(BaseModel):
    name: str
    city: str = ""
    age: int | None = None
    linkedin_url: str
    instagram_url: str
    linkedin_text: str = ""
    instagram_text: str = ""
    consent: bool = False


@app.post("/api/people")
def add_person(body: NewPerson):
    if not body.consent:
        raise HTTPException(422, "Consent is required: only add yourself, or someone who has agreed to take part.")
    if not LI_RX.match(body.linkedin_url.strip()):
        raise HTTPException(422, "LinkedIn link must look like https://www.linkedin.com/in/your-name")
    if not IG_RX.match(body.instagram_url.strip()):
        raise HTTPException(422, "Instagram link must look like https://www.instagram.com/your.handle")
    if len(body.linkedin_text.strip()) < 40 or len(body.instagram_text.strip()) < 40:
        raise HTTPException(
            422,
            "This site does not scrape LinkedIn or Instagram. Paste the text of your own public profiles "
            "(LinkedIn headline + about, Instagram bio + a few captions) next to each link.")
    if not body.name.strip():
        raise HTTPException(422, "Name is required")
    handle = body.instagram_url.rstrip("/").split("/")[-1].split("?")[0]
    pid = re.sub(r"[^a-z0-9]+", "-", body.name.lower()).strip("-") + "-" + uuid.uuid4().hex[:4]
    person = {
        "id": pid, "name": body.name.strip(), "age": body.age, "city": body.city.strip() or "Unknown",
        "fictional": False,
        "linkedin": {"url": body.linkedin_url.strip(), **parse_linkedin(body.linkedin_text)},
        "instagram": {"url": body.instagram_url.strip(), **parse_instagram(body.instagram_text, handle)},
    }
    with LOCK:
        PEOPLE[pid] = _with_analysis(person)
        _save_custom()
    return public_person(PEOPLE[pid])


@app.delete("/api/people/{pid}")
def delete_person(pid: str):
    p = PEOPLE.get(pid)
    if not p:
        raise HTTPException(404, "not found")
    if p.get("fictional"):
        raise HTTPException(403, "Demo personas cannot be deleted")
    with LOCK:
        del PEOPLE[pid]
        for k in [k for k in DATES if pid in k]:
            del DATES[k]
        _save_custom()
    return {"deleted": pid}


# ------------------------------------------------------------------ read API
@app.get("/api/state")
def state():
    return {
        "people": [public_person(p) for p in people_list()],
        "claude_available": claude_available(),
        "mode": "live",
    }


@app.get("/api/date/{a}/{b}")
def api_date(a: str, b: str, engine: str = "scripted"):
    if a not in PEOPLE or b not in PEOPLE or a == b:
        raise HTTPException(404, "unknown pair")
    return get_date(a, b, engine)


@app.get("/api/rankings/{pid}")
def api_rank(pid: str):
    if pid not in PEOPLE:
        raise HTTPException(404, "unknown person")
    for o in PEOPLE:
        if o != pid:
            get_date(pid, o)
    return dating.rank_for(pid, people_list(), DATES)


@app.get("/api/rankings")
def api_rank_all():
    ensure_all_dates()
    ps = people_list()
    return {p["id"]: dating.rank_for(p["id"], ps, DATES) for p in ps}


@app.get("/api/export")
def export():
    ensure_all_dates()
    ps = people_list()
    return JSONResponse({
        "people": [public_person(p) for p in ps],
        "dates": list(DATES.values()),
        "rankings": {p["id"]: dating.rank_for(p["id"], ps, DATES) for p in ps},
    })


@app.get("/api/stream")
def stream(person: str | None = None, engine: str = "scripted", delay: float = 0.35):
    """Server-sent events: the agents going on their dates, one by one."""
    ids = list(PEOPLE)
    if person:
        if person not in PEOPLE:
            raise HTTPException(404, "unknown person")
        pairs = [(person, o) for o in ids if o != person]
    else:
        pairs = [(ids[i], ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))]
    delay = max(0.0, min(delay, 3.0))

    def gen():
        yield f"event: start\ndata: {json.dumps({'total': len(pairs), 'engine': engine if claude_available() else 'scripted'})}\n\n"
        for n, (a, b) in enumerate(pairs, 1):
            d = get_date(a, b, engine, force=(engine == "claude" and claude_available()))
            yield f"event: date\ndata: {json.dumps({'n': n, 'total': len(pairs), 'date': d})}\n\n"
            time.sleep(delay)
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/health")
def health():
    return {"ok": True, "people": len(PEOPLE), "dates_cached": len(DATES), "claude": claude_available()}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
