"""Profile analyzer.

Reads exactly two sources per person - LinkedIn text and Instagram text - and
produces a structured analysis. Every extracted item carries the evidence
(source + snippet) it came from, and is labelled "stated" or "inferred".

Two engines:
  * heuristic (default, offline, deterministic) - lexicon + pattern based
  * claude (if ANTHROPIC_API_KEY is set) - adds a written read of the person
    and extra qualities on top of the heuristic result
"""
import json
import os
import re
from collections import defaultdict

# ---------------------------------------------------------------- lexicons
INTERESTS = {
    "hiking & trekking": r"\b(hik\w*|trek\w*|trail\w*|mountain\w*|himalay\w*)",
    "running & fitness": r"\b(marathon|running|run|runs|10k|gym|workout|lift\w*|deadlift|strength|fitness)\b",
    "yoga & wellness": r"\b(yoga|meditat\w*|mindful\w*|pilates|wellness)",
    "cooking & food": r"\b(cook\w*|bak(e|ing)|recipe\w*|street food|biryani|chef|sambar|dal baati|feast\w*|food)\b",
    "coffee & chai": r"\b(coffee|espresso|roaster\w*|kaapi|chai)\b",
    "travel": r"\b(travel\w*|backpack\w*|passport\w*|road ?trips?|wanderlust|solo trip|itinerary|trip)\b",
    "photography": r"\b(photograph\w*|camera\w*|film roll|roll of film)\b",
    "live music": r"\b(concert\w*|live music|gigs?|festival\w*|vinyl|raag|hindustani|music)\b",
    "reading": r"\b(books?|reading|novel\w*|bookshop\w*|library)\b",
    "gaming": r"\b(gaming|games?|controllers?|esports)\b",
    "art & design": r"\b(sketch\w*|painting\w*|illustrat\w*|galler(y|ies)|museum\w*|art)\b",
    "film & theatre": r"\b(film\w*|cinema|movies?|theatre|documentar\w*)\b",
    "tech & AI": r"\b(machine learning|AI|LLMs?|python|rust|software|code|open source|data|engineer\w*)\b",
    "startups": r"\b(startups?|founder|bootstrapp\w*|venture|seed round)\b",
    "dogs & pets": r"\b(dogs?|puppy|puppies|cats?|rescue|pets?)\b",
    "cycling": r"\b(cycl\w*)\b",
    "fashion & thrifting": r"\b(thrift\w*|fashion|styl\w*|vintage)\b",
    "volunteering": r"\b(volunteer\w*|NGO|nonprofit|community)\b",
    "board games & chess": r"\b(board games?|chess|catan)\b",
    "gardening": r"\b(garden\w*|plants?|balcony|tulsi|tomatoes)\b",
    "dance": r"\b(danc\w*|bharatanatyam|kathak|salsa)\b",
    "sports": r"\b(cricket|football|badminton|tennis|swim\w*|surf\w*|div(e|ed|ing))\b",
    "investing": r"\b(invest\w*|mutual funds?|SIPs?|stocks?|stock market|savings)\b",
    "writing & poetry": r"\b(writing|writer|journal\w*|poetry|poem\w*|essay\w*|blog\w*)\b",
    "sustainability": r"\b(sustainab\w*|climate|zero waste|solar|renewable)\b",
    "comedy": r"\b(stand-up|comedy|open mics?|jokes?)\b",
}
PROFESSIONAL = {"tech & AI", "startups", "investing"}

VALUES = {
    "ambition": r"\b(ambiti\w*|driven|hustl\w*|grind\w*|goals?|discipline|hardworking|build\w*)\b",
    "curiosity": r"\b(curio\w*|ideas?|debate\w*|learn\w*|explor\w*)\b",
    "kindness & empathy": r"\b(kind\w*|empath\w*|gentle|warm\w*|patience|caring?|cares?)\b",
    "family": r"\b(family|tradition\w*|grandmother|settle\w*)\b",
    "independence": r"\b(independen\w*|space|solo|alone)\b",
    "adventure": r"\b(adventur\w*|spontaneous\w*|wander\w*|explor\w*|say yes)\b",
    "creativity": r"\b(creativ\w*|art\w*|sketch\w*|poetry|design\w*)\b",
    "stability": r"\b(stab(le|ility)|grounded|routine|calm|peace|boring is beautiful)\b",
    "humor": r"\b(funny|laugh\w*|jokes?|humou?r|comedy)\b",
    "honesty": r"\b(honest\w*|loyal\w*|integrity)\b",
    "community": r"\b(communit\w*|friends|volunteer\w*|together)\b",
}

# need / offer tags: what someone wants from a partner / what they bring
TAGS = {
    "supports_ambition": r"\b(ambiti\w*|driven|big goals|hustl\w*|grind\w*|discipline|hardworking)\b",
    "adventure_partner": r"\b(adventur\w*|spontaneous\w*|travel\w*|trek\w*|hik\w*|road ?trips?|wander\w*|backpack\w*)\b",
    "calm_home": r"\b(calm|quiet|homebod\w*|cozy|slow (weekends?|mornings?)|stab(le|ility)|recharg\w*|peace)\b",
    "creative_spark": r"\b(creativ\w*|artsy|sketch\w*|poetry|film\w*|cinema|music|design\w*)\b",
    "intellectual_talk": r"\b(curio\w*|deep conversations?|long conversations?|ideas?|books?|reading|debate\w*|adda)\b",
    "active_lifestyle": r"\b(active|gym|hik\w*|run|running|runs|yoga|cycl\w*|badminton|workout|lift\w*|fitness)\b",
    "family_oriented": r"\b(family|settle\w*|tradition\w*|kids)\b",
    "humor": r"\b(funny|laugh\w*|humou?r|jokes?|comedy|fun)\b",
    "emotional_depth": r"\b(kind\w*|empath\w*|emotionally|warm\w*|gentle|honest\w*)\b",
    "independence_respect": r"\b(independen\w*|space|solo)\b",
    "social_energy": r"\b(part(y|ies)|crowd|big group|hosted|loud|social|outgoing|gigs?)\b",
}
TAG_LABEL = {
    "supports_ambition": "a partner who supports big goals",
    "adventure_partner": "an adventurous travel/trek partner",
    "calm_home": "calm, slow-paced time together",
    "creative_spark": "creative spark (art, film, music)",
    "intellectual_talk": "deep, curious conversation",
    "active_lifestyle": "an active lifestyle",
    "family_oriented": "a family-oriented outlook",
    "humor": "humour and fun",
    "emotional_depth": "kindness and emotional honesty",
    "independence_respect": "respect for independence",
    "social_energy": "a social, outgoing energy",
}

DEALBREAKER_ATTR = {
    "smoking": r"\b(cigarette\w*|vape\w*|hookah|smoker)\b",
    "heavy partying": r"\b(clubbing|bar ?hopping|party at mine|late nights and loud|everyone invited)\b",
    "workaholic": r"\b(workaholic|80 hour|hustle culture|red-eye)\b",
}
DEALBREAKER_TRIGGER = r"(deal-?breakers?|not into|can't stand)\s*:?\s*([^.!\n]+)"

NEED_SENTENCE = r"(looking for|want(s)? (someone|a partner)|i'd (like|love) (someone|a partner)|i want (someone|a partner)|need(s)? (someone|a partner))"


# ---------------------------------------------------------------- helpers
def _sentences(text):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _source_chunks(person):
    li, ig = person["linkedin"], person["instagram"]
    chunks = []
    for s in _sentences(li.get("headline", "")):
        chunks.append(("LinkedIn headline", s))
    for s in _sentences(li.get("about", "")):
        chunks.append(("LinkedIn about", s))
    for s in li.get("experience", []):
        chunks.append(("LinkedIn experience", s))
    for s in li.get("skills", []):
        chunks.append(("LinkedIn skills", s))
    for s in _sentences(ig.get("bio", "")):
        chunks.append(("Instagram bio", s))
    for s in ig.get("captions", []):
        for sub in _sentences(s):
            chunks.append(("Instagram caption", sub))
    return chunks


def _match(lexicon, chunks, flags=re.I):
    """returns {key: {"count": n, "evidence": [...]}}"""
    out = defaultdict(lambda: {"count": 0, "evidence": [], "sources": set()})
    for key, pat in lexicon.items():
        rx = re.compile(pat, flags)
        for src, text in chunks:
            if rx.search(text):
                d = out[key]
                d["count"] += 1
                d["sources"].add(src.split()[0])
                if len(d["evidence"]) < 3 and not any(e["text"] == text for e in d["evidence"]):
                    d["evidence"].append({"source": src, "text": text})
    return out


def _full_text(person):
    return " ".join(t for _, t in _source_chunks(person))


# ---------------------------------------------------------------- analysis
def analyze(person):
    chunks = _source_chunks(person)
    text = _full_text(person)

    # interests / hobbies
    ints = _match(INTERESTS, chunks)
    interests = []
    for k, d in sorted(ints.items(), key=lambda kv: -kv[1]["count"]):
        interests.append({
            "name": k,
            "strength": min(1.0, 0.35 + 0.25 * d["count"]),
            "professional": k in PROFESSIONAL,
            "from": sorted(d["sources"]),
            "evidence": d["evidence"],
            "basis": "stated",
        })
    hobbies = [i for i in interests if not i["professional"] and "Instagram" in i["from"]]
    if len(hobbies) < 3:
        hobbies = [i for i in interests if not i["professional"]][:6]

    # values
    vals = _match(VALUES, chunks)
    values = []
    for k, d in sorted(vals.items(), key=lambda kv: -kv[1]["count"]):
        values.append({
            "name": k,
            "strength": min(1.0, 0.3 + 0.25 * d["count"]),
            "from": sorted(d["sources"]),
            "evidence": d["evidence"],
            "basis": "inferred",
        })
    values = values[:6]

    # offers (what they bring) and needs (what they want)
    tag_hits = _match(TAGS, chunks)
    offers = {t: min(1.0, 0.4 + 0.2 * d["count"]) for t, d in tag_hits.items()}

    needs = []
    need_weights = {}
    explicit_sents = [(s, t) for s, t in chunks if re.search(NEED_SENTENCE, t, re.I)]
    for src, sent in explicit_sents:
        after = re.split(NEED_SENTENCE, sent, flags=re.I)[-1]
        for tag, pat in TAGS.items():
            if re.search(pat, sent, re.I):
                need_weights[tag] = max(need_weights.get(tag, 0), 1.0)
                needs.append({"need": TAG_LABEL[tag], "tag": tag, "basis": "stated",
                              "evidence": [{"source": src, "text": sent}]})
    seen = {n["tag"] for n in needs}
    # inferred needs: people tend to want a partner who shares what they value most
    own_tags = sorted(offers.items(), key=lambda kv: -kv[1])
    for tag, strength in own_tags:
        if tag in seen or tag == "social_energy" and "social" not in text.lower():
            continue
        if len([n for n in needs if n["basis"] == "inferred"]) >= 3:
            break
        ev = tag_hits[tag]["evidence"][:1]
        need_weights[tag] = 0.4
        needs.append({"need": TAG_LABEL[tag], "tag": tag, "basis": "inferred", "evidence": ev})

    # deal-breakers: stated ones + attributes of the person
    deal_stated = []
    for src, t in chunks:
        m = re.search(DEALBREAKER_TRIGGER, t, re.I)
        if m:
            deal_stated.append({"text": m.group(0).strip(), "source": src})
    own_text = " ".join(t for _, t in chunks if not re.search(DEALBREAKER_TRIGGER, t, re.I))
    attrs = [a for a, pat in DEALBREAKER_ATTR.items() if re.search(pat, own_text, re.I)]
    # social energy
    intro = len(re.findall(r"\b(introvert\w*|quiet|homebod\w*|cozy|recharg\w*|alone)\b", text, re.I))
    extro = len(re.findall(r"\b(part(y|ies)|crowd|big group|hosted|loud|outgoing|social|friends|gigs?|everyone invited)\b", text, re.I))
    energy = 0.5 + 0.12 * (extro - intro)
    energy = max(0.15, min(0.9, energy))
    energy_label = "high-energy and social" if energy > 0.68 else "low-key and reflective" if energy < 0.35 else "balanced - social but needs downtime"

    # communication style from Instagram captions
    caps = person["instagram"].get("captions", [])
    avg_len = sum(len(c) for c in caps) / max(1, len(caps))
    excl = sum(c.count("!") for c in caps)
    quest = sum(c.count("?") for c in caps)
    style_bits = []
    if avg_len < 75:
        style_bits.append("short and punchy")
    elif avg_len > 110:
        style_bits.append("storyteller")
    else:
        style_bits.append("concise but warm")
    if excl >= 2:
        style_bits.append("enthusiastic")
    if quest:
        style_bits.append("asks questions")
    if not style_bits:
        style_bits.append("understated")
    comm = {
        "label": ", ".join(style_bits),
        "basis": "inferred",
        "stats": {"avg_caption_chars": round(avg_len), "exclamations": excl, "questions": quest},
    }

    # lifestyle flags
    open_move = bool(re.search(r"open to (moving|relocat)", text, re.I))

    # professional read
    pro = {
        "headline": person["linkedin"].get("headline", ""),
        "skills": person["linkedin"].get("skills", []),
        "career": person["linkedin"].get("experience", []),
        "work_intensity": "very high" if "workaholic" in attrs or "hustle" in text.lower() else "moderate",
    }

    top_h = [h["name"] for h in hobbies[:3]]
    top_v = [v["name"] for v in values[:2]]
    top_n = needs[0]["need"] if needs else "someone compatible"
    summary = (f"{person['name'].split()[0]} reads as {energy_label}. Core values: {', '.join(top_v) or 'not clear'}. "
               f"Free time goes to {', '.join(top_h) or 'unclear hobbies'}. "
               f"Seems to want {top_n}.")

    qualities = []
    for v in values[:4]:
        qualities.append(v["name"])
    if energy > 0.68:
        qualities.append("sociable")
    elif energy < 0.35:
        qualities.append("introspective")
    if offers.get("humor", 0) > 0.55:
        qualities.append("playful")

    return {
        "summary": summary,
        "engine": "heuristic",
        "hobbies": hobbies,
        "interests": interests,
        "professional": pro,
        "values": values,
        "needs": needs,
        "need_weights": need_weights,
        "offers": offers,
        "offer_evidence": {t: d["evidence"][:1] for t, d in tag_hits.items()},
        "deal_breakers_stated": deal_stated,
        "traits_that_others_may_mind": attrs,
        "social_energy": {"score": round(energy, 2), "label": energy_label, "basis": "inferred"},
        "communication_style": comm,
        "qualities": list(dict.fromkeys(qualities)),
        "city": person.get("city", ""),
        "open_to_relocate": open_move,
    }


# ---------------------------------------------------------------- Claude engine
def analyze_with_claude(person, base):
    """Adds a written psychological-free, evidence-grounded read on top of the
    heuristic result. Falls back silently to the heuristic result on any error."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return base
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        prompt = (
            "You are the agent that represents one person on a dating platform. You may only use the two sources "
            "below (their LinkedIn text and Instagram text). Do not invent facts. Return JSON with keys: "
            "summary (2-3 sentences), extra_qualities (list of up to 5 short strings, each grounded in the text), "
            "ideal_partner (one sentence, marked as inferred unless stated), watch_outs (list of up to 2).\n\n"
            f"LINKEDIN:\n{json.dumps(person['linkedin'])}\n\nINSTAGRAM:\n{json.dumps(person['instagram'])}"
        )
        msg = client.messages.create(
            model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"),
            max_tokens=700,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text
        raw = raw[raw.find("{"): raw.rfind("}") + 1]
        extra = json.loads(raw)
        base = dict(base)
        base["summary"] = extra.get("summary", base["summary"])
        base["qualities"] = list(dict.fromkeys(base["qualities"] + extra.get("extra_qualities", [])))
        base["ideal_partner"] = extra.get("ideal_partner")
        base["watch_outs"] = extra.get("watch_outs", [])
        base["engine"] = "claude"
    except Exception as e:  # pragma: no cover
        base = dict(base)
        base["engine_note"] = f"claude analysis failed, used heuristic: {type(e).__name__}"
    return base


def full_analysis(person):
    return analyze_with_claude(person, analyze(person))
