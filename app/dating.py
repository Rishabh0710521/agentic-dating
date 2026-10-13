"""The dating harness.

Each person is represented by an agent that only knows what the analyzer read
from that person's LinkedIn + Instagram. Two agents "go on a date" (a
multi-turn conversation), then each side gives its own verdict.

Engines
  scripted : deterministic. Turns are assembled from each agent's own analysis
             and evidence (no LLM). Always available, used for the pre-run demo
             unless regenerated with Claude.
  claude   : every turn is a separate Claude call. Each agent sees ONLY its own
             person's sources plus the conversation so far (information
             asymmetry, like a real first date), then gives a JSON verdict.

Scoring (per side, 0-100) is always computed by the same deterministic model,
so scores are comparable across engines. In claude mode the agent's own verdict
is blended in (40%).
"""
import json
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor

from .analyzer import DEALBREAKER_ATTR, TAG_LABEL

W = {"interests": 0.33, "values": 0.17, "needs": 0.27, "energy": 0.08, "city": 0.15}
PENALTY = 18


# ----------------------------------------------------------------- scoring
def _wj(da, db):
    """weighted jaccard between {name: strength} dicts"""
    keys = set(da) | set(db)
    if not keys:
        return 0.0
    num = sum(min(da.get(k, 0), db.get(k, 0)) for k in keys)
    den = sum(max(da.get(k, 0), db.get(k, 0)) for k in keys)
    return num / den if den else 0.0


def _needs_met(need_w, offers):
    if not need_w:
        return 0.5, [], []
    met, missed, tot, got = [], [], 0.0, 0.0
    for tag, w in need_w.items():
        tot += w
        o = offers.get(tag, 0)
        if o >= 0.4:
            got += w * min(1.0, o / 0.6)
            met.append(tag)
        else:
            missed.append(tag)
    return (got / tot if tot else 0.5), met, missed


def _clash(analysis_x, analysis_y):
    """x's stated deal-breakers that y's own profile triggers"""
    hits = []
    for d in analysis_x["deal_breakers_stated"]:
        for attr in analysis_y["traits_that_others_may_mind"]:
            if re.search(DEALBREAKER_ATTR[attr], d["text"], re.I) or attr.split()[-1] in d["text"].lower():
                hits.append({"deal_breaker": d["text"], "trait": attr})
    return hits


def score_pair(pa, pb):
    A, B = pa["analysis"], pb["analysis"]
    ia = {i["name"]: i["strength"] * (0.6 if i["professional"] else 1.0) for i in A["interests"]}
    ib = {i["name"]: i["strength"] * (0.6 if i["professional"] else 1.0) for i in B["interests"]}
    va = {v["name"]: v["strength"] for v in A["values"]}
    vb = {v["name"]: v["strength"] for v in B["values"]}
    interest = min(1.0, 2.2 * _wj(ia, ib))
    values = min(1.0, 1.7 * _wj(va, vb))
    n_a, met_a, miss_a = _needs_met(A["need_weights"], B["offers"])
    n_b, met_b, miss_b = _needs_met(B["need_weights"], A["offers"])
    energy = max(0.0, 1 - 1.2 * abs(A["social_energy"]["score"] - B["social_energy"]["score"]))
    same_city = pa["city"] == pb["city"]
    city = 1.0 if same_city else (0.7 if (A["open_to_relocate"] or B["open_to_relocate"]) else 0.45)
    clash_a = _clash(A, B)   # a's deal-breakers hit by b
    clash_b = _clash(B, A)
    # raw 0-100 -> 25-100 so a merely-different pair is not scored as a disaster
    cal = lambda raw: 25 + 0.75 * raw
    view_a = cal(100 * (W["interests"] * interest + W["values"] * values + W["needs"] * n_a + W["energy"] * energy + W["city"] * city)) - PENALTY * len(clash_a)
    view_b = cal(100 * (W["interests"] * interest + W["values"] * values + W["needs"] * n_b + W["energy"] * energy + W["city"] * city)) - PENALTY * len(clash_b)
    shared_i = sorted(set(ia) & set(ib), key=lambda k: -(min(ia[k], ib[k])))
    shared_v = sorted(set(va) & set(vb), key=lambda k: -(min(va[k], vb[k])))
    return {
        "view_a": max(0, round(view_a)), "view_b": max(0, round(view_b)),
        "breakdown": {"interests": round(interest, 2), "values": round(values, 2), "needs_a": round(n_a, 2),
                      "needs_b": round(n_b, 2), "energy": round(energy, 2), "city": city,
                      "penalty_a": PENALTY * len(clash_a), "penalty_b": PENALTY * len(clash_b)},
        "shared_interests": shared_i, "shared_values": shared_v,
        "met_a": met_a, "missed_a": miss_a, "met_b": met_b, "missed_b": miss_b,
        "clash_a": clash_a, "clash_b": clash_b, "same_city": same_city,
    }


def label_for(score):
    return ("strong match" if score >= 70 else "promising" if score >= 58 else
            "maybe" if score >= 45 else "unlikely")


def meet_again(score):
    return "yes" if score >= 62 else "maybe" if score >= 48 else "no"


# ----------------------------------------------------------------- reasons
def reasons(pa, pb, sc, side):
    """reasons from `side` ('a'|'b') perspective. Returns (why, watch)."""
    me, other = (pa, pb) if side == "a" else (pb, pa)
    met = sc["met_" + side]
    missed = sc["missed_" + side]
    why, watch = [], []
    if sc["shared_interests"]:
        why.append("Shared interests: " + ", ".join(sc["shared_interests"][:3]))
    if sc["shared_values"]:
        why.append("Shared values: " + ", ".join(sc["shared_values"][:2]))
    stated = {n["tag"] for n in me["analysis"]["needs"] if n["basis"] == "stated"}
    for t in met:
        why.append(f"{other['name'].split()[0]} offers {TAG_LABEL[t]}" + (" - which you asked for" if t in stated else ""))
    if sc["breakdown"]["energy"] >= 0.8:
        why.append("Similar social energy")
    if sc["same_city"]:
        why.append(f"Both in {me['city']}")
    for t in missed:
        if t in stated:
            watch.append(f"Nothing in {other['name'].split()[0]}'s profile shows {TAG_LABEL[t]} (you asked for it)")
    if sc["breakdown"]["energy"] < 0.5:
        watch.append(f"Different social energy ({me['analysis']['social_energy']['label']} vs {other['analysis']['social_energy']['label']})")
    if not sc["same_city"]:
        watch.append(f"Different cities ({me['city']} / {other['city']})")
    for c in sc["clash_" + side]:
        watch.insert(0, f"Deal-breaker clash: you wrote \"{c['deal_breaker']}\"; {other['name'].split()[0]}'s profile suggests {c['trait']}")
    return why[:5], watch[:4]


# ----------------------------------------------------------------- scripted dates
def _short_title(p):
    h = p["linkedin"]["headline"].split("|")[0].strip()
    art = "an" if h[:1] in "AEIO" else "a"
    return f"{art} {h}"


def _quote(analysis, name=None, prefer="Instagram caption"):
    """-> (text, source) or None. Prefers Instagram captions, then any evidence."""
    pools = analysis["interests"] if name is None else [i for i in analysis["interests"] if i["name"] == name]
    for it in pools:
        for e in it["evidence"]:
            if e["source"] == prefer:
                return _trim(e["text"]), e["source"]
    for it in pools:
        if it["evidence"]:
            e = it["evidence"][0]
            return _trim(e["text"]), e["source"]
    return None


def _trim(s, n=115):
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _top_h(p, k=2):
    hs = [h["name"] for h in p["analysis"]["hobbies"][:k]]
    return " and ".join(hs) if hs else "a mix of things"


def scripted_date(pa, pb, sc):
    rng = random.Random(f"{pa['id']}|{pb['id']}")
    fa, fb = pa["name"].split()[0], pb["name"].split()[0]
    A, B = pa["analysis"], pb["analysis"]
    shared = sc["shared_interests"]
    turns = []

    def say(who, kind, text):
        p = pa if who == "a" else pb
        turns.append({"speaker": p["id"], "name": p["name"], "agent": f"{p['name'].split()[0]}'s agent", "kind": kind, "text": text})

    # 1 opener
    if shared:
        say("a", "opener", rng.choice([
            f"Hi, I'm {fa}'s agent. {fa} is {_short_title(pa)}. I spotted {shared[0]} on {fb}'s profile, and it's a big part of {fa}'s life too. Tell me about {fb}.",
            f"Hello! I represent {fa}, {_short_title(pa)}. We seem to overlap on {shared[0]}. What's {fb} like when not working?"]))
    else:
        say("a", "opener", rng.choice([
            f"Hi, I'm {fa}'s agent. {fa} is {_short_title(pa)} who spends free time on {_top_h(pa)}. What's {fb}'s world like?",
            f"Hello! I represent {fa}, {_short_title(pa)}. Free time goes to {_top_h(pa)}. I don't see an obvious overlap yet, so tell me about {fb}."]))
    # 2 reply
    q = _quote(B, shared[0]) if shared else _quote(B)
    ptxt = f"Nice to meet you. I'm {fb}'s agent: {fb} is {_short_title(pb)}. "
    if shared and q:
        ptxt += f"On {shared[0]}, this is from {fb}'s {q[1]}: \"{q[0]}\" That gives you the flavour."
    elif q:
        ptxt += f"{fb}'s world is mostly {_top_h(pb)}. This line from the {q[1]} sums it up: \"{q[0]}\""
    else:
        ptxt += f"{fb}'s free time goes to {_top_h(pb)}."
    say("b", "interests", ptxt)
    # 3 deepen
    qa = _quote(A, shared[0]) if shared else _quote(A)
    if shared and qa:
        say("a", "interests", f"Good sign. {fa}'s version, from the {qa[1]}: \"{qa[0]}\" How central is {shared[0]} to {fb}'s weekends?")
        it = next((i for i in B["interests"] if i["name"] == shared[0]), None)
        srcs = it["from"] if it else []
        if len(srcs) >= 2:
            say("b", "interests", f"It appears in both {fb}'s {' and '.join(srcs)} text, so it isn't a one-off. I'd call it a real interest.")
        else:
            say("b", "interests", f"I can only point to {fb}'s {srcs[0] if srcs else 'profile'} for that. It's a real signal, but one source, so I'd hold it lightly.")
    else:
        say("a", "interests", f"{fa} lives for {_top_h(pa)}; {fb} for {_top_h(pb)}. Is {fb} open to trying things outside that, or do routines rule?")
        va = B["values"][0]["name"] if B["values"] else "routine"
        say("b", "values", f"From {fb}'s profile, {va} comes first, so {fb} is open when it fits what matters to them. It would depend on how {fa} proposes it.")
    # 4 values
    vals_a = ", ".join(v["name"] for v in A["values"][:2]) or "being genuine"
    vals_b = ", ".join(v["name"] for v in B["values"][:2]) or "being genuine"
    say("a", "values", f"What matters most to {fb} in a relationship? For {fa} it's {vals_a}.")
    sv = sc["shared_values"]
    say("b", "values", f"{fb}'s profile points to {vals_b}." + (f" That overlaps on {sv[0]}." if sv else " Different emphasis from {0}, but not opposed.".format(fa)))
    # 5 needs, both ways
    need_a = next((n for n in A["needs"] if n["basis"] == "stated"), A["needs"][0] if A["needs"] else None)
    if need_a:
        say("a", "needs", f"{fa} is looking for {need_a['need']} ({need_a['basis']}). Does {fb} bring that?")
        ev = B["offer_evidence"].get(need_a["tag"])
        if need_a["tag"] in sc["met_a"] and ev:
            say("b", "needs", f"Yes. {fb}'s profile shows it: \"{_trim(ev[0]['text'])}\" ({ev[0]['source']}).")
        elif need_a["tag"] in sc["met_a"]:
            say("b", "needs", f"Yes, it fits {fb} well.")
        else:
            say("b", "needs", f"Honestly, nothing in {fb}'s profile points to that. I won't claim it.")
    need_b = next((n for n in B["needs"] if n["basis"] == "stated"), B["needs"][0] if B["needs"] else None)
    if need_b:
        say("b", "needs", f"Fair question back: {fb} wants {need_b['need']} ({need_b['basis']}). Is that {fa}?")
        ev = A["offer_evidence"].get(need_b["tag"])
        if need_b["tag"] in sc["met_b"] and ev:
            say("a", "needs", f"It is. {fa}'s profile shows it: \"{_trim(ev[0]['text'])}\" ({ev[0]['source']}).")
        elif need_b["tag"] in sc["met_b"]:
            say("a", "needs", f"Yes, that fits {fa}.")
        else:
            say("a", "needs", f"Not clearly, no. {fa}'s profile doesn't show it, and I'd rather be straight about that.")
    # 6 friction
    for side, clashes, pw, po in (("a", sc["clash_a"], pa, pb), ("b", sc["clash_b"], pb, pa)):
        for c in clashes:
            fn_w, fn_o = pw["name"].split()[0], po["name"].split()[0]
            say(side, "deal-breaker", f"One flag. {fn_w}'s profile says \"{c['deal_breaker']}\", and {fn_o}'s posts suggest {c['trait']}.")
            say("b" if side == "a" else "a", "deal-breaker", f"Understood. That's a real gap and I won't pretend otherwise.")
    # 7 energy / logistics
    if not sc["same_city"]:
        say("a", "logistics", f"Logistics: {fa} is in {pa['city']}, {fb} in {pb['city']}. Does that work?")
        say("b", "logistics", f"It's a hurdle. Nothing in {fb}'s profile says they'd relocate, so it would need a conversation.")
    # 8 verdicts
    for side, p, other, v in (("a", pa, pb, sc["view_a"]), ("b", pb, pa, sc["view_b"])):
        why, watch = reasons(pa, pb, sc, side)
        if v < 45 and watch:
            tail = f"Main issue: {watch[0]}." + (f" Upside: {why[0]}." if why else "")
        else:
            tail = (f"{why[0]}." if why else "No strong overlap.") + (f" Watch-out: {watch[0]}." if watch else "")
        say(side, "verdict", f"Verdict for {p['name'].split()[0]}: {v}/100 ({label_for(v)}), would meet again: {meet_again(v)}. {tail}")
    return turns


# ----------------------------------------------------------------- claude dates
_client = None


def llm_call(system, messages, max_tokens=300):
    """single Claude call. Separate function so it can be swapped in tests."""
    global _client
    import anthropic
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    msg = _client.messages.create(model=os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5"),
                                  max_tokens=max_tokens, system=system, messages=messages)
    return msg.content[0].text


def _agent_system(p, other_first):
    return (
        f"You are {p['name'].split()[0]}'s dating agent. You speak on {p['name'].split()[0]}'s behalf in a first date "
        f"with {other_first}'s agent. You know ONLY these two sources about your person; never invent facts:\n"
        f"LINKEDIN: {json.dumps(p['linkedin'])}\nINSTAGRAM: {json.dumps(p['instagram'])}\n"
        f"Your analysis of them: {json.dumps({k: p['analysis'][k] for k in ('summary', 'values', 'needs', 'social_energy')})[:1800]}\n"
        "Be warm but honest: if something is not in the sources, say so. Reply in 1-3 sentences, plain text, no markdown."
    )


def claude_date(pa, pb, sc, turns_n=6):
    fa, fb = pa["name"].split()[0], pb["name"].split()[0]
    sys_a, sys_b = _agent_system(pa, fb), _agent_system(pb, fa)
    convo = []
    steps = ["Open the date: introduce yourself and your person, and ask something about the other person.",
             "Reply, and explore shared interests or hobbies.", "Ask about values and what they want in a relationship.",
             "Answer honestly, then probe whether your person's needs would be met.",
             "Answer the needs question honestly, with evidence from your sources.",
             "Close the conversation naturally."]
    turns = []
    for i in range(turns_n):
        who, p, system = ("a", pa, sys_a) if i % 2 == 0 else ("b", pb, sys_b)
        history = "\n".join(f"{t['agent']}: {t['text']}" for t in turns) or "(no messages yet)"
        text = llm_call(system, [{"role": "user", "content": f"Conversation so far:\n{history}\n\nYour move. {steps[i]}"}]).strip()
        turns.append({"speaker": p["id"], "name": p["name"], "agent": f"{p['name'].split()[0]}'s agent",
                      "kind": ["opener", "interests", "values", "needs", "needs", "closing"][i], "text": text})
    history = "\n".join(f"{t['agent']}: {t['text']}" for t in turns)
    verdicts = {}
    for who, p, system in (("a", pa, sys_a), ("b", pb, sys_b)):
        raw = llm_call(system, [{"role": "user", "content": f"Date transcript:\n{history}\n\nGive your verdict for {p['name'].split()[0]}. "
                                 'Return ONLY JSON: {"score": 0-100, "reason": "one sentence", "meet_again": "yes|maybe|no"}'}])
        try:
            verdicts[who] = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        except Exception:
            verdicts[who] = {"score": sc["view_" + who], "reason": "(agent verdict unparsable)", "meet_again": meet_again(sc["view_" + who])}
    for who, p in (("a", pa), ("b", pb)):
        v = verdicts[who]
        blended = round(0.6 * sc["view_" + who] + 0.4 * float(v.get("score", sc["view_" + who])))
        sc["view_" + who] = blended
        turns.append({"speaker": p["id"], "name": p["name"], "agent": f"{p['name'].split()[0]}'s agent", "kind": "verdict",
                      "text": f"Verdict for {p['name'].split()[0]}: {blended}/100 ({label_for(blended)}), would meet again: {v.get('meet_again', meet_again(blended))}. {v.get('reason', '')}"})
    return turns


# ----------------------------------------------------------------- public API
def run_date(pa, pb, engine="scripted"):
    """pa, pb: person dicts with 'analysis'. Order-independent."""
    if pa["id"] > pb["id"]:
        pa, pb = pb, pa
    sc = score_pair(pa, pb)
    used = "scripted"
    if engine == "claude" and os.getenv("ANTHROPIC_API_KEY"):
        try:
            turns = claude_date(pa, pb, sc)
            used = "claude"
        except Exception as e:
            turns = scripted_date(pa, pb, sc)
            used = f"scripted (claude failed: {type(e).__name__})"
    else:
        turns = scripted_date(pa, pb, sc)
    why_a, watch_a = reasons(pa, pb, sc, "a")
    why_b, watch_b = reasons(pa, pb, sc, "b")
    va, vb = sc["view_a"], sc["view_b"]
    return {
        "a": pa["id"], "b": pb["id"], "engine": used, "turns": turns,
        "scores": {"a_view": va, "b_view": vb, "mutual": round((va + vb) / 2)},
        "breakdown": sc["breakdown"],
        "shared_interests": sc["shared_interests"], "shared_values": sc["shared_values"],
        "reasons": {"a": {"why": why_a, "watch": watch_a}, "b": {"why": why_b, "watch": watch_b}},
        "verdict": {"a": {"score": va, "label": label_for(va), "meet_again": meet_again(va)},
                    "b": {"score": vb, "label": label_for(vb), "meet_again": meet_again(vb)}},
    }


def run_all(people, engine="scripted", workers=1, progress=None):
    pairs = [(people[i], people[j]) for i in range(len(people)) for j in range(i + 1, len(people))]
    out = {}

    def one(pr):
        d = run_date(pr[0], pr[1], engine)
        return d
    if workers > 1:
        with ThreadPoolExecutor(workers) as ex:
            for d in ex.map(one, pairs):
                out[(d["a"], d["b"])] = d
                if progress:
                    progress(len(out), len(pairs))
    else:
        for pr in pairs:
            d = one(pr)
            out[(d["a"], d["b"])] = d
            if progress:
                progress(len(out), len(pairs))
    return out


def rank_for(pid, people, dates):
    """ranking for person pid from their own perspective (60% own view, 40% the other's view)"""
    byid = {p["id"]: p for p in people}
    rows = []
    for o in people:
        if o["id"] == pid:
            continue
        key = (pid, o["id"]) if (pid, o["id"]) in dates else (o["id"], pid)
        d = dates.get(key)
        if not d:
            continue
        side = "a" if d["a"] == pid else "b"
        other_side = "b" if side == "a" else "a"
        mine, theirs = d["scores"][side + "_view"], d["scores"][other_side + "_view"]
        rows.append({
            "id": o["id"], "name": o["name"], "city": o["city"],
            "score": round(0.6 * mine + 0.4 * theirs), "your_view": mine, "their_view": theirs,
            "mutual": d["scores"]["mutual"],
            "label": label_for(round(0.6 * mine + 0.4 * theirs)),
            "why": d["reasons"][side]["why"], "watch": d["reasons"][side]["watch"],
            "shared_interests": d["shared_interests"],
        })
    rows.sort(key=lambda r: (-r["score"], r["name"]))
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows
