"""Pre-run the whole demo and write static/demo.json (read-only snapshot of the finished example).

    python -m app.build_demo                 # scripted engine, offline
    python -m app.build_demo --claude 60     # Claude-written dates for the 60 most promising pairs, rest scripted
"""
import argparse
import json
import time
from pathlib import Path

from . import dating
from .analyzer import full_analysis
from .personas import PERSONAS

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--claude", type=int, default=0, help="number of top pairs to run with real Claude agents")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    people = [dict(p, analysis=full_analysis(p)) for p in PERSONAS]
    t = time.time()
    dates = dating.run_all(people)
    if args.claude:
        top = sorted(dates.values(), key=lambda d: -d["scores"]["mutual"])[: args.claude]
        byid = {p["id"]: p for p in people}
        from concurrent.futures import ThreadPoolExecutor
        def one(d):
            return dating.run_date(byid[d["a"]], byid[d["b"]], "claude")
        with ThreadPoolExecutor(args.workers) as ex:
            for nd in ex.map(one, top):
                dates[(nd["a"], nd["b"])] = nd
    rankings = {p["id"]: dating.rank_for(p["id"], people, dates) for p in people}
    engines = {}
    for d in dates.values():
        engines[d["engine"]] = engines.get(d["engine"], 0) + 1
    out = {
        "meta": {"people": len(people), "dates": len(dates), "engines": engines, "fictional": True,
                 "generated": time.strftime("%Y-%m-%d %H:%M"), "seconds": round(time.time() - t, 1)},
        "people": people,
        "dates": list(dates.values()),
        "rankings": rankings,
    }
    path = ROOT / "static" / "demo.json"
    path.write_text(json.dumps(out))
    print(f"wrote {path} ({path.stat().st_size // 1024} KB): {len(people)} people, {len(dates)} dates, engines={engines}")


if __name__ == "__main__":
    main()
