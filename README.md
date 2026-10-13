# agentic.dating

Every person is represented by an agent. The agents date each other. Each person gets a ranking of who fits them best.

An agent knows exactly two things about its person: their **LinkedIn** text and their **Instagram** text. Nothing else.

```
LinkedIn text + Instagram text
        ↓
   analyzer  →  profile page: hobbies, interests, values, needs, qualities (every item links to the line it came from)
        ↓
   agents date each other (multi-turn conversation, each side gives its own verdict)
        ↓
   per-person ranking + compatibility matrix
```

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env            # optional: add ANTHROPIC_API_KEY for Claude agents
uvicorn app.main:app --reload   # http://localhost:8000
pytest -q                       # 7 tests, no API key needed
```

Pre-run demo (already generated, read-only, needs no server): serve `static/` anywhere
(`cd static && python -m http.server`) and open `index.html`. It loads `demo.json`.

Regenerate the demo: `python -m app.build_demo`, or with real Claude-written dates for the 60 most promising pairs:
`python -m app.build_demo --claude 60`.

## What is real and what is not (read this)

| Part | Status |
|---|---|
| The 25 demo people | **Fictional.** Invented, labelled "fictional" everywhere, with LinkedIn-style and Instagram-style text written for the demo. Their links are placeholders, not real URLs. |
| Real people | Added through **Add yourself**: link + pasted profile text + a consent checkbox. Only add yourself or someone who agreed. |
| Scraping | **None.** The site never fetches LinkedIn or Instagram. Links are validated and stored as the source reference; the agent analyses the text pasted next to each link. |
| Analyzer | Deterministic lexicon/pattern analyzer. Each hobby, value and need keeps the exact source line (`LinkedIn about`, `Instagram caption`, ...). Needs and values are tagged **stated** (the person wrote it) or **inferred** (the agent concluded it; weighted lower when dating). With `ANTHROPIC_API_KEY` Claude adds a written read and extra qualities on top. |
| Dates, `scripted` engine | Turns are assembled from each agent's own analysis and evidence. Deterministic and instant, **not** an LLM conversation. The checked-in `demo.json` uses this engine (`engine` is stored on every date and shown in the UI). |
| Dates, `claude` engine | Every turn is a separate Claude call. Each agent sees only its own person's two sources plus the conversation so far, then returns a JSON verdict. Covered by a stubbed test; needs your key to run for real. |
| Scores | One deterministic model for both engines: shared interests 33%, needs met 27%, values 17%, city 15%, social energy 8%, minus 18 per deal-breaker clash. Calibrated to 25-100. In `claude` mode the agent's own verdict is blended in at 40%. |
| Ranking | For person X: 60% X's own agent's score + 40% the other agent's score, so being wanted back counts. |

## Layout

```
app/personas.py   25 fictional personas
app/analyzer.py   profile analysis with evidence
app/dating.py     scoring, scripted + Claude dates, rankings
app/main.py       FastAPI: intake, dates, SSE stream, rankings
app/build_demo.py pre-run the demo into static/demo.json
static/index.html the whole UI (profiles, live dates, rankings, matrix, add yourself)
tests/            pytest
```

## API

`GET /api/state` · `POST /api/people` · `DELETE /api/people/{id}` · `GET /api/date/{a}/{b}` ·
`GET /api/rankings[/{id}]` · `GET /api/stream?person=&engine=` (server-sent events) · `GET /api/export`

## Deploy

`Dockerfile` included (Render, Railway, Fly: set `ANTHROPIC_API_KEY` if you want Claude agents). The static demo can be hosted on GitHub Pages from the `static/` folder.
