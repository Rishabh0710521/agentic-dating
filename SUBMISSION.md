# Submission pack

## Overall explanation (197 / 200)
Agentic dating: each person's agent reads their LinkedIn + Instagram text, builds an evidence-backed profile, dates every other agent, and ranks best fits. 25 fictional personas; real users opt in.

## Technical section (358 / 500)
No scraping: users paste their own LinkedIn/Instagram links plus profile text (consent required); links are validated and kept as references. Python/FastAPI; regex-lexicon analyzer extracts hobbies, values, needs with source evidence; optional Claude API agents hold multi-turn dates seeing only their own person's text. Vanilla JS UI, SSE, static demo.json.

## Links to fill in
- YouTube: (record, see shot list)
- Demo link: host `static/` (GitHub Pages / Netlify drop) — loads `demo.json`, no server needed
- Live website: deploy the Dockerfile (Render / Railway)
- GitHub: push this folder to a **public** repo

## 3-minute video shot list (profiles first, then rankings, with the agents actually dating)
Say on camera, once and early: "These 25 people are fictional demo personas; real people join only through the consent form."

| Time | Show | Say |
|---|---|---|
| 0:00-0:15 | Profiles grid | Every person is an agent. It reads two sources only: LinkedIn and Instagram. |
| 0:15-0:55 | Open one profile. Expand 2-3 evidence rows (a hobby, a stated need, an inferred need) | Needs, hobbies, values, qualities. Each item shows the exact line it came from, and whether the person *stated* it or the agent *inferred* it. |
| 0:55-1:50 | Agents date → pick one person → Cinematic speed (use Claude engine if your key is set) | Here is the agent dating on its person's behalf. Watch it ask about interests, values and needs, answer honestly when its profile has no evidence, and flag a deal-breaker. Each side gives its own verdict. |
| 1:50-2:00 | Switch to Turbo, everyone vs everyone | All 300 dates run. |
| 2:00-2:30 | Rankings for that person: why / watch-outs, open one date from it | Ranking blends my agent's verdict with theirs. |
| 2:30-2:45 | Matrix | Full picture, click any cell to read the date. |
| 2:45-3:00 | Add yourself: paste links + text, tick consent, profile appears | Real people opt in with their own links. |

Tip: record at 1280x800, browser zoom 110%. For a real-LLM dating shot, set `ANTHROPIC_API_KEY` first and choose "Claude agents".
