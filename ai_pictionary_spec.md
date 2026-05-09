# AI Pictionary (你畫我猜) — System Specification

**Status:** Design spec — implementation in a follow-up conversation.

---

## 1. Overview

A prompt-engineering competition for a CS camp. Each team tries to **recreate a reference image** by writing a prompt that Gemini renders into their attempt. The system scores how close the generated image is to the reference using CLIP visual similarity.

Teaching goal: students learn how to write precise, structured prompts (subject + environment + style).

---

## 2. Roles

| Role | Description |
|------|-------------|
| **Host** (關主) | Camp instructor running the game. Operates the back-end console: uploads questions, picks the active question, controls what appears on the projector. |
| **Participant** | Camp student. One of 3 sharing a single computer with their group. Discusses verbally with the other 2, agrees on a prompt, generates and submits one image per question. |
| **Projector** | Public display visible to the whole room. Shows the scoreboard and selected submissions. |

---

## 3. Team Structure (hard-coded for 2 teams)

- **2 teams**: Team A vs Team B.
- Each team = **4 groups × 3 students = 12 students per team**.
- Each group shares **1 computer**.
- Total devices: 8 group computers (4 for A, 4 for B) + 1 host laptop + 1 projector.
- Each group submits one prompt and one generated image per question.

---

## 4. Game Flow (per question)

1. **Host selects a question** (e.g., `Q1.png`) on the host console. The reference image becomes visible on every group's page simultaneously.
2. **Group writes a prompt** in the prompt textarea on their page.
3. **Group clicks Generate.** Backend calls the Gemini API with the prompt; the generated image appears in the image box on the same page. Groups can iterate: edit prompt, regenerate, freely — these are just drafts.
4. **Group clicks Submit.** This locks the current image as the official entry. The page becomes read-only and **shows the group's own score**. (One-shot; cannot re-submit for this question.)
5. **Scoreboard progress dots fill** as groups submit. **Public scores stay hidden.**
6. When all 8 groups have submitted, the **host clicks "show average"** to reveal team averages on the projector.
7. Host can click **"show pics"** for any single group to project that group's image + score to the whole room.
8. **Host advances to the next question**, which resets every group's page to the new question's empty state.

Pacing is **synchronized**: all 8 groups work on the same question at the same time, and only the host can advance.

---

## 5. Front-end — Group Interface

**URL:** one per group, e.g., `/team/A/group/2`. The group's single shared computer stays on this URL the whole game.

**Layout (matches wireframe):**

```
┌──────────────────────────────────────────────┐
│  Team A — Group 2                            │
│  Question:                                   │
│  ┌──────────────┐    ┌──────────────────┐    │
│  │              │    │ generated image  │    │
│  │   Q1.png     │    │ (after Generate) │    │
│  │ (reference)  │    └──────────────────┘    │
│  │              │    ┌──────────────────┐    │
│  └──────────────┘    │ prompt textarea  │    │
│                      └──────────────────┘    │
│                      [ Generate ] [ Submit ] │
└──────────────────────────────────────────────┘
```

**Page states:**

| State | What's shown | What's interactive |
|-------|--------------|--------------------|
| **Drafting** | Reference image, empty/edited prompt, last generated image (if any) | Prompt textarea, Generate, Submit |
| **Generating** | Same + loading spinner over the image box | Disabled while waiting for Gemini |
| **Submitted** | Reference image, final prompt, final generated image, **score: ___ %** | Nothing — read-only until host advances to next question |

**Notes:**
- Gemini API call happens **server-side** (API key stays on the host's laptop; not exposed to clients).
- After submit, only this group's page sees their score. Other groups and the projector do not.
- When the host advances to the next question, the page resets to **Drafting** state with the new reference image.

---

## 6. Front-end — Projector View

**URL:** dedicated, e.g., `/projector`. Displayed on the room's screen. Host controls what's shown.

### Mode A — Scoreboard (default)

```
┌──────────────────┬──────────────────┐
│     Team A       │     Team B       │
│  ● ● ○ ○         │  ● ● ● ○         │
│                  │                  │
│   ___ %          │   ___ %          │
└──────────────────┴──────────────────┘
```

- 4 dots per team representing the 4 groups; dot fills as soon as that group submits.
- Average team score (`%`) is **hidden until host clicks "show average"**, even after all groups have submitted.
- The "show average" reveal is a single global toggle — it reveals both teams' averages at once.

### Mode B — Single submission

```
┌─────────────────────────────────┐
│  Team A — Group 1               │
│  ┌─────────────┐                │
│  │             │   Score:       │
│  │   Image     │   ___ %        │
│  │             │                │
│  └─────────────┘                │
└─────────────────────────────────┘
```

- Triggered when host picks a group via "show pics".
- Shows that group's submitted image and its final score.

---

## 7. Back-end — Host Console

**URL:** `/host`. Used only by the instructor.

| Element | Function |
|---------|----------|
| **Question list** (`Q1.png, Q2.png, …`) | Click a row to set the active question. All group pages immediately switch to the new reference. |
| **`upload` button** | Upload a new question image. |
| **Stats grid** | Rows = questions, columns = teams (A, B). Each cell shows `uploaded: x/4` and per-group scores (visible to host even before public reveal). |
| **`show average` button** | Reveal team-average scores on the projector (Mode A). |
| **`show pics` button** | Pick any group's submission and project it (Mode B). |

**Example stats cell (host view):**
```
Q2.png — Team A:  uploaded 3/4
                  Group 1: 80%
                  Group 2: 90%
                  Group 3: --
                  Group 4: --
```

---

## 8. Scoring

```
score = round(clip_similarity × 100)
```

- Model: `openai/clip-vit-base-patch32` (already prototyped in `compare_images.py`).
- Pipeline: extract image features via `vision_model.pooler_output → visual_projection`, L2-normalize both vectors, compute cosine similarity, multiply by 100.
- **No remapping, no floor.** Raw similarity × 100. Unrelated images naturally land around 45–55% (CLIP's noise floor); decent attempts in the 70s; near-replicas in the 90s.
- Per-question score per group is stored once at submit time. There is no "best of N" — Submit is one-shot.

---

## 9. Data Model

```
Question
  id              ("Q1", "Q2", …)
  image_path      (reference image on disk)

Submission
  question_id
  team             ("A" | "B")
  group_number     (1–4)
  prompt           (final submitted prompt)
  image_path       (final generated image on disk)
  score            (0–100)
  submitted_at

GameState
  current_question_id
  projector_mode             ("scoreboard" | "single")
  projector_show_average     (bool)
  projector_target           ({team, group_number} or null, used in "single" mode)
```

Persistence: SQLite file in the project directory. Survives a host-laptop restart.

---

## 10. Suggested Tech Stack

One opinionated choice — change anything that doesn't fit.

| Layer | Choice | Why |
|-------|--------|-----|
| **Backend** | **FastAPI** (Python) | Matches your existing Python/CLIP code. Lightweight, well-documented, ergonomic for both REST endpoints and SSE. |
| **Frontend** | **Plain HTML + CSS + vanilla JS** (no framework, no build step) | The UI is small (3 pages, simple widgets). Avoids a React/Node toolchain you don't need. Each page is one `.html` file served by FastAPI. |
| **Real-time updates** | **Server-Sent Events (SSE)** | One-way push from server to clients (group submission → projector dots update; host clicks reveal → projector switches mode). Simpler than WebSockets and fits this app exactly — clients only listen, they don't push live data. |
| **Persistence** | **SQLite** (one `.db` file) | Zero setup. Survives crashes. Fine for 8 clients. |
| **Image generation** | **Google Gemini API**, called server-side | API key stays on the host laptop. Use the official `google-generativeai` Python SDK. |
| **Image scoring** | **Existing CLIP code** | Lift `calculate_similarity()` from `compare_images.py` essentially as-is. |
| **Networking** | All clients on the **same Wi-Fi**, pointing at the host laptop's local IP (e.g., `http://192.168.1.42:8000/team/A/group/2`) | No tunnels, no public URLs, no auth complexity. |

### Why not Gradio (the original prototype)

Gradio is great for one-page ML demos. This app has 3 distinct page types, live cross-page updates, and the host pushing state to the projector. That's a poor fit for Gradio's component model — you'd be fighting the framework. FastAPI + a few HTML files is simpler for this shape of app.

### Project layout sketch

```
ai-pictionary/
├── app.py                  # FastAPI app, routes, SSE, scoring, Gemini calls
├── db.py                   # SQLite helpers
├── clip_score.py           # Lifted from compare_images.py
├── templates/
│   ├── group.html          # /team/{X}/group/{N}
│   ├── projector.html      # /projector
│   └── host.html           # /host
├── static/
│   ├── style.css
│   └── app.js              # Tiny SSE listener + button handlers
├── data/
│   ├── questions/          # Reference images uploaded by host
│   ├── submissions/        # Generated images per group per question
│   └── game.db             # SQLite
└── .env                    # GEMINI_API_KEY
```

---

## 11. Implementation Notes (small but worth deciding once)

1. **Per-group URL gating.** Anyone on the Wi-Fi could open `/team/B/group/1` and submit on another team's behalf. For a friendly camp, a static per-group token in the URL (e.g., `/team/A/group/2?token=abc123`) is enough. Hand each group their URL on a printed slip.

2. **What "advance to next question" does.** Resets every group page from **Submitted** back to **Drafting** with the new reference image. Past submissions stay in the database (visible in the host stats grid) but the per-group page only shows the current question.

3. **Generated-image cleanup.** Saved at `data/submissions/{question_id}/{team}_{group}.png`. Overwritten each time the group regenerates pre-submit. After Submit, that file is the locked entry.

4. **Reference-image upload.** When uploading, the host names it (e.g., `Q1`) and the file is saved to `data/questions/Q1.png`.

5. **Score display rounding.** Round to integer for display (`74%`, not `73.8%`). Keep the float in the database in case you ever want finer analysis.

6. **Reveal idempotency.** "show average" is a toggle — host can hide it again. "show pics" replaces the current target group. Both update via SSE so the projector flips immediately.

7. **No authentication for host.** The `/host` page is just a URL. Host opens it on their laptop. Anyone who finds the URL on the LAN could press buttons. Acceptable for a camp; if you want, gate it with a basic-auth password or a secret URL prefix.
