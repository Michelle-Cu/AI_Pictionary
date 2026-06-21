# AI 你畫我猜

A prompt-engineering competition game for CS camp. Teams compete to recreate a reference image by writing prompts that Gemini renders into AI-generated images. Submissions are scored automatically using a blended CLIP + DINO visual similarity score.

---

## How It Works

1. The **host** uploads a reference image and activates it as the current question.
2. Each **group** sees the reference image, writes a prompt, and generates an image via Gemini (or uploads one manually). Up to 3 attempts per question.
3. Groups click **Submit** to lock in one of their drafts — scored instantly.
4. The **projector** shows a live scoreboard with submission progress dots. The host reveals team averages and individual submissions when ready.

---

## Team Structure

- 2 teams: **Team A** and **Team B**
- Each team has 4 groups (1 shared computer per group)
- 8 group computers + 1 host laptop + 1 projector

---

## Setup

### Option A — Docker (recommended for deployment)

**1. Install Docker** on the workstation ([get.docker.com](https://get.docker.com) for Linux, Docker Desktop for Mac/Windows)

**2. Fill in `.env`**
```
OPENROUTER_API_KEY=your_key_here
APP_PASSWORD=your_game_password
SECRET_KEY=run: python -c "import secrets; print(secrets.token_hex(32))"
```

**3. Start**
```bash
docker compose up --build
```

First run downloads the CPU torch wheel + CLIP/DINO model weights (~1.4 GB total). Subsequent starts are fast — weights are cached in a Docker volume.

**4. Stop**
```bash
docker compose down      # keeps data
docker compose down -v   # wipes everything including DB and images
```

---

### Option B — Local (development)

**1. Install dependencies**
```bash
pip install -r requirements.txt
```
> First run downloads CLIP and DINO model weights (~600 MB). Do this before camp on good internet.

**2. Fill in `.env`**
```
OPENROUTER_API_KEY=your_key_here
APP_PASSWORD=your_game_password
SECRET_KEY=any_random_string
```

**3. Start**
```bash
python app.py
```

---

## Pages

All pages require a password login. After logging in, the session is remembered via a cookie.

| Page | URL | Who uses it |
|------|-----|-------------|
| Login | `http://HOST_IP:10060/login` | Everyone |
| Host console | `http://HOST_IP:10060/host` | Instructor |
| Projector | `http://HOST_IP:10060/projector` | Room screen |
| Group A-1 | `http://HOST_IP:10060/team/A/group/1?token=a1-alpha` | Group A-1 |
| Group A-2 | `http://HOST_IP:10060/team/A/group/2?token=a2-bravo` | Group A-2 |
| Group A-3 | `http://HOST_IP:10060/team/A/group/3?token=a3-charlie` | Group A-3 |
| Group A-4 | `http://HOST_IP:10060/team/A/group/4?token=a4-delta` | Group A-4 |
| Group B-1 | `http://HOST_IP:10060/team/B/group/1?token=b1-echo` | Group B-1 |
| Group B-2 | `http://HOST_IP:10060/team/B/group/2?token=b2-foxtrot` | Group B-2 |
| Group B-3 | `http://HOST_IP:10060/team/B/group/3?token=b3-golf` | Group B-3 |
| Group B-4 | `http://HOST_IP:10060/team/B/group/4?token=b4-hotel` | Group B-4 |

> For local development the port is `8000`. Replace `HOST_IP` with the workstation's local IP (`ip addr` on Linux, `ipconfig getifaddr en0` on Mac). All devices must be on the same network.

To log out: visit `/logout`.

---

## Scoring

Submissions are scored using a blended CLIP + DINO similarity against the reference image, then passed through a sigmoid curve to spread scores:

```
clip_sim  = cosine_similarity(CLIP(ref), CLIP(submission))
dino_sim  = cosine_similarity(DINO(ref), DINO(submission))
blended   = 0.7 × clip_sim + 0.3 × dino_sim
score     = sigmoid(12 × (blended − 0.5)) × 100
```

- **CLIP** (`openai/clip-vit-base-patch32`) captures semantic meaning
- **DINO** (`facebook/dinov2-base`) captures structural/visual layout
- The sigmoid (k=12) spreads mid-range scores and compresses extremes
- Unrelated images score ~45–55%, good attempts ~70s, near-replicas ~90s
- Submit is one-shot — no resubmissions once locked in

---

## Host Controls

| Button | What it does |
|--------|-------------|
| **Upload Question** | Upload a reference image with a question ID (e.g. `Q1`) |
| **Activate** | Sets that question as active — all group pages update instantly via SSE |
| **Show Average** | Reveals team average scores on the projector |
| **Show Pics** (click a group's score) | Projects that group's submitted image and score |
| **↩ Scoreboard** | Returns projector to the scoreboard |
| **🗑 Clear Submissions** | Wipes all submissions and generated images for the current question, keeps reference images |

---

## Project Structure

```
├── app.py              # FastAPI backend — routes, SSE, OpenRouter calls, scoring
├── db.py               # SQLite helpers (DB stored at data/game.db)
├── clip_score.py       # Blended CLIP + DINO image similarity scoring
├── templates/
│   ├── login.html      # Password login page
│   ├── group.html      # Group page (/team/{X}/group/{N})
│   ├── projector.html  # Projector view (/projector)
│   └── host.html       # Host console (/host)
├── static/
│   ├── style.css
│   └── app.js
├── data/               # Created at runtime — mounted as Docker volume (gitignored)
│   ├── game.db         # SQLite database
│   ├── questions/      # Reference images uploaded by host
│   └── submissions/    # Generated/uploaded images per group per question
├── Dockerfile
├── docker-compose.yml
├── prototype/          # Original Gradio prototype (reference only)
└── .env                # Secrets (gitignored)
```

---

## Further Improvements

### Host Controls
- [ ] Show reference image + the prompt used to generate it (host page)
- [ ] Add final results screen showing total team scores after all questions

### Projector & Scoreboard UX
- [ ] Make images and icons on the projector larger
- [ ] Change team colors to blue (Team A) and red (Team B)
- [ ] Dramatic score reveal animation when host clicks "Show Average"
- [ ] When projecting a group's submission, also show the prompt they used
