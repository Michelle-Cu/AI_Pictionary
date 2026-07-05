# AI 你畫我猜

Teams compete to recreate a reference image by writing prompts that Gemini renders into AI-generated images. Submissions are scored automatically using a blended CLIP + DINO + LPIPS visual similarity score.

---

## How It Works

1. The **host** activates a reference image as the current question.
2. Each **group** sees the reference image, writes a prompt, and generates an image via Gemini. Up to 3 draft attempts per question.
3. Groups select one draft and click **Submit** to lock it in — scored instantly.
4. The **projector** shows a live scoreboard with submission progress dots. The host can reveal team averages and project individual submissions.

---

## Team Structure

- 2 teams: **Team A** and **Team B**
- Each team has 4 groups (1 shared computer per group)
- 8 group computers + 1 host laptop + 1 projector screen

---

## Setup

### 1. Clone and configure

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

Edit `.env`:
```
OPENROUTER_API_KEY=your_openrouter_api_key_here
APP_PASSWORD=your_host_console_password
SECRET_KEY=run: python -c "import secrets; print(secrets.token_hex(32))"
```

> `APP_PASSWORD` is the host-only login. Group passwords are set separately in `app.py` (see below).

---

### 2. Set group passwords

Open `app.py` and edit `GROUP_CREDENTIALS` near the top of the file:

```python
GROUP_CREDENTIALS: dict[str, tuple[str, int, str]] = {
    "A1": ("A", 1, "your_password_here"),
    "A2": ("A", 2, "your_password_here"),
    "A3": ("A", 3, "your_password_here"),
    "A4": ("A", 4, "your_password_here"),
    "B1": ("B", 1, "your_password_here"),
    "B2": ("B", 2, "your_password_here"),
    "B3": ("B", 3, "your_password_here"),
    "B4": ("B", 4, "your_password_here"),
}
```

Each entry maps a username (e.g. `"A1"`) to `(team, group_number, password)`.

---

### Option A — Docker (recommended for deployment)

**1. Install Docker** ([get.docker.com](https://get.docker.com) for Linux, Docker Desktop for Mac/Windows)

**2. Start**
```bash
docker compose up --build
```

First run downloads the CLIP/DINO model weights and TensorFlow (~1.5 GB total). Subsequent starts are fast — weights are cached in a Docker volume.

**3. Stop**
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

> First run downloads CLIP, DINO, and LPIPS model weights. Do this before camp on good internet.

**2. Start**
```bash
python app.py
```

---

## Pages

All pages are served under the `/AI-Pictionary` base path.

| Page | URL | Who uses it |
|------|-----|-------------|
| Host login | `http://HOST_IP:10060/AI-Pictionary/login` | Instructor |
| Host console | `http://HOST_IP:10060/AI-Pictionary/host` | Instructor |
| Projector | `http://HOST_IP:10060/AI-Pictionary/projector` | Room screen |
| Group login | `http://HOST_IP:10060/AI-Pictionary/group-login` | All groups |
| Group page | `http://HOST_IP:10060/AI-Pictionary/team/{A or B}/group/{1–4}` | Auto-redirect after login |

> For local development the port is `8000`. Replace `HOST_IP` with the workstation's local IP (`ip addr` on Linux, `ipconfig getifaddr en0` on Mac). All devices must be on the same network.

Groups navigate to the **group login page**, enter their username (e.g. `A1`) and password, and are redirected to their group page automatically.

---

## Scoring

Submissions are scored using a weighted blend of three metrics, then passed through a sigmoid + power curve:

```
clip_sim   = cosine_similarity(CLIP(ref), CLIP(submission))
dino_sim   = cosine_similarity(DINO(ref), DINO(submission))
lpips_sim  = 1 - LPIPS_distance(ref, submission)

blended    = 0.40 × clip_sim + 0.45 × dino_sim + 0.15 × lpips_sim
curve      = sigmoid(k × (blended − midpoint)) ^ power × 100
```

- **CLIP** (`openai/clip-vit-base-patch32`) — semantic meaning
- **DINO** (`facebook/dinov2-base`) — structural / visual layout
- **LPIPS** (AlexNet, TF1) — perceptual pixel-level similarity

### Per-question tuning

All scoring parameters can be adjusted per question in `scoring_config.json` without restarting the server:

```json
{
  "default": {
    "weight_clip": 0.40,
    "weight_dino": 0.45,
    "weight_lpips": 0.15,
    "sigmoid_k": 12.0,
    "sigmoid_midpoint": 0.78,
    "power": 2.0
  },
  "Q3": {
    "sigmoid_k": 8.0,
    "sigmoid_midpoint": 0.75
  }
}
```

Any key in a question's block overrides the default; missing keys fall back to `default`.

| Parameter | Effect |
|-----------|--------|
| `weight_clip/dino/lpips` | Relative importance of each metric (should sum to 1.0) |
| `sigmoid_k` | Steepness of S-curve — higher = sharper pass/fail cutoff |
| `sigmoid_midpoint` | Center of the curve — raise toward 0.75–0.80 so real image scores spread evenly |
| `power` | Exponent after sigmoid — higher values punish low scores more |

---

## Host Controls

| Button | What it does |
|--------|-------------|
| **Upload Question** | Upload a reference image with a question ID (e.g. `Q1`) |
| **✕** (next to question) | Delete that question and its image file |
| **Activate** | Sets that question as active — all group pages update instantly |
| **⏳ Pending Screen** | Clears all group pages (hides reference image before revealing the next question) |
| **Show Average** | Reveals team average scores on the projector |
| **Click a group's score** | Projects that group's submitted image, prompt, and score |
| **↩ Scoreboard** | Returns projector to the scoreboard |
| **🗑 Clear Submissions** | Wipes all submissions and generated images, keeps reference images |
| **🔒 Logout All Groups** | Invalidates all group sessions — groups must log in again |
| **Host Logout** | Logs out the host session |

---

## Project Structure

```
├── app.py                  # FastAPI backend — routes, SSE, OpenRouter calls
├── db.py                   # SQLite helpers (DB at data/game.db)
├── clip_score.py           # CLIP + DINO + LPIPS scoring
├── scoring_config.json     # Per-question scoring parameters
├── templates/
│   ├── login.html          # Host login page
│   ├── group_login.html    # Group login page
│   ├── group.html          # Group page (/AI-Pictionary/team/{X}/group/{N})
│   ├── projector.html      # Projector view (/AI-Pictionary/projector)
│   └── host.html           # Host console (/AI-Pictionary/host)
├── static/
│   ├── style.css
│   └── app.js
├── utils/
│   └── lpips/
│       ├── lpips_tf.py     # TF1-based LPIPS implementation
│       └── net-lin_alex_v0.1.pb  # Auto-downloaded on first run
├── data/
│   ├── questions/          # Reference images (committed to repo)
│   └── submissions/        # Generated images per group (gitignored)
├── Dockerfile
├── docker-compose.yml
├── .env                    # Secrets (gitignored — copy from .env.example)
└── .env.example            # Template for required environment variables
```
