# AI Pictionary (你畫我猜)

A prompt-engineering competition game for CS camp. Teams compete to recreate a reference image by writing prompts that Gemini renders into AI-generated images. Submissions are scored automatically using CLIP visual similarity.

---

## How It Works

1. The **host** uploads a reference image and activates it as the current question.
2. Each **group** sees the reference image, writes a prompt, and generates an image via Gemini (or uploads one manually).
3. Groups click **Submit** to lock in their image — scored instantly by CLIP.
4. The **projector** shows a live scoreboard with submission progress dots. The host reveals team averages when ready.

---

## Team Structure

- 2 teams: **Team A** and **Team B**
- Each team has 4 groups of 3 students (1 shared computer per group)
- 8 group computers + 1 host laptop + 1 projector

---

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```
> First run downloads the CLIP model (~1.7 GB). Do this before camp on good internet.

**2. Set your Gemini API key**

Copy `.env` and fill in your key from [Google AI Studio](https://aistudio.google.com/app/apikey):
```
GEMINI_API_KEY=your_key_here
```

**3. Start the server**
```bash
python app.py
```

---

## Pages

| Page | URL | Who uses it |
|------|-----|-------------|
| Host console | `http://HOST_IP:8000/host` | Instructor |
| Projector | `http://HOST_IP:8000/projector` | Room screen |
| Group A-1 | `http://HOST_IP:8000/team/A/group/1?token=a1-alpha` | Group A-1 |
| Group A-2 | `http://HOST_IP:8000/team/A/group/2?token=a2-bravo` | Group A-2 |
| Group A-3 | `http://HOST_IP:8000/team/A/group/3?token=a3-charlie` | Group A-3 |
| Group A-4 | `http://HOST_IP:8000/team/A/group/4?token=a4-delta` | Group A-4 |
| Group B-1 | `http://HOST_IP:8000/team/B/group/1?token=b1-echo` | Group B-1 |
| Group B-2 | `http://HOST_IP:8000/team/B/group/2?token=b2-foxtrot` | Group B-2 |
| Group B-3 | `http://HOST_IP:8000/team/B/group/3?token=b3-golf` | Group B-3 |
| Group B-4 | `http://HOST_IP:8000/team/B/group/4?token=b4-hotel` | Group B-4 |

Replace `HOST_IP` with your laptop's local IP (`ipconfig getifaddr en0`). All devices must be on the same Wi-Fi.

---

## Scoring

```
score = cosine_similarity(CLIP(reference), CLIP(submission)) × 100
```

- Model: `openai/clip-vit-base-patch32`
- Unrelated images score ~45–55%, good attempts ~70s, near-replicas ~90s
- Submit is one-shot — no resubmissions per question

---

## Host Controls

| Button | What it does |
|--------|-------------|
| **Activate** (next to a question) | Sets that question as active — all group pages update instantly |
| **Upload Question** | Upload a reference image with a question ID (e.g. `Q1`) |
| **Show Average** | Reveals team average scores on the projector |
| **↩ Scoreboard** | Returns projector to the scoreboard after showing a group's image |
| **Show Pics** (click a group's score) | Projects that group's submitted image and score |
| **🗑 Clear Submissions** | Wipes all submissions and generated images, keeps questions |

---

## Project Structure

```
├── app.py              # FastAPI backend — routes, SSE, Gemini calls, scoring
├── db.py               # SQLite helpers
├── clip_score.py       # CLIP image similarity scoring
├── templates/
│   ├── group.html      # Group page (/team/{X}/group/{N})
│   ├── projector.html  # Projector view (/projector)
│   └── host.html       # Host console (/host)
├── static/
│   ├── style.css
│   └── app.js
├── data/               # Created at runtime (gitignored)
│   ├── questions/      # Reference images uploaded by host
│   └── submissions/    # Generated images per group per question
├── prototype/          # Original Gradio prototype (reference only)
├── .env                # GEMINI_API_KEY (gitignored)
└── game.db             # SQLite database (gitignored)
```

---

## Testing Without a Gemini API Key

All features work without an API key except image generation. To test the full flow:

1. Start the server and open `/host`
2. Upload a test image as `Q1` and activate it
3. On a group page, use **Upload Image** to manually upload any image instead of generating
4. Click Submit — CLIP scoring, SSE updates, and the projector all work normally

---

## Further Improvements

### Setup & Integration
- [ ] Add Gemini API key and test prompt writing + image generation end-to-end
- [ ] Remove the manual upload button on group pages once image generation is confirmed working
- [ ] Add a generation prompt field per question when uploading (to store the prompt used to create the reference image)
- [ ] Prepare lots of questions (reference images + their generation prompts)

### Host Controls
- [ ] Show reference image + the prompt used to generate it (host page)
- [ ] Add final results screen showing total team scores after all questions — either a dedicated projector Mode C or a cumulative scoreboard (TBD)

### Projector & Scoreboard UX
- [ ] Make images and icons on the projector larger
- [ ] Change team colors to blue (Team A) and red (Team B) — group page headers and scoreboard dots
- [ ] Dramatic score reveal animation with larger score text when host clicks "Show Average"

### Projector — Show Pics Enhancement
- [ ] When projecting a group's submission, also show the prompt they used — so the whole room can see what they typed
