import asyncio
import io
import json
import os
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from itsdangerous import BadSignature, URLSafeSerializer
from PIL import Image
from pydantic import BaseModel

import clip_score
import db

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")
_signer = URLSafeSerializer(os.getenv("SECRET_KEY", "change-me-in-production"), salt="session")


def _is_authenticated(request: Request) -> bool:
    token = request.cookies.get("session")
    if not token:
        return False
    try:
        return _signer.loads(token) == "authenticated"
    except BadSignature:
        return False
TEAMS = ["A", "B"]
GROUPS = list(range(1, 5))

# Static per-group URL tokens — hand each group their URL on a printed slip.
GROUP_TOKENS: dict[tuple[str, int], str] = {
    ("A", 1): "a1-alpha",
    ("A", 2): "a2-bravo",
    ("A", 3): "a3-charlie",
    ("A", 4): "a4-delta",
    ("B", 1): "b1-echo",
    ("B", 2): "b2-foxtrot",
    ("B", 3): "b3-golf",
    ("B", 4): "b4-hotel",
}

_sse_queues: list[asyncio.Queue] = []


async def broadcast(event_type: str, data: dict):
    payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    for q in _sse_queues:
        await q.put(payload)


def _path_to_url(db_path: str) -> str:
    """Convert a data/... path stored in DB to a /images/... URL."""
    p = db_path.replace("\\", "/")
    if p.startswith("data/"):
        return "/images/" + p[5:]
    return "/images/" + p


def _build_full_state() -> dict:
    """Public state payload for the SSE init event and /api/state."""
    state = db.get_game_state()
    qid = state.get("current_question_id")
    submitted: dict[str, bool] = {}
    averages: dict[str, int] = {}
    projector_image_url = None
    projector_score = None

    if qid:
        subs = db.get_submissions_for_question(qid)
        for s in subs:
            submitted[f"{s['team']}{s['group_number']}"] = True

        if state.get("projector_show_average"):
            for team in TEAMS:
                ts = [s for s in subs if s["team"] == team]
                if ts:
                    averages[team] = round(sum(s["score"] for s in ts) / len(ts))

        if state.get("projector_mode") == "single" and state.get("projector_target"):
            t = state["projector_target"]
            sub = db.get_submission(qid, t["team"], t["group_number"])
            if sub:
                projector_image_url = _path_to_url(sub["image_path"])
                projector_score = round(sub["score"])

    state["submitted"] = submitted
    state["averages"] = averages
    state["projector_image_url"] = projector_image_url
    state["projector_score"] = projector_score
    return state


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()
    clip_score.load_models()
    Path("data/questions").mkdir(parents=True, exist_ok=True)
    Path("data/submissions").mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory="data"), name="images")
templates = Jinja2Templates(directory="templates")


# ── SSE ───────────────────────────────────────────────────────────────────────

@app.get("/events")
async def sse_endpoint(request: Request):
    queue: asyncio.Queue = asyncio.Queue()

    async def stream():
        _sse_queues.append(queue)
        try:
            state = _build_full_state()
            yield f"event: init\ndata: {json.dumps(state)}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(queue.get(), timeout=20)
                    yield msg
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            if queue in _sse_queues:
                _sse_queues.remove(queue)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Auth ──────────────────────────────────────────────────────────────────────

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, error: str = ""):
    return templates.TemplateResponse(request, "login.html", {"error": error})


@app.post("/login")
async def login_submit(request: Request, password: str = Form(...)):
    if password == APP_PASSWORD:
        token = _signer.dumps("authenticated")
        response = RedirectResponse("/host", status_code=303)
        response.set_cookie("session", token, httponly=True, samesite="lax")
        return response
    return RedirectResponse("/login?error=1", status_code=303)


@app.get("/logout")
async def logout():
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie("session")
    return response


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.get("/team/{team}/group/{group_number}", response_class=HTMLResponse)
async def group_page(request: Request, team: str, group_number: int, token: str = ""):
    if not _is_authenticated(request):
        return RedirectResponse("/login", status_code=303)
    team = team.upper()
    if team not in TEAMS or group_number not in GROUPS:
        raise HTTPException(404, "Unknown team or group")
    expected = GROUP_TOKENS.get((team, group_number), "")
    if expected and token != expected:
        raise HTTPException(403, "Invalid token. Check your printed URL slip.")
    return templates.TemplateResponse(
        request, "group.html",
        {"team": team, "group_number": group_number, "token": token},
    )


@app.get("/projector", response_class=HTMLResponse)
async def projector_page(request: Request):
    if not _is_authenticated(request):
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(request, "projector.html", {})


@app.get("/host", response_class=HTMLResponse)
async def host_page(request: Request):
    if not _is_authenticated(request):
        return RedirectResponse("/login", status_code=303)
    return templates.TemplateResponse(
        request, "host.html",
        {"group_tokens": GROUP_TOKENS, "teams": TEAMS, "groups": GROUPS},
    )


# ── Group API ─────────────────────────────────────────────────────────────────

@app.get("/api/group-state/{team}/{group_number}")
async def api_group_state(team: str, group_number: int):
    team = team.upper()
    gs = db.get_game_state()
    result: dict = {"game_state": gs, "question": None, "submission": None, "drafts": []}
    if gs and gs.get("current_question_id"):
        qid = gs["current_question_id"]
        result["question"] = db.get_question(qid)
        sub = db.get_submission(qid, team, group_number)
        if sub:
            sub = dict(sub)
            sub["image_url"] = _path_to_url(sub["image_path"])
        result["submission"] = sub
        for v in range(1, 4):
            p = Path(f"data/submissions/{qid}/{team}_{group_number}_draft_v{v}.png")
            if p.exists():
                result["drafts"].append({
                    "version": v,
                    "image_url": f"/images/submissions/{qid}/{team}_{group_number}_draft_v{v}.png",
                })
    return result


class GenerateReq(BaseModel):
    team: str
    group_number: int
    prompt: str
    token: str = ""


@app.post("/api/upload-image")
async def api_upload_image(
    team: str = Form(...),
    group_number: int = Form(...),
    token: str = Form(""),
    file: UploadFile = File(...),
):
    team = team.upper()
    _check_group(team, group_number, token)

    gs = db.get_game_state()
    if not gs or not gs.get("current_question_id"):
        raise HTTPException(400, "No active question — wait for the host to start one.")

    qid = gs["current_question_id"]
    if db.get_submission(qid, team, group_number):
        raise HTTPException(400, "Already submitted for this question.")

    draft_dir = Path(f"data/submissions/{qid}")
    draft_dir.mkdir(parents=True, exist_ok=True)
    existing = list(draft_dir.glob(f"{team}_{group_number}_draft_v*.png"))
    if len(existing) >= 3:
        raise HTTPException(400, "Maximum 3 uploads reached for this question.")
    version = len(existing) + 1

    out = draft_dir / f"{team}_{group_number}_draft_v{version}.png"
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    out.write_bytes(buf.getvalue())

    return {
        "image_url": f"/images/submissions/{qid}/{team}_{group_number}_draft_v{version}.png",
        "version": version,
        "total_drafts": version,
    }


@app.post("/api/generate")
async def api_generate(req: GenerateReq):
    team = req.team.upper()
    _check_group(team, req.group_number, req.token)

    gs = db.get_game_state()
    if not gs or not gs.get("current_question_id"):
        raise HTTPException(400, "No active question — wait for the host to start one.")

    qid = gs["current_question_id"]
    if db.get_submission(qid, team, req.group_number):
        raise HTTPException(400, "Already submitted for this question.")

    draft_dir = Path(f"data/submissions/{qid}")
    draft_dir.mkdir(parents=True, exist_ok=True)
    existing = list(draft_dir.glob(f"{team}_{req.group_number}_draft_v*.png"))
    if len(existing) >= 3:
        raise HTTPException(400, "Maximum 3 generations reached for this question.")
    version = len(existing) + 1

    if not OPENROUTER_API_KEY:
        raise HTTPException(500, "OPENROUTER_API_KEY is not set on the server.")

    img_bytes = await _gemini_generate(req.prompt)

    out = draft_dir / f"{team}_{req.group_number}_draft_v{version}.png"
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    out.write_bytes(buf.getvalue())

    return {
        "image_url": f"/images/submissions/{qid}/{team}_{req.group_number}_draft_v{version}.png",
        "version": version,
        "total_drafts": version,
    }


class SubmitReq(BaseModel):
    team: str
    group_number: int
    prompt: str
    token: str = ""
    version: int = 1


@app.post("/api/submit")
async def api_submit(req: SubmitReq):
    team = req.team.upper()
    _check_group(team, req.group_number, req.token)

    gs = db.get_game_state()
    if not gs or not gs.get("current_question_id"):
        raise HTTPException(400, "No active question.")

    qid = gs["current_question_id"]
    if db.get_submission(qid, team, req.group_number):
        raise HTTPException(400, "Already submitted — one submission per question.")

    draft = Path(f"data/submissions/{qid}/{team}_{req.group_number}_draft_v{req.version}.png")
    if not draft.exists():
        raise HTTPException(400, f"Draft v{req.version} not found. Please generate an image first.")

    question = db.get_question(qid)
    if not question:
        raise HTTPException(400, "Active question not found in database.")

    score = clip_score.calculate_score(question["image_path"], str(draft))
    db.save_submission(qid, team, req.group_number, req.prompt, str(draft), score)

    await broadcast("submission", {
        "team": team,
        "group_number": req.group_number,
        "question_id": qid,
    })
    return {"score": score}


# ── Host API ──────────────────────────────────────────────────────────────────

@app.post("/api/host/upload")
async def api_upload(question_id: str = Form(...), file: UploadFile = File(...)):
    qid = question_id.strip()
    if not qid:
        raise HTTPException(400, "question_id is required.")
    path = Path(f"data/questions/{qid}.png")

    # Normalize to PNG
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    path.write_bytes(buf.getvalue())

    db.add_question(qid, str(path))
    await broadcast("questions_updated", {"question_id": qid})
    return {"question_id": qid, "image_path": str(path)}


@app.post("/api/host/set-question")
async def api_set_question(body: dict):
    qid = body.get("question_id", "").strip()
    if not db.get_question(qid):
        raise HTTPException(404, f"Question '{qid}' not found.")
    db.set_game_state(
        current_question_id=qid,
        projector_mode="scoreboard",
        projector_show_average=False,
        projector_target=None,
    )
    await broadcast("question_changed", {"question_id": qid})
    return {"ok": True}


@app.post("/api/host/show-average")
async def api_show_average(body: dict):
    show = bool(body.get("show", True))
    db.set_game_state(projector_show_average=show)

    gs = db.get_game_state()
    qid = gs.get("current_question_id")
    averages: dict[str, int] = {}
    if qid:
        subs = db.get_submissions_for_question(qid)
        for team in TEAMS:
            ts = [s for s in subs if s["team"] == team]
            if ts:
                averages[team] = round(sum(s["score"] for s in ts) / len(ts))

    await broadcast("show_average", {"show": show, "averages": averages})
    return {"ok": True, "averages": averages}


@app.post("/api/host/show-pics")
async def api_show_pics(body: dict):
    team = body.get("team", "").upper()
    gn = body.get("group_number")

    if team and gn:
        gn = int(gn)
        db.set_game_state(
            projector_mode="single",
            projector_target={"team": team, "group_number": gn},
        )
        gs = db.get_game_state()
        qid = gs.get("current_question_id")
        sub = db.get_submission(qid, team, gn) if qid else None
        await broadcast("show_pics", {
            "team": team,
            "group_number": gn,
            "image_url": _path_to_url(sub["image_path"]) if sub else None,
            "score": round(sub["score"]) if sub else None,
        })
    else:
        db.set_game_state(projector_mode="scoreboard", projector_target=None)
        await broadcast("show_scoreboard", {})

    return {"ok": True}


@app.post("/api/host/clear-submissions")
async def api_clear_submissions():
    with db._connect() as conn:
        conn.execute("DELETE FROM submissions")
    db.set_game_state(
        projector_mode="scoreboard",
        projector_show_average=False,
        projector_target=None,
    )
    shutil.rmtree("data/submissions", ignore_errors=True)
    Path("data/submissions").mkdir(parents=True, exist_ok=True)
    await broadcast("question_changed", {"question_id": db.get_game_state().get("current_question_id", "")})
    return {"ok": True}


@app.get("/api/questions")
async def api_questions():
    return db.list_questions()


@app.get("/api/state")
async def api_state():
    return _build_full_state()


@app.get("/api/host/stats")
async def api_host_stats():
    questions = db.list_questions()
    all_subs = db.get_all_submissions()
    sub_map = {(s["question_id"], s["team"], s["group_number"]): s for s in all_subs}

    result = []
    for q in questions:
        qdata: dict = {"question_id": q["id"], "image_url": _path_to_url(q["image_path"]), "teams": {}}
        for team in TEAMS:
            groups: dict[str, dict] = {}
            for g in GROUPS:
                sub = sub_map.get((q["id"], team, g))
                draft_urls = [
                    f"/images/submissions/{q['id']}/{team}_{g}_draft_v{v}.png"
                    for v in range(1, 4)
                    if Path(f"data/submissions/{q['id']}/{team}_{g}_draft_v{v}.png").exists()
                ]
                groups[str(g)] = {
                    "submitted": sub is not None,
                    "score": round(sub["score"]) if sub else None,
                    "prompt": sub["prompt"] if sub else None,
                    "image_url": _path_to_url(sub["image_path"]) if sub else None,
                    "draft_urls": draft_urls,
                }
            count = sum(1 for g in GROUPS if sub_map.get((q["id"], team, g)))
            avg = None
            if count > 0:
                ts = [sub_map[(q["id"], team, g)] for g in GROUPS if sub_map.get((q["id"], team, g))]
                avg = round(sum(s["score"] for s in ts) / len(ts))
            qdata["teams"][team] = {"groups": groups, "count": count, "avg": avg}
        result.append(qdata)
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_group(team: str, group_number: int, token: str):
    if team not in TEAMS or group_number not in GROUPS:
        raise HTTPException(400, "Invalid team or group number.")
    expected = GROUP_TOKENS.get((team, group_number), "")
    if expected and token != expected:
        raise HTTPException(403, "Invalid token.")


# async def _gemini_generate(prompt: str) -> bytes:
#     try:
#         import base64
#         from openai import AsyncOpenAI

#         client = AsyncOpenAI(
#             api_key=OPENROUTER_API_KEY,
#             base_url="https://openrouter.ai/api/v1",
#         )
#         response = await client.chat.completions.create(
#             model="google/gemini-2.5-flash-image",
#             messages=[{"role": "user", "content": prompt}],
#         )
#         print(f"[DEBUG] Full response: {response.model_dump()}")
#         import httpx
#         for choice in response.choices:
#             content = choice.message.content
#             print(f"[DEBUG] OpenRouter response content type: {type(content)}")
#             print(f"[DEBUG] OpenRouter response content: {repr(content)[:500]}")
#             if not content:
#                 continue
#             if isinstance(content, str) and content.startswith("data:image"):
#                 header, b64 = content.split(",", 1)
#                 return base64.b64decode(b64)
#             if isinstance(content, str) and content.startswith("http"):
#                 async with httpx.AsyncClient() as hclient:
#                     r = await hclient.get(content)
#                     r.raise_for_status()
#                     return r.content
#             if isinstance(content, list):
#                 for part in content:
#                     if isinstance(part, dict) and part.get("type") == "image_url":
#                         url = part["image_url"]["url"]
#                         if url.startswith("data:image"):
#                             header, b64 = url.split(",", 1)
#                             return base64.b64decode(b64)
#                         if url.startswith("http"):
#                             async with httpx.AsyncClient() as hclient:
#                                 r = await hclient.get(url)
#                                 r.raise_for_status()
#                                 return r.content
#         raise ValueError("No image data found in OpenRouter response.")
#     except HTTPException:
#         raise
#     except Exception as exc:
#         raise HTTPException(500, f"OpenRouter error: {exc}") from exc

async def _gemini_generate(prompt: str) -> bytes:
    import base64
    import httpx
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "google/gemini-2.5-flash-image",
        "messages": [{"role": "user", "content": prompt}],
        # 1. Explicitly request image generation
        "modalities": ["image", "text"] 
    }
    
    try:
        # Make a direct request to avoid OpenAI SDK schema stripping
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
            
        # 2. Extract the OpenRouter-specific `images` array
        for choice in data.get("choices", []):
            message = choice.get("message", {})
            images = message.get("images", [])
            
            for img in images:
                # APIs sometimes camelCase or snake_case this property
                url_obj = img.get("image_url") or img.get("imageUrl")
                if not url_obj:
                    continue
                
                url = url_obj.get("url", "")
                
                # Parse Base64
                if url.startswith("data:image"):
                    _, b64 = url.split(",", 1)
                    return base64.b64decode(b64)
                    
                # Parse standard HTTP URL 
                elif url.startswith("http"):
                    async with httpx.AsyncClient() as hclient:
                        r = await hclient.get(url)
                        r.raise_for_status()
                        return r.content
                        
        raise ValueError("No image data found in OpenRouter response.")
        
    except httpx.HTTPStatusError as exc:
        raise HTTPException(500, f"OpenRouter API error: {exc.response.text}") from exc
    except Exception as exc:
        raise HTTPException(500, f"Error generating image: {exc}") from exc


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
