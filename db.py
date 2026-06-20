import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

DB_PATH = Path("data/game.db")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS questions (
                id          TEXT PRIMARY KEY,
                image_path  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS submissions (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id   TEXT    NOT NULL,
                team          TEXT    NOT NULL,
                group_number  INTEGER NOT NULL,
                prompt        TEXT    NOT NULL,
                image_path    TEXT    NOT NULL,
                score         REAL    NOT NULL,
                submitted_at  TEXT    NOT NULL,
                UNIQUE(question_id, team, group_number)
            );

            CREATE TABLE IF NOT EXISTS game_state (
                id                      INTEGER PRIMARY KEY CHECK(id = 1),
                current_question_id     TEXT,
                projector_mode          TEXT    NOT NULL DEFAULT 'scoreboard',
                projector_show_average  INTEGER NOT NULL DEFAULT 0,
                projector_target        TEXT
            );

            INSERT OR IGNORE INTO game_state(id) VALUES(1);
        """)


def get_game_state() -> dict:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM game_state WHERE id = 1").fetchone()
        if not row:
            return {}
        d = dict(row)
        d["projector_show_average"] = bool(d["projector_show_average"])
        if d.get("projector_target"):
            d["projector_target"] = json.loads(d["projector_target"])
        return d


def set_game_state(**kwargs):
    if "projector_target" in kwargs and kwargs["projector_target"] is not None:
        kwargs["projector_target"] = json.dumps(kwargs["projector_target"])
    sets = ", ".join(f"{k} = ?" for k in kwargs)
    vals = list(kwargs.values())
    with _connect() as conn:
        conn.execute(f"UPDATE game_state SET {sets} WHERE id = 1", vals)


def add_question(qid: str, image_path: str):
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO questions(id, image_path) VALUES(?, ?)",
            (qid, image_path),
        )


def list_questions() -> list[dict]:
    with _connect() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM questions ORDER BY id")]


def get_question(qid: str) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM questions WHERE id = ?", (qid,)).fetchone()
        return dict(row) if row else None


def save_submission(qid: str, team: str, group_number: int,
                    prompt: str, image_path: str, score: float):
    with _connect() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO submissions
               (question_id, team, group_number, prompt, image_path, score, submitted_at)
               VALUES(?, ?, ?, ?, ?, ?, ?)""",
            (qid, team, group_number, prompt, image_path, score,
             datetime.now(timezone.utc).isoformat()),
        )


def get_submission(qid: str, team: str, group_number: int) -> Optional[dict]:
    with _connect() as conn:
        row = conn.execute(
            """SELECT * FROM submissions
               WHERE question_id=? AND team=? AND group_number=?""",
            (qid, team, group_number),
        ).fetchone()
        return dict(row) if row else None


def get_submissions_for_question(qid: str) -> list[dict]:
    with _connect() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM submissions WHERE question_id=?", (qid,)
        )]


def get_all_submissions() -> list[dict]:
    with _connect() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM submissions")]
