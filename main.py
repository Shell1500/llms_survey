import csv
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates
from motor.motor_asyncio import AsyncIOMotorClient
from starlette.status import HTTP_303_SEE_OTHER


load_dotenv()


class Settings:
    """Centralized runtime configuration."""

    def __init__(self) -> None:
        self.mongo_uri = os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.mongo_db = os.getenv("MONGO_DB", "automation_bias")
        self.questions_file = os.getenv("QUESTIONS_FILE", "questions.csv")

        self.num_questions = int(os.getenv("NUM_QUESTIONS", "0"))
        self.question_time_seconds = int(os.getenv("QUESTION_TIME_SECONDS", "15"))
        trap_prob = float(os.getenv("TRAP_PROBABILITY", "0.25"))
        self.trap_probability = min(max(trap_prob, 0.0), 1.0)

        if self.question_time_seconds <= 0:
            raise ValueError("QUESTION_TIME_SECONDS must be a positive integer.")


def load_questions(file_path: str) -> List[Dict[str, str]]:
    """Read the CSV file and return structured questions."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"questions file not found at {file_path}")

    with path.open("r", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        questions = [row for row in reader]

    if not questions:
        raise ValueError("questions.csv is empty. Provide at least one question.")

    required_columns = {
        "question_id",
        "question_text",
        "ground_truth",
        "option_1",
        "option_2",
        "option_3",
        "option_4",
        "ai_true_response",
        "ai_false_response",
    }
    missing_columns = required_columns - set(questions[0].keys())
    if missing_columns:
        raise ValueError(f"Missing columns in CSV: {', '.join(sorted(missing_columns))}")

    return questions


class SessionManager:
    """Lightweight in-memory tracker for active survey sessions."""

    def __init__(self, questions_bank: List[Dict[str, str]], settings: Settings) -> None:
        self.questions_bank = [dict(q) for q in questions_bank]
        self.settings = settings
        self.sessions: Dict[str, Dict] = {}

    def create_session(self, student_id: str) -> str:
        if not self.questions_bank:
            raise RuntimeError("No questions available.")

        session_id = str(uuid4())
        group = random.choice(["CONTROL", "INTERVENTION"])
        questions = [dict(q) for q in self.questions_bank]
        random.shuffle(questions)

        limit = self.settings.num_questions
        if limit > 0:
            questions = questions[: min(limit, len(questions))]

        traps = {
            q["question_id"]: random.random() < self.settings.trap_probability
            for q in questions
        }

        self.sessions[session_id] = {
            "student_id": student_id,
            "group": group,
            "questions": questions,
            "traps": traps,
            "created_at": datetime.utcnow(),
        }
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict]:
        return self.sessions.get(session_id)

    def discard(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    def get_question(
        self, session_id: str, question_number: int
    ) -> Tuple[Dict[str, str], bool, str]:
        session = self.get_session(session_id)
        if not session:
            raise KeyError("Session not found.")

        questions = session["questions"]
        if question_number < 1 or question_number > len(questions):
            raise IndexError("Question number out of range.")

        question = dict(questions[question_number - 1])
        question["options"] = [
            question.get("option_1", ""),
            question.get("option_2", ""),
            question.get("option_3", ""),
            question.get("option_4", ""),
        ]

        is_trap = session["traps"][question["question_id"]]
        ai_text = (
            question.get("ai_false_response", "")
            if is_trap
            else question.get("ai_true_response", "")
        )
        return question, is_trap, ai_text

    def question_count(self, session_id: str) -> int:
        session = self.get_session(session_id)
        return len(session["questions"]) if session else 0


settings = Settings()
questions_bank = load_questions(settings.questions_file)

app = FastAPI(title="Automation Bias Pilot")
templates = Jinja2Templates(directory="templates")
session_manager = SessionManager(questions_bank, settings)

mongo_client = AsyncIOMotorClient(settings.mongo_uri)
db = mongo_client[settings.mongo_db]
responses_collection = db["responses"]


@app.on_event("shutdown")
async def shutdown_event() -> None:
    mongo_client.close()


@app.get("/")
async def landing(request: Request):
    return templates.TemplateResponse(
        "landing.html",
        {"request": request},
    )


@app.post("/start")
async def start_survey(request: Request, student_id: str = Form(...)):
    clean_student = student_id.strip()
    if not clean_student:
        raise HTTPException(status_code=400, detail="Student ID is required.")

    session_id = session_manager.create_session(clean_student)
    return RedirectResponse(
        url=f"/question/1?session_id={session_id}",
        status_code=HTTP_303_SEE_OTHER,
    )


@app.get("/question/{question_number}")
async def question_page(
    request: Request, question_number: int, session_id: str
):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session expired or invalid.")

    total_questions = session_manager.question_count(session_id)
    try:
        question, is_trap, ai_text = session_manager.get_question(
            session_id, question_number
        )
    except (KeyError, IndexError):
        raise HTTPException(status_code=404, detail="Question not found.")

    start_timestamp = datetime.utcnow().isoformat() + "Z"

    context = {
        "request": request,
        "session_id": session_id,
        "group": session["group"],
        "question_number": question_number,
        "total_questions": total_questions,
        "question": question,
        "ai_text": ai_text,
        "is_trap": is_trap,
        "timer_seconds": settings.question_time_seconds,
        "start_timestamp": start_timestamp,
    }
    return templates.TemplateResponse("question.html", context)


@app.post("/question/{question_number}")
async def submit_question(
    request: Request,
    question_number: int,
    session_id: str = Form(...),
    question_id: str = Form(...),
    user_initial_choice: Optional[str] = Form(None),
    user_final_choice: Optional[str] = Form(None),
    start_time: str = Form(...),
):
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session expired or invalid.")

    total_questions = session_manager.question_count(session_id)
    if question_number < 1 or question_number > total_questions:
        raise HTTPException(status_code=400, detail="Invalid question number.")

    question, is_trap, _ = session_manager.get_question(session_id, question_number)
    if question["question_id"] != question_id:
        raise HTTPException(status_code=400, detail="Question mismatch.")

    initial_choice = (user_initial_choice or "").strip() or None
    final_choice = (user_final_choice or "").strip() or "NO_ANSWER"

    try:
        start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    except ValueError:
        start_dt = datetime.now(timezone.utc)

    now = datetime.now(timezone.utc)
    if start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=timezone.utc)

    elapsed_ms = max(
        int((now - start_dt).total_seconds() * 1000),
        0,
    )

    record = {
        "student_id": session["student_id"],
        "session_id": session_id,
        "group": session["group"],
        "question_id": question_id,
        "is_trap": is_trap,
        "user_initial_choice": initial_choice,
        "user_final_choice": final_choice,
        "time_taken_ms": elapsed_ms,
        "timestamp": datetime.utcnow(),
    }
    await responses_collection.insert_one(record)

    next_question = question_number + 1
    if next_question > total_questions:
        session_manager.discard(session_id)
        return RedirectResponse(url="/done", status_code=HTTP_303_SEE_OTHER)

    return RedirectResponse(
        url=f"/question/{next_question}?session_id={session_id}",
        status_code=HTTP_303_SEE_OTHER,
    )


@app.get("/done")
async def completed(request: Request):
    return templates.TemplateResponse(
        "done.html",
        {"request": request},
    )

