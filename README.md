# Automation Bias Pilot Study App

FastAPI + Jinja2 web application for running a lightweight cognitive science pilot on automation bias. Participants answer timed CRT-style questions while optionally seeing AI suggestions, and every interaction is logged per question in MongoDB.

## Features

- FastAPI backend with async Motor client for MongoDB logging.
- Questions loaded from `questions.csv` at startup; customize via `create_questions.py`.
- Randomized participant group assignment (`CONTROL` vs `INTERVENTION`) and per-question trap selection with configurable probability.
- Tailwind-styled Jinja templates with timer, AI assistant reveal workflow, and auto-submit at timeout.
- Granular response documents stored in `responses` collection for downstream analysis.

## Quick Start

1. **Install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate    # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Prepare MongoDB**

   - Run a local instance (default URI `mongodb://localhost:27017`), or supply a connection string via `MONGO_URI`.
   - The app writes to the database defined by `MONGO_DB` (`automation_bias` by default) and the `responses` collection.

3. **Seed questions (optional)**

   The repo already includes `questions.csv`. Regenerate or customize it with:

   ```bash
   python create_questions.py
   ```

4. **Run the app**

   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

   Visit `http://localhost:8000` to begin the survey.

## Configuration

All knobs are exposed via environment variables (use a `.env` file or export in your shell):

| Variable | Description | Default |
| --- | --- | --- |
| `MONGO_URI` | Mongo connection string | `mongodb://localhost:27017` |
| `MONGO_DB` | Database name | `automation_bias` |
| `QUESTIONS_FILE` | Path to the CSV file loaded at startup | `questions.csv` |
| `NUM_QUESTIONS` | How many questions to ask (0 = use all) | `0` |
| `QUESTION_TIME_SECONDS` | Timer per question | `15` |
| `TRAP_PROBABILITY` | Probability a question shows a false AI suggestion | `0.25` |

Update these values to adjust the number of questions, the time per question, or the trap rate without touching the code.

## Data Model

Each submission inserts one document into `responses`:

```json
{
  "student_id": "abc123",
  "session_id": "uuid4",
  "group": "CONTROL" | "INTERVENTION",
  "question_id": "Q1",
  "is_trap": false,
  "user_initial_choice": "5 minutes",   // null for CONTROL
  "user_final_choice": "5 minutes",
  "time_taken_ms": 8231,
  "timestamp": "2024-01-01T12:34:56Z"
}
```

Use Mongo queries to aggregate by group, trap condition, or response latency.

## Frontend Flow

1. **Landing** – Collect `student_id` and start a session (randomly assigns group).
2. **Question pages** – Show progress, 15s countdown, AI insight panel.
   - Control: AI text visible immediately; submit after choosing.
   - Intervention: AI panel locked until a tentative choice; final answer recorded after confirmation.
   - Timer auto-submits as `NO_ANSWER` when it expires.
3. **Completion** – Simple acknowledgement screen.

Tailwind is loaded via CDN for a clean, Apple-esque layout; tweak styles inside `templates/`.

## Development Notes

- All session state (assignment, question order, trap selection) lives in-memory for simplicity. For production, migrate to a durable store.
- CSV columns must include the schema outlined in the task description.
- Keep `python-multipart` installed to allow FastAPI to parse form submissions.

Happy experimenting!

