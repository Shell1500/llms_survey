"""
Utility script to (re)generate questions.csv with a few example CRT prompts.

Usage:
    python create_questions.py
"""

import csv
from pathlib import Path

QUESTIONS = [
    {
        "question_id": "Q1",
        "question_text": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
        "ground_truth": "$0.05",
        "option_1": "$0.05",
        "option_2": "$0.10",
        "option_3": "$0.15",
        "option_4": "$0.20",
        "ai_true_response": "AI suggests $0.05 because if the ball were five cents, the bat would be $1.05 and that totals $1.10.",
        "ai_false_response": "AI suggests $0.10 because ten cents plus the extra dollar for the bat seems to make $1.10 at a glance.",
    },
    {
        "question_id": "Q2",
        "question_text": "If it takes 5 machines 5 minutes to make 5 gadgets, how long would it take 100 machines to make 100 gadgets?",
        "ground_truth": "5 minutes",
        "option_1": "5 minutes",
        "option_2": "100 minutes",
        "option_3": "20 minutes",
        "option_4": "1 minute",
        "ai_true_response": "AI suggests 5 minutes because each machine produces one gadget in five minutes, so more machines finish in the same time.",
        "ai_false_response": "AI suggests 100 minutes because scaling the number of machines feels like it should scale the time as well.",
    },
    {
        "question_id": "Q3",
        "question_text": "In a lake, a patch of lily pads doubles in size every day. It takes 48 days for the patch to cover the entire lake. How long would it take to cover half the lake?",
        "ground_truth": "47 days",
        "option_1": "24 days",
        "option_2": "47 days",
        "option_3": "48 days",
        "option_4": "12 days",
        "ai_true_response": "AI suggests 47 days because doubling daily means half coverage occurs one day before the lake is full.",
        "ai_false_response": "AI suggests 24 days because halfway through the time period feels intuitively like half coverage.",
    },
]

OUTPUT_PATH = Path("questions.csv")


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "question_id",
                "question_text",
                "ground_truth",
                "option_1",
                "option_2",
                "option_3",
                "option_4",
                "ai_true_response",
                "ai_false_response",
            ],
        )
        writer.writeheader()
        writer.writerows(QUESTIONS)
    print(f"Wrote {len(QUESTIONS)} questions to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()

