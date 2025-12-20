"""
Export survey responses from MongoDB to CSV.

Usage:
    python export_survey_data.py [--output OUTPUT_FILE] [--database DB_NAME] [--collection COLLECTION_NAME]

Defaults:
    - Output file: survey_responses.csv
    - Database: automation_bias (from MONGO_DB env var or default)
    - Collection: responses
"""
import argparse
import asyncio
import csv
import json
import os
import sys
from datetime import datetime
from typing import List, Dict, Any

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient


load_dotenv()


async def export_responses_to_csv(
    mongo_uri: str,
    db_name: str,
    collection_name: str,
    output_file: str
) -> None:
    """Fetch all responses from MongoDB and write to CSV."""
    client = AsyncIOMotorClient(mongo_uri)
    db = client[db_name]
    collection = db[collection_name]

    print(f"Connecting to MongoDB at {mongo_uri}...")
    print(f"Database: {db_name}, Collection: {collection_name}")

    # Fetch all documents
    cursor = collection.find({})
    documents: List[Dict[str, Any]] = []
    
    async for doc in cursor:
        # Convert ObjectId to string if present
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])
        # Convert datetime to ISO format string
        if "timestamp" in doc and isinstance(doc["timestamp"], datetime):
            doc["timestamp"] = doc["timestamp"].isoformat()
        documents.append(doc)

    if not documents:
        print("No documents found in the collection.")
        client.close()
        return

    print(f"Found {len(documents)} response(s).")

    # Determine all possible field names from all documents
    all_fields = set()
    for doc in documents:
        all_fields.update(doc.keys())

    # Order fields logically (put common ones first)
    field_order = [
        "student_id",
        "email",
        "session_id",
        "group",
        "question_id",
        "user_initial_choice",
        "user_final_choice",
        "ai_suggested_option",
        "ai_confidence_shown",
        "ai_was_correct",
        "correct_answer",
        "time_taken_ms",
        "timestamp",
        "_id",
    ]
    
    # Add any remaining fields that weren't in the ordered list
    remaining_fields = sorted(all_fields - set(field_order))
    fieldnames = [f for f in field_order if f in all_fields] + remaining_fields

    # Write to CSV
    print(f"Writing to {output_file}...")
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        
        for doc in documents:
            # Ensure all fields are strings or None
            row = {}
            for field in fieldnames:
                value = doc.get(field)
                if value is None:
                    row[field] = ""
                elif isinstance(value, (dict, list)):
                    # Convert complex types to JSON-like string
                    row[field] = json.dumps(value)
                else:
                    row[field] = str(value)
            writer.writerow(row)

    print(f"Successfully exported {len(documents)} responses to {output_file}")
    client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Export survey responses from MongoDB to CSV"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="survey_responses.csv",
        help="Output CSV file path (default: survey_responses.csv)",
    )
    parser.add_argument(
        "--database",
        "-d",
        default=None,
        help="MongoDB database name (default: from MONGO_DB env var or 'automation_bias')",
    )
    parser.add_argument(
        "--collection",
        "-c",
        default="responses",
        help="MongoDB collection name (default: responses)",
    )
    parser.add_argument(
        "--uri",
        "-u",
        default=None,
        help="MongoDB connection URI (default: from MONGO_URI env var or 'mongodb://localhost:27017')",
    )

    args = parser.parse_args()

    # Get MongoDB connection settings
    mongo_uri = args.uri or os.getenv("MONGO_URI", "mongodb://localhost:27017")
    db_name = args.database or os.getenv("MONGO_DB", "automation_bias")

    # Run the async export
    try:
        asyncio.run(
            export_responses_to_csv(
                mongo_uri=mongo_uri,
                db_name=db_name,
                collection_name=args.collection,
                output_file=args.output,
            )
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
