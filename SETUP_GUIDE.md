# Setup and Testing Guide

## Prerequisites

- Python 3.8 or higher
- MongoDB Atlas account (free tier works fine)
- Git (already installed)

## Step 1: Set Up Python Environment

1. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```

2. **Activate the virtual environment:**
   - **Windows (PowerShell):**
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - **Windows (Command Prompt):**
     ```cmd
     .venv\Scripts\activate.bat
     ```
   - **Linux/Mac:**
     ```bash
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Step 2: Set Up MongoDB Atlas

### A. Create MongoDB Atlas Account & Cluster

1. Go to [MongoDB Atlas](https://www.mongodb.com/cloud/atlas/register)
2. Sign up for a free account (or sign in if you already have one)
3. Create a new **FREE** cluster (M0 Sandbox)
4. Choose a cloud provider and region (closest to you)
5. Click "Create Cluster" (takes 3-5 minutes)

### B. Configure Database Access

1. In the left sidebar, click **"Database Access"**
2. Click **"Add New Database User"**
3. Choose **"Password"** authentication
4. Create a username and password (save these!)
5. Set privileges to **"Atlas admin"** (or "Read and write to any database")
6. Click **"Add User"**

### C. Configure Network Access

1. In the left sidebar, click **"Network Access"**
2. Click **"Add IP Address"**
3. For testing, click **"Allow Access from Anywhere"** (adds `0.0.0.0/0`)
   - ⚠️ **Security Note:** For production, add only your specific IP addresses
4. Click **"Confirm"**

### D. Get Your Connection String

1. In the left sidebar, click **"Database"** (or "Clusters")
2. Click **"Connect"** on your cluster
3. Choose **"Connect your application"**
4. Select **"Python"** and version **"3.6 or later"**
5. Copy the connection string (looks like):
   ```
   mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```
6. **Replace `<username>` and `<password>`** with your actual database user credentials
7. **Add your database name** at the end (before `?`):
   ```
   mongodb+srv://myuser:mypassword@cluster0.xxxxx.mongodb.net/automation_bias?retryWrites=true&w=majority
   ```

## Step 3: Configure Environment Variables

1. **Create a `.env` file** in the project root (copy from `.env.example` if it exists)
2. **Add your MongoDB connection string:**

   ```env
   MONGO_URI=mongodb+srv://yourusername:yourpassword@cluster0.xxxxx.mongodb.net/automation_bias?retryWrites=true&w=majority
   MONGO_DB=automation_bias
   QUESTIONS_FILE=new_questions.csv
   NUM_QUESTIONS=0
   QUESTION_TIME_SECONDS=30
   ```

   **Important:** Replace `yourusername`, `yourpassword`, and `cluster0.xxxxx.mongodb.net` with your actual Atlas credentials!

### Environment Variables Explained

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `MONGO_URI` | MongoDB connection string | `mongodb://localhost:27017` | Your Atlas connection string |
| `MONGO_DB` | Database name | `automation_bias` | `automation_bias` |
| `QUESTIONS_FILE` | CSV file with questions | `new_questions.csv` | `new_questions.csv` |
| `NUM_QUESTIONS` | Number of questions (0 = all) | `0` | `5` for testing |
| `QUESTION_TIME_SECONDS` | Timer per question | `30` | `15` for faster testing |

## Step 4: Run the Application

1. **Make sure your virtual environment is activated** (you should see `(.venv)` in your terminal)

2. **Start the FastAPI server:**
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

3. **You should see output like:**
   ```
   INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
   INFO:     Started reloader process
   INFO:     Started server process
   INFO:     Waiting for application startup.
   INFO:     Application startup complete.
   ```

4. **Open your browser** and go to: `http://localhost:8000`

## Step 5: Test the Application

### Basic Flow Test

1. **Landing Page:**
   - Go to `http://localhost:8000`
   - Enter a test student ID (e.g., `test_student_001`)
   - Click "Start Survey"

2. **Question Page:**
   - You'll see a question with a timer
   - Select an answer
   - If you're in the **INTERVENTION** group, you'll see AI suggestions after selecting
   - If you're in the **CONTROL** group, AI suggestions are visible immediately
   - Submit your answer

3. **Complete the Survey:**
   - Answer all questions
   - You'll see a completion page

### Testing Different Groups

- The app alternates between **CONTROL** and **INTERVENTION** groups
- **CONTROL**: AI suggestions visible immediately
- **INTERVENTION**: AI suggestions appear after you make an initial choice

### Quick Test Mode

To test faster, set in `.env`:
```env
NUM_QUESTIONS=3
QUESTION_TIME_SECONDS=10
```

This will show only 3 questions with 10-second timers.

## Step 6: Verify Data in MongoDB Atlas

1. Go to MongoDB Atlas dashboard
2. Click **"Database"** → **"Browse Collections"**
3. Select your cluster → `automation_bias` database → `responses` collection
4. You should see documents with your test responses!

### Sample Document Structure:
```json
{
  "student_id": "test_student_001",
  "session_id": "uuid-here",
  "group": "CONTROL",
  "question_id": "H1",
  "user_initial_choice": null,
  "user_final_choice": "Day 24",
  "ai_suggested_option": "Day 24",
  "ai_confidence_shown": 92,
  "ai_was_correct": false,
  "correct_answer": "Day 47",
  "time_taken_ms": 8231,
  "timestamp": "2024-01-01T12:34:56Z"
}
```

## Troubleshooting

### Connection Issues

**Error: "ServerSelectionTimeoutError"**
- Check your `MONGO_URI` in `.env` file
- Verify your IP is whitelisted in Network Access
- Make sure username/password are correct (no `<` or `>` brackets)

**Error: "Authentication failed"**
- Double-check your database username and password
- Make sure you replaced `<username>` and `<password>` in the connection string

### Application Issues

**Error: "questions file not found"**
- Make sure `new_questions.csv` exists in the project root
- Check `QUESTIONS_FILE` in `.env` matches the actual filename

**Error: "Missing columns in CSV"**
- The CSV must have these columns:
  - `question_id`, `question_text`, `ground_truth`, `option_1`, `option_2`, `option_3`, `option_4`
  - `ai_suggested_option`, `ai_explanation`, `ai_confidence`, `time_question`

**Port already in use:**
- Change the port: `uvicorn main:app --reload --host 0.0.0.0 --port 8001`

## Testing Checklist

- [ ] Virtual environment created and activated
- [ ] Dependencies installed
- [ ] MongoDB Atlas cluster created
- [ ] Database user created
- [ ] IP address whitelisted
- [ ] `.env` file created with correct `MONGO_URI`
- [ ] Application starts without errors
- [ ] Can access `http://localhost:8000`
- [ ] Can complete a full survey
- [ ] Data appears in MongoDB Atlas

## Next Steps

- Customize questions in `new_questions.csv`
- Adjust timer and question count in `.env`
- Modify templates in `templates/` folder
- Analyze responses in MongoDB Atlas

Happy testing! 🚀

