# 🎯 Resume → Job Application Agent

An intelligent AI agent built with **LangGraph** and **Google Gemini** that automates job application preparation.

## 🚀 Features

- **ATS Score Analysis** - Match your resume against job requirements
- **Tailored Cover Letter** - Auto-generated personalized cover letters
- **Resume Optimization** - Improved bullet points aligned with JD
- **Interview Prep** - Likely interview questions based on role

## 🏗️ Architecture

Built using **LangGraph** state machine with 6 nodes:

```
Parse → ATS Analysis → Cover Letter → Resume Optimizer → Interview Prep → Compile Output
```

### Agent Tools (4)
1. `calculate_ats_score` - Skills matching & scoring
2. `generate_cover_letter` - Personalized cover letters
3. `optimize_resume_bullets` - STAR-method bullet points
4. `generate_interview_questions` - Role-specific prep

## 📦 Installation

### 1. Clone Repository
```bash
git clone <your-repo-url>
cd resume-job-agent
```

### 2. Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup API Key
Create a `.env` file:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```

Get your free API key: https://aistudio.google.com/app/apikey

## 🎮 Usage

### Run the App
```bash
python app.py
```

### Steps:
1. Upload your **Resume PDF**
2. Upload **Job Description PDF**
3. (Optional) Enter company name
4. Click **Generate**
5. Download the complete package

## 🧪 Testing

Use sample PDFs in `samples/` folder (create your own test files).

## 🎥 Demo Video

[Link to demo video]

## 🛠️ Tech Stack

- **LangGraph** - Agent orchestration
- **LangChain** - LLM framework
- **Google Gemini** - LLM (gemini-1.5-flash)
- **Gradio** - Web UI
- **PyPDF2** - PDF parsing

## 📊 LangGraph Workflow

The agent uses a sequential state machine:

1. **Parse Node** - Initialize state
2. **ATS Analysis** - Calculate match score
3. **Cover Letter** - Generate personalized letter
4. **Resume Optimizer** - Improve bullet points
5. **Interview Prep** - Generate questions
6. **Compile Output** - Create final package

## 🎓 Hackathon Submission

**Theme:** Building AI Agents with LangChain/LangGraph

**Key Highlights:**
- ✅ Multi-step agentic reasoning
- ✅ Tool orchestration via LangGraph
- ✅ Real-world utility (job applications)
- ✅ Clean UI with Gradio
- ✅ Free tier compatible (Gemini API)

## 📝 License

MIT

## 👤 Author

[Your Name] - [Hackathon Submission]
