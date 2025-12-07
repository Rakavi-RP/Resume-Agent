# 🎯 Resume → Job Application Agent

An intelligent AI agent built with **LangGraph** and **Google Gemini** that automates comprehensive job application preparation with smart ATS analysis and personalized outputs.

## ✨ Key Features

### 🎯 Smart ATS Analysis
- **Intelligent Skill Extraction** - Extracts skills ONLY from relevant sections (Skills, Projects, Experience)
- **Accurate Matching** - Precise skill matching with no hallucination
- **Score Calculation** - (Matched Skills / Required Skills) × 100

### 📝 Conditional Resume Suggestions
- **Score ≥ 90**: No suggestions needed
- **Score ≥ 85**: Minimal feedback
- **Score < 85**: 2-3 crisp, actionable suggestions

### 📋 Complete Application Package
1. **ATS Score Analysis** - Matched/missing skills breakdown
2. **Resume Improvement Suggestions** - Conditional based on score
3. **Tailored Cover Letter** - Personalized for the role
4. **Optimized Resume Bullets** - STAR-method improvements
5. **Interview Preparation** - 8-10 likely questions
6. **Role Expectations** - Industry insights and trends
7. **Skill Growth Plan** - Learning roadmap with resources

### 💬 Interactive Q&A
- Ask follow-up questions about your resume-JD match
- Get instant answers based on analyzed documents

## 🏗️ Architecture

### LangGraph Workflow
```
┌─────────┐     ┌──────────────┐     ┌─────────────────┐
│  Parse  │ ──▶ │ ATS Analysis │ ──▶ │ Resume Improve  │
└─────────┘     └──────────────┘     └─────────────────┘
                                              │
                                              ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│ Self-Review │ ◀── │ Compile      │ ◀── │ Interview    │
└─────────────┘     └──────────────┘     │ Prep + Role  │
      │                                   └──────────────┘
      ▼                                          ▲
┌─────────────┐     ┌──────────────┐           │
│   Revise    │ ──▶ │ Cover Letter │ ──────────┘
└─────────────┘     └──────────────┘
```

### Agent Tools (9)
1. `calculate_ats_score` - Smart skill extraction & matching
2. `generate_cover_letter` - Personalized cover letters
3. `optimize_resume_bullets` - STAR-method improvements
4. `generate_interview_questions` - Role-specific questions
5. `generate_resume_improvements` - Conditional suggestions
6. `research_role_expectations` - Industry insights
7. `generate_learning_plan` - Skill development roadmap
8. `self_review_output` - Quality assurance
9. `revise_content` - Content refinement

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

The app will open in your browser at `http://localhost:7860`

### Steps:
1. **Upload Resume PDF** - Your current resume
2. **Upload Job Description PDF** - Target job posting
3. **(Optional) Enter Company Name** - For personalization
4. **Click "🚀 Run Agent"** - Process takes 30-90 seconds
5. **Review Results** - 7 separate sections with insights
6. **Download Full Report** - Complete package as text file
7. **Ask Questions** - Use Q&A tab for follow-ups

## 🛠️ Tech Stack

- **LangGraph** - Agent orchestration & state management
- **LangChain** - LLM framework & prompt templates
- **Google Gemini 2.5 Flash** - Fast, efficient LLM
- **Gradio** - Modern web UI with card-based layout
- **PyPDF2** - PDF text extraction
- **Python-dotenv** - Environment management

## 📊 Code Structure

```
resume-job-agent/
├── app.py              # Gradio UI & main application
├── agent.py            # LangGraph workflow & nodes
├── tools.py            # LLM-powered tools (9 functions)
├── parser.py           # PDF parsing utilities
├── requirements.txt    # Python dependencies
├── .env               # API keys (create this)
└── README.md          # This file
```

## 🎯 How It Works

### 1. ATS Analysis (Smart & Accurate)
- Extracts skills from Skills, Projects, Experience sections only
- Matches against job requirements
- Calculates precise score: (Matched / Required) × 100

### 2. Conditional Suggestions
- High scores (≥90): No suggestions
- Good scores (≥85): Minimal feedback
- Lower scores (<85): 2-3 actionable improvements

### 3. Comprehensive Outputs
Each section is generated independently:
- Cover letter tailored to company & role
- Resume bullets using STAR method
- Interview questions (technical + behavioral)
- Role expectations & industry trends
- Learning plan with specific resources

### 4. Quality Assurance
- Self-review node critiques outputs
- Revision node improves quality
- Final package combines all sections

## 🎓 Hackathon Submission

**Theme:** Building AI Agents with LangChain/LangGraph

**Key Highlights:**
- ✅ **Multi-step Agentic Reasoning** - 9-node LangGraph workflow
- ✅ **Conditional Logic** - Smart routing based on ATS score
- ✅ **Tool Orchestration** - 9 specialized LLM tools
- ✅ **State Management** - Typed state with LangGraph
- ✅ **Real-world Utility** - Solves actual job application pain
- ✅ **Clean UI** - Card-based Gradio interface
- ✅ **Free Tier Compatible** - Uses Gemini 2.5 Flash
- ✅ **No Hallucination** - Accurate skill extraction
- ✅ **Quality Control** - Self-review & revision loop

## 🎥 Demo Video

[Link to demo video]

## 📝 License

MIT License - Feel free to use and modify!

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.


