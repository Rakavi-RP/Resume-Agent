# 🎯 Resume Job Application Agent

An AI agent that automates job application prep using **LangGraph** and **Google Gemini**. Upload your resume and a job description, and get a complete application package in under 2 minutes.

## What It Does

This isn't just another resume analyzer. It's a full agentic workflow that:

- Analyzes your ATS score with smart skill matching
- Generates a personalized cover letter
- Optimizes your resume bullets using the STAR method
- Creates role-specific interview questions
- Researches industry expectations for the role
- Builds a skill development plan
- **Reviews and improves its own output** (self-critique loop)
- Lets you refine results with dynamically generated options

## Why It's Actually Agentic

Most "AI agents" are just LLM wrappers. This one actually makes decisions:

1. **Conditional Routing** - Routes to different workflows based on your ATS score
2. **Self-Review Loop** - Critiques its own output and revises it
3. **Dynamic Refinement** - Generates context-specific improvement options using LLMs
4. **Tool Orchestration** - Coordinates 9+ specialized tools in a complex workflow
5. **State Management** - Uses LangGraph to maintain state across the entire process

## Quick Start

### Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd resume-job-agent

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Setup API key
echo "GOOGLE_API_KEY=your_key_here" > .env
```

Get a free Gemini API key: https://aistudio.google.com/app/apikey

### Run It

```bash
python app.py
```

Open `http://localhost:7860` in your browser.

## How to Use

1. Upload your resume (PDF)
2. Upload the job description (PDF)
3. Optionally enter the company name
4. Click "Run Agent" and wait 30-90 seconds
5. Review the 7 sections of output
6. Use the refinement dropdown to adjust emphasis
7. Download the full report

## Architecture

Built with **LangGraph**, the workflow looks like this:

```
Parse → ATS Analysis → Conditional Routing
                            ↓
                    (Score < 70: Deep Improvements)
                    (Score 70-85: Standard Suggestions)
                    (Score > 85: Minimal Feedback)
                            ↓
Cover Letter + Resume Bullets + Interview Prep
                            ↓
                    Self-Review (Agent critiques itself)
                            ↓
                    Revision (Agent improves based on critique)
                            ↓
                    Final Package
```

### The 9 Tools

Each tool is a specialized LLM call with a specific prompt:

1. **calculate_ats_score** - Smart skill extraction from resume sections
2. **generate_cover_letter** - Personalized to company and role
3. **optimize_resume_bullets** - STAR method formatting
4. **generate_interview_questions** - Technical + behavioral questions
5. **generate_resume_improvements** - Conditional based on ATS score
6. **research_role_expectations** - Industry insights and trends
7. **generate_learning_plan** - Skill development roadmap
8. **self_review_output** - Quality assurance check
9. **revise_content** - Content refinement based on review
10. **refine_with_preference_tool** - User-guided content adjustment
11. **generate_refinement_options** - LLM-generated context-specific options

## File Structure

```
├── app.py          # Gradio UI and main application
├── agent.py        # LangGraph workflow with 9 nodes
├── tools.py        # All 11 LLM-powered tools
├── parser.py       # PDF text extraction
├── requirements.txt
├── .env           # Your API key (create this)
└── README.md
```

## Key Features

### Smart ATS Analysis
Extracts skills ONLY from Skills, Projects, and Experience sections. No hallucination - just accurate matching against job requirements.

### Conditional Logic
The agent decides which path to take based on your ATS score. Low score? You get deep, actionable improvements. High score? Just minor tweaks.

### Self-Review Loop
After generating all content, the agent reviews its own work, identifies issues, and revises. This is true agentic behavior - self-evaluation and improvement.

### Dynamic Refinement
After the initial run, the agent generates 3-5 refinement options specific to YOUR resume and job. These aren't hard-coded - they're created by analyzing the context. Select one, and the agent rewrites the content to emphasize that aspect.

## Tech Stack

- **LangGraph** - Agent orchestration and state management
- **LangChain** - LLM framework and prompt templates
- **Google Gemini 2.5 Flash** - Fast, efficient, free-tier LLM
- **Gradio** - Modern web UI
- **PyPDF2** - PDF parsing
- **Python-dotenv** - Environment management


## Demo Video

[Link to demo video will be added]


## Contributing

Found a bug? Have an idea? Open an issue or PR!
