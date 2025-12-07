"""Gradio UI for Resume Job Application Agent."""

import gradio as gr
from dotenv import load_dotenv
import os
from parser import parse_documents
from agent import run_agent
from tools import (
    calculate_ats_score,
    generate_cover_letter,
    generate_interview_questions,
    create_llm
)
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Verify API key
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in .env file!")

# Global state to store last processed documents
last_state = {
    "resume": "",
    "jd": "",
    "ats_result": None
}


def process_application(resume_file, jd_file, company_name):
    """
    Process resume and job description through the agent.
    
    Args:
        resume_file: Uploaded resume PDF
        jd_file: Uploaded job description PDF
        company_name: Company name for personalization
        
    Returns:
        Tuple of (ats_md, ats_copy, cover_md, cover_copy, bullets_md, bullets_copy, 
                  interview_md, interview_copy, role_md, role_copy, skill_md, skill_copy)
    """
    try:
        # Parse documents
        docs = parse_documents(resume_file, jd_file)
        
        # Store in global state for Q&A
        last_state["resume"] = docs["resume"]
        last_state["jd"] = docs["jd"]
        
        # Run agent
        result = run_agent(
            docs["resume"],
            docs["jd"],
            company_name or "the company"
        )
        
        # Return both markdown display and copy textbox values
        return (
            result["ats"], result["ats"],
            result["cover_letter"], result["cover_letter"],
            result["bullets"], result["bullets"],
            result["interview"], result["interview"],
            result["role_expectations"], result["role_expectations"],
            result["skill_growth"], result["skill_growth"]
        )
    
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nPlease check your API key and try again."
        return (error_msg, "", "", "", "", "", "", "", "", "", "", "")


def qa_about_match(question):
    """
    Answer questions about the resume-JD match using stored state.
    
    Args:
        question: User's question
        
    Returns:
        Answer from LLM
    """
    if not last_state["resume"] or not last_state["jd"]:
        return "⚠️ Please run the agent first in 'Agent Mode' tab to analyze your resume and JD."
    
    try:
        llm = create_llm()
        
        prompt = PromptTemplate(
            input_variables=["resume", "jd", "question"],
            template="""You are a career advisor assistant. Answer the user's question based on the resume and job description.

Resume:
{resume}

Job Description:
{jd}

User Question: {question}

Provide a helpful, specific answer based on the documents above. Be concise but thorough.

Answer:
"""
        )
        
        response = llm.invoke(prompt.format(
            resume=last_state["resume"],
            jd=last_state["jd"],
            question=question
        ))
        
        return response.content
    
    except Exception as e:
        return f"❌ Error: {str(e)}"


def run_ats_only(resume_file, jd_file):
    """Run ATS analysis only."""
    try:
        docs = parse_documents(resume_file, jd_file)
        result = calculate_ats_score(docs["resume"], docs["jd"])
        
        output = f"""
📊 ATS MATCH SCORE: {result['score']}/100

✅ MATCHED SKILLS:
{chr(10).join(f"  • {skill}" for skill in result['matched_skills'][:15])}

❌ MISSING SKILLS:
{chr(10).join(f"  • {skill}" for skill in result['missing_skills'][:15])}
"""
        return output
    except Exception as e:
        return f"❌ Error: {str(e)}"


def run_cover_letter_only(resume_file, jd_file, company_name):
    """Run cover letter generation only."""
    try:
        docs = parse_documents(resume_file, jd_file)
        cover_letter = generate_cover_letter(
            docs["resume"],
            docs["jd"],
            company_name or "the company"
        )
        return cover_letter
    except Exception as e:
        return f"❌ Error: {str(e)}"


def run_interview_prep_only(resume_file, jd_file):
    """Run interview questions generation only."""
    try:
        docs = parse_documents(resume_file, jd_file)
        questions = generate_interview_questions(docs["jd"], docs["resume"])
        return questions
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Create Gradio interface with tabs
with gr.Blocks(title="Resume Job Application Agent", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown(
        """
        <h1 style='text-align:center;'>🎯 Resume → Job Application Agent</h1>
        <p style='text-align:center;'>Powered by LangGraph + Google Gemini</p>
        """
    )
    
    with gr.Tabs():
        # Tab 1: Agent Mode
        with gr.Tab("🤖 Agent Mode"):
            gr.Markdown("**Full AI Agent Workflow** - Complete job application package with ATS analysis, cover letter, resume optimization, interview prep, role research, and learning plan.")
            
            with gr.Row():
                # Left Column: Inputs
                with gr.Column(scale=1):
                    agent_resume = gr.File(label="📄 Upload Resume (PDF)", file_types=[".pdf"])
                    agent_jd = gr.File(label="📋 Upload Job Description (PDF)", file_types=[".pdf"])
                    agent_company = gr.Textbox(
                        label="🏢 Company Name (Optional)",
                        placeholder="e.g., Google, Microsoft"
                    )
                    agent_run_btn = gr.Button("🚀 Run Agent", variant="primary", size="lg")
                
                # Right Column: Outputs
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Results")
                    
                    # ATS Analysis
                    gr.Markdown("#### ATS Analysis")
                    ats_output = gr.Markdown()
                    ats_copy = gr.Textbox(visible=False, show_copy_button=True, label="Copy ATS")
                    
                    # Cover Letter
                    gr.Markdown("#### ✍️ Cover Letter")
                    cover_letter_output = gr.Markdown()
                    cover_copy = gr.Textbox(visible=False, show_copy_button=True, label="Copy Cover Letter")
                    
                    # Resume Bullets
                    gr.Markdown("#### 📝 Optimized Resume Bullets")
                    bullets_output = gr.Markdown()
                    bullets_copy = gr.Textbox(visible=False, show_copy_button=True, label="Copy Bullets")
                    
                    # Interview Prep
                    gr.Markdown("#### 💼 Interview Preparation")
                    interview_output = gr.Markdown()
                    interview_copy = gr.Textbox(visible=False, show_copy_button=True, label="Copy Interview")
                    
                    # Role Expectations
                    gr.Markdown("#### 🔬 Role Expectations")
                    role_output = gr.Markdown()
                    role_copy = gr.Textbox(visible=False, show_copy_button=True, label="Copy Role")
                    
                    # Skill Growth Plan
                    gr.Markdown("#### 📚 Skill Growth Plan")
                    skill_growth_output = gr.Markdown()
                    skill_copy = gr.Textbox(visible=False, show_copy_button=True, label="Copy Skills")
            
            # Button click handler
            agent_run_btn.click(
                fn=process_application,
                inputs=[agent_resume, agent_jd, agent_company],
                outputs=[
                    ats_output, ats_copy,
                    cover_letter_output, cover_copy,
                    bullets_output, bullets_copy,
                    interview_output, interview_copy,
                    role_output, role_copy,
                    skill_growth_output, skill_copy
                ]
            )
        
        # Tab 2: Ask the Agent
        with gr.Tab("💬 Ask the Agent"):
            gr.Markdown("**Q&A Mode** - Ask questions about your resume-JD match after running the agent.")
            
            qa_question = gr.Textbox(
                label="❓ Your Question",
                placeholder="e.g., What are my strongest qualifications for this role?",
                lines=3
            )
            qa_btn = gr.Button("Ask", variant="primary")
            qa_output = gr.Markdown(label="💡 Answer")
            
            qa_btn.click(
                fn=qa_about_match,
                inputs=qa_question,
                outputs=qa_output
            )
            
            gr.Markdown("*Note: Run the agent in 'Agent Mode' first to analyze your documents.*")
    
    gr.Markdown(
        """
        ---
        **Note:** Processing takes 30-90 seconds. PDFs must be text-based (not scanned images).
        """
    )


if __name__ == "__main__":
    demo.launch(share=False)


