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
        Final output text
    """
    try:
        # Parse documents
        docs = parse_documents(resume_file, jd_file)
        
        # Store in global state for Q&A
        last_state["resume"] = docs["resume"]
        last_state["jd"] = docs["jd"]
        
        # Run agent
        output = run_agent(
            docs["resume"],
            docs["jd"],
            company_name or "the company"
        )
        
        return output
    
    except Exception as e:
        return f"❌ Error: {str(e)}\n\nPlease check your API key and try again."


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
        # 🎯 Resume → Job Application Agent
        ### Powered by LangGraph + Google Gemini
        """
    )
    
    with gr.Tabs():
        # Tab 1: Agent Mode
        with gr.Tab("🤖 Agent Mode"):
            gr.Markdown("**Full AI Agent Workflow** - Complete job application package with ATS analysis, cover letter, resume optimization, interview prep, role research, and learning plan.")
            
            with gr.Row():
                with gr.Column():
                    agent_resume = gr.File(label="📄 Upload Resume (PDF)", file_types=[".pdf"])
                    agent_jd = gr.File(label="📋 Upload Job Description (PDF)", file_types=[".pdf"])
                    agent_company = gr.Textbox(
                        label="🏢 Company Name (Optional)",
                        placeholder="e.g., Google, Microsoft"
                    )
                    agent_run_btn = gr.Button("🚀 Run Agent", variant="primary", size="lg")
                
                with gr.Column():
                    agent_output = gr.Textbox(
                        label="📦 Complete Job Application Package",
                        lines=30,
                        max_lines=35,
                        show_copy_button=True
                    )
            
            agent_run_btn.click(
                fn=process_application,
                inputs=[agent_resume, agent_jd, agent_company],
                outputs=agent_output
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
            qa_output = gr.Textbox(
                label="💡 Answer",
                lines=15,
                show_copy_button=True
            )
            
            qa_btn.click(
                fn=qa_about_match,
                inputs=qa_question,
                outputs=qa_output
            )
            
            gr.Markdown("*Note: Run the agent in 'Agent Mode' first to analyze your documents.*")
        
        # Tab 3: Manual Tools
        with gr.Tab("🛠️ Manual Tools"):
            gr.Markdown("**Individual Tools** - Run specific tools independently.")
            
            with gr.Accordion("📊 ATS Analysis Only", open=False):
                with gr.Row():
                    with gr.Column():
                        ats_resume = gr.File(label="Resume PDF", file_types=[".pdf"])
                        ats_jd = gr.File(label="JD PDF", file_types=[".pdf"])
                        ats_btn = gr.Button("Analyze ATS Score", variant="secondary")
                    with gr.Column():
                        ats_output = gr.Textbox(label="ATS Results", lines=15, show_copy_button=True)
                
                ats_btn.click(
                    fn=run_ats_only,
                    inputs=[ats_resume, ats_jd],
                    outputs=ats_output
                )
            
            with gr.Accordion("✍️ Cover Letter Only", open=False):
                with gr.Row():
                    with gr.Column():
                        cl_resume = gr.File(label="Resume PDF", file_types=[".pdf"])
                        cl_jd = gr.File(label="JD PDF", file_types=[".pdf"])
                        cl_company = gr.Textbox(label="Company Name", placeholder="Optional")
                        cl_btn = gr.Button("Generate Cover Letter", variant="secondary")
                    with gr.Column():
                        cl_output = gr.Textbox(label="Cover Letter", lines=15, show_copy_button=True)
                
                cl_btn.click(
                    fn=run_cover_letter_only,
                    inputs=[cl_resume, cl_jd, cl_company],
                    outputs=cl_output
                )
            
            with gr.Accordion("💼 Interview Prep Only", open=False):
                with gr.Row():
                    with gr.Column():
                        int_resume = gr.File(label="Resume PDF", file_types=[".pdf"])
                        int_jd = gr.File(label="JD PDF", file_types=[".pdf"])
                        int_btn = gr.Button("Generate Interview Questions", variant="secondary")
                    with gr.Column():
                        int_output = gr.Textbox(label="Interview Questions", lines=15, show_copy_button=True)
                
                int_btn.click(
                    fn=run_interview_prep_only,
                    inputs=[int_resume, int_jd],
                    outputs=int_output
                )
    
    gr.Markdown(
        """
        ---
        **Note:** Processing takes 30-90 seconds depending on the mode. PDFs must be text-based (not scanned images).
        """
    )


if __name__ == "__main__":
    demo.launch(share=False)

