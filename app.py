"""Gradio UI for Resume Job Application Agent."""

import gradio as gr
from dotenv import load_dotenv
import os
from parser import parse_documents
from agent import run_agent
import re
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
    "ats_result": None,
    "full_report": ""
}


def process_application(resume_file, jd_file, company_name):
    """
    Process resume and job description through the agent.
    
    Args:
        resume_file: Uploaded resume PDF
        jd_file: Uploaded job description PDF
        company_name: Company name for personalization
        
    Returns:
        Tuple of (ats_section, resume_suggestions, cover_letter, interview, role, skill_growth, full_report)
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
        
        # Store full report for potential download
        last_state["full_report"] = result.get("full_report", "")
        
        # Return results for each section (order matches UI outputs)
        # Wrap HTML sections for proper rendering
        return (
            result["ats_section"],
            html_wrap(result["resume_suggestions_section"]),
            result["cover_letter_section"],
            html_wrap(result["bullets_section"]),
            html_wrap(result["interview_section"]),
            html_wrap(result["role_expectations_section"]),
            html_wrap(result["skill_growth_section"]),
            result["full_report"],
            docs["resume"],  # Return raw resume text
            docs["jd"]       # Return raw JD text
        )
    
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}\n\nPlease check your API key and try again."
        return (error_msg, html_wrap(""), "", html_wrap(""), html_wrap(""), html_wrap(""), html_wrap(""), "", "", "")


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


def html_wrap(content):
    """
    Wrap content in HTML div for proper rendering.
    
    Args:
        content: Text content to wrap
        
    Returns:
        HTML-wrapped content
    """
    if not content or content.startswith("❌"):
        return f"<div style='padding: 20px; color: #888;'>{content if content else 'No content available.'}</div>"
    
    # Wrap in a styled div
    return f"<div style='padding: 20px; line-height: 1.6; white-space: pre-wrap;'>{content}</div>"


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


def prepare_download(full_report_text):
    """
    Prepare the full report for download.
    
    Args:
        full_report_text: The complete report text
        
    Returns:
        Path to the saved report file
    """
    import tempfile
    import os
    
    if not full_report_text or full_report_text.startswith("❌"):
        return None
    
    # Create a temporary file
    temp_dir = tempfile.gettempdir()
    report_path = os.path.join(temp_dir, "job_application_report.txt")
    
    # Write the report to file
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_report_text)
    
    return report_path


def refine_with_preference(cover_letter, bullets, preference, resume_text, jd_text):
    """
    Refine output based on user preference using the backend tool.
    """
    if not cover_letter or not bullets or not resume_text:
        return cover_letter, bullets
        
    try:
        from tools import refine_with_preference_tool
        
        # Strip HTML wrappers if present to get raw text for processing
        # Simple regex or string manipulation if needed, but the tool expects text
        # For now assuming simple text or that the tool handles it.
        # Actually, bullets might be wrapped in HTML div from previous steps. 
        # But refine_with_preference_tool expects plain text bullets.
        
        # Handle HTML wrapping removal if needed (simple check)
        

        # Extract only text inside the HTML wrapper
        clean_bullets = re.sub(r"<[^>]+>", "", bullets).strip()

        result = refine_with_preference_tool(
            cover_letter=cover_letter,
            bullets=clean_bullets,
            preference=preference,
            resume_text=resume_text,
            jd_text=jd_text
        )
        
        # Re-wrap the refined bullets in HTML since the UI expects HTML for that component
        # (The original bullets_output is gr.HTML, cover_letter is gr.Textbox)
        # We need to maintain consistency.
        
        refined_cover_letter = result["cover_letter"]
        refined_bullets = result["bullets"]
        
        # Wrap bullets again for consistent UI rendering 
        # (using html_wrap function which we can't call directly if it's not in scope or just re-apply simple div)
        # Note: html_wrap is defined in app.py, so we can use it.
        refined_bullets_html = html_wrap(refined_bullets)
        
        return refined_cover_letter, refined_bullets_html
        
    except Exception as e:
        return f"Error: {str(e)}", bullets


def populate_refinement_options(resume_text, jd_text, cover_letter, bullets):
    """
    Generate dynamic refinement options based on the application package.
    
    Returns:
        gr.Dropdown update with new choices
    """
    if not resume_text or not cover_letter:
        return gr.Dropdown(choices=["No options available yet"], value=None)
    
    try:
        from tools import generate_refinement_options
        
        # Strip HTML from bullets if needed
        clean_bullets = bullets
        if isinstance(bullets, str) and '<div' in bullets:
            # Simple HTML tag removal
            import re
            clean_bullets = re.sub(r'<[^>]+>', '', bullets)
        
        options = generate_refinement_options(
            resume_text=resume_text,
            jd_text=jd_text,
            cover_letter=cover_letter,
            bullets=clean_bullets
        )

        
        # Return dropdown update with new choices and first option selected
        return gr.Dropdown(choices=options, value=options[0] if options else None)
        
    except Exception as e:
        print(f"Error populating refinement options: {str(e)}")
        # Fallback options
        fallback = [
            "Make tone more professional",
            "Increase technical depth",
            "Focus on quantifiable achievements"
        ]
        return gr.Dropdown(choices=fallback, value=fallback[0])



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
            
            # Add top margin and center the input section
            gr.Markdown("<div style='margin-top: 30px;'></div>")
            
            # Input Section - Top Center with better styling
            with gr.Row():
                with gr.Column(scale=1):
                    pass  # Left spacer
                with gr.Column(scale=2):
                    with gr.Group():
                        agent_resume = gr.File(label="📄 Upload Resume (PDF)", file_types=[".pdf"])
                        agent_jd = gr.File(label="📋 Upload Job Description (PDF)", file_types=[".pdf"])
                        agent_company = gr.Textbox(
                            label="🏢 Company Name (Optional)",
                            placeholder="e.g., Google, Microsoft"
                        )
                        agent_run_btn = gr.Button("🚀 Run Agent", variant="primary", size="lg")
                with gr.Column(scale=1):
                    pass  # Right spacer
            
            gr.Markdown("<div style='margin-top: 40px;'></div>")
            gr.Markdown("<h2 style='text-align: center;'>📊 Results Dashboard</h2>")
            gr.Markdown("<div style='margin-bottom: 20px;'></div>")
            
            # Results Section - Full Width with card-like panels
            # ATS Analysis Section
            with gr.Group():
                gr.Markdown("<h3>📊 ATS Analysis</h3>")
                ats_output = gr.Textbox(
                    label="",
                    lines=10,
                    show_copy_button=True,
                    placeholder="Results will appear here after running the agent...",
                    interactive=False
                )
            
            # Resume Improvement Suggestions Section
            with gr.Group():
                gr.Markdown("<h3>📝 Resume Improvement Suggestions</h3>")
                bullets_output = gr.HTML(
                    label="",
                    value="<p style='color: #888; padding: 20px;'>Results will appear here after running the agent...</p>"
                )
            
            # Cover Letter Section
            with gr.Group():
                gr.Markdown("<h3>✍️ Cover Letter</h3>")
                cover_letter_output = gr.Textbox(
                    label="",
                    lines=15,
                    show_copy_button=True,
                    placeholder="Results will appear here after running the agent...",
                    interactive=False
                )
            
            # Optimized Resume Bullets Section
            with gr.Group():
                gr.Markdown("<h3>✨ Optimized Resume Bullets</h3>")
                optimized_bullets_output = gr.HTML(
                    label="",
                    value="<p style='color: #888; padding: 20px;'>Results will appear here after running the agent...</p>"
                )
            
            # Refinement Section
            gr.Markdown("<div style='margin-bottom: 10px;'></div>")
            with gr.Group():
                gr.Markdown("<h3>✨ Refine Results</h3>")
                with gr.Row():
                    refine_preference = gr.Dropdown(
                        label="🎛 Final Touch Preference",
                        choices=[],
                        value=None,
                        scale=3,
                        interactive=True
                    )
                    refine_btn = gr.Button("✨ Refine Output", variant="secondary", scale=1)
            
            # Hidden state components for refinement
            resume_text_hidden = gr.State("")
            jd_text_hidden = gr.State("")
            
            # Interview Preparation Section
            with gr.Group():
                gr.Markdown("<h3>💼 Interview Preparation</h3>")
                interview_output = gr.HTML(
                    label="",
                    value="<p style='color: #888; padding: 20px;'>Results will appear here after running the agent...</p>"
                )
            
            # Role Expectations Section
            with gr.Group():
                gr.Markdown("<h3>🔬 Role Expectations</h3>")
                role_output = gr.HTML(
                    label="",
                    value="<p style='color: #888; padding: 20px;'>Results will appear here after running the agent...</p>"
                )
            
            # Skill Growth Plan Section
            with gr.Group():
                gr.Markdown("<h3>📚 Skill Growth Plan</h3>")
                skill_growth_output = gr.HTML(
                    label="",
                    value="<p style='color: #888; padding: 20px;'>Results will appear here after running the agent...</p>"
                )
            
            # Download Full Report Section
            gr.Markdown("<div style='margin-top: 30px;'></div>")
            with gr.Group():
                gr.Markdown("<h3>📥 Download Complete Report</h3>")
                full_report_output = gr.Textbox(
                    label="",
                    lines=3,
                    visible=False,
                    interactive=False
                )
                download_btn = gr.DownloadButton(
                    label="📥 Download Full Report",
                    variant="secondary",
                    size="lg"
                )
            
            # Button click handler
            agent_run_btn.click(
                fn=process_application,
                inputs=[agent_resume, agent_jd, agent_company],
                outputs=[
                    ats_output,
                    bullets_output,
                    cover_letter_output,
                    optimized_bullets_output,  # New: Optimized bullets
                    interview_output,
                    role_output,
                    skill_growth_output,
                    full_report_output,
                    resume_text_hidden,  # New: Hidden resume text
                    jd_text_hidden       # New: Hidden JD text
                ]
            ).then(
                fn=populate_refinement_options,
                inputs=[
                    resume_text_hidden,
                    jd_text_hidden,
                    cover_letter_output,
                    optimized_bullets_output
                ],
                outputs=[refine_preference]
            )
            
            # Wire refinement button
            refine_btn.click(
                fn=refine_with_preference,
                inputs=[
                    cover_letter_output,
                    optimized_bullets_output,
                    refine_preference,
                    resume_text_hidden,
                    jd_text_hidden
                ],
                outputs=[
                    cover_letter_output,
                    optimized_bullets_output
                ]
            )
            
            # Wire download button to full report
            full_report_output.change(
                fn=prepare_download,
                inputs=[full_report_output],
                outputs=[download_btn]
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


