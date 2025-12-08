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

load_dotenv()

if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in .env file!")

last_state = {
    "resume": "",
    "jd": "",
    "ats_result": None,
    "full_report": ""
}


def process_application(resume_file, jd_file, company_name):
    try:
        docs = parse_documents(resume_file, jd_file)
        
        last_state["resume"] = docs["resume"]
        last_state["jd"] = docs["jd"]
        
        result = run_agent(
            docs["resume"],
            docs["jd"],
            company_name or "the company"
        )
        
        last_state["full_report"] = result.get("full_report", "")
        
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
    if not content or content.startswith("❌"):
        return f"<div style='padding: 20px; color: #888;'>{content if content else 'No content available.'}</div>"
    
    return f"<div style='padding: 20px; line-height: 1.6; white-space: pre-wrap;'>{content}</div>"


def run_ats_only(resume_file, jd_file):
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
    try:
        docs = parse_documents(resume_file, jd_file)
        questions = generate_interview_questions(docs["jd"], docs["resume"])
        return questions
    except Exception as e:
        return f"❌ Error: {str(e)}"


def prepare_download(full_report_text):
    import tempfile
    import os
    
    if not full_report_text or full_report_text.startswith("❌"):
        return None
    
    temp_dir = tempfile.gettempdir()
    report_path = os.path.join(temp_dir, "job_application_report.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_report_text)
    
    return report_path


def refine_with_preference(cover_letter, bullets, preference, resume_text, jd_text):
    if not cover_letter or not bullets or not resume_text:
        return cover_letter, bullets
        
    try:
        from tools import refine_with_preference_tool
        
        clean_bullets = re.sub(r"<[^>]+>", "", bullets).strip()

        result = refine_with_preference_tool(
            cover_letter=cover_letter,
            bullets=clean_bullets,
            preference=preference,
            resume_text=resume_text,
            jd_text=jd_text
        )
        
        refined_cover_letter = result["cover_letter"]
        refined_bullets = result["bullets"]
        
        refined_bullets_html = html_wrap(refined_bullets)
        
        return refined_cover_letter, refined_bullets_html
        
    except Exception as e:
        return f"Error: {str(e)}", bullets


def populate_refinement_options(resume_text, jd_text, cover_letter, bullets):
    if not resume_text or not cover_letter:
        return gr.Dropdown(choices=["No options available yet"], value=None)
    
    try:
        from tools import generate_refinement_options
        
        clean_bullets = bullets
        if isinstance(bullets, str) and '<div' in bullets:
            import re
            clean_bullets = re.sub(r'<[^>]+>', '', bullets)
        
        options = generate_refinement_options(
            resume_text=resume_text,
            jd_text=jd_text,
            cover_letter=cover_letter,
            bullets=clean_bullets
        )

        return gr.Dropdown(choices=options, value=options[0] if options else None)
        
    except Exception as e:
        print(f"Error populating refinement options: {str(e)}")
        fallback = [
            "Make tone more professional",
            "Increase technical depth",
            "Focus on quantifiable achievements"
        ]
        return gr.Dropdown(choices=fallback, value=fallback[0])



with gr.Blocks(title="Resume Job Application Agent", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown(
        """
        <h1 style='text-align:center;'>🎯 Resume → Job Application Agent</h1>
        <p style='text-align:center;'>Powered by LangGraph + Google Gemini</p>
        """
    )
    
    with gr.Tabs():
        with gr.Tab("🤖 Agent Mode"):
            gr.Markdown("**Full AI Agent Workflow** - Complete job application package with ATS analysis, cover letter, resume optimization, interview prep, role research, and learning plan.")
            
            gr.Markdown("<div style='margin-top: 30px;'></div>")
            
            with gr.Row():
                with gr.Column(scale=1):
                    pass
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
                    pass
            
            gr.Markdown("<div style='margin-top: 40px;'></div>")
            gr.Markdown("<h2 style='text-align: center;'>📊 Results Dashboard</h2>")
            gr.Markdown("<div style='margin-bottom: 20px;'></div>")
            
            with gr.Group():
                gr.Markdown("<h3>📊 ATS Analysis</h3>")
                ats_output = gr.Textbox(
                    label="",
                    lines=10,
                    show_copy_button=True,
                    placeholder="Results will appear here after running the agent...",
                    interactive=False
                )
            
            with gr.Group():
                gr.Markdown("<h3>📝 Resume Improvement Suggestions</h3>")
                bullets_output = gr.HTML(
                    label="",
                    value="<p style='color: #888; padding: 20px;'>Results will appear here after running the agent...</p>"
                )
            
            with gr.Group():
                gr.Markdown("<h3>✍️ Cover Letter</h3>")
                cover_letter_output = gr.Textbox(
                    label="",
                    lines=15,
                    show_copy_button=True,
                    placeholder="Results will appear here after running the agent...",
                    interactive=False
                )
            
            with gr.Group():
                gr.Markdown("<h3>✨ Optimized Resume Bullets</h3>")
                optimized_bullets_output = gr.HTML(
                    label="",
                    value="<p style='color: #888; padding: 20px;'>Results will appear here after running the agent...</p>"
                )
            
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
            
            resume_text_hidden = gr.State("")
            jd_text_hidden = gr.State("")
            
            with gr.Group():
                gr.Markdown("<h3>💼 Interview Preparation</h3>")
                interview_output = gr.HTML(
                    label="",
                    value="<p style='color: #888; padding: 20px;'>Results will appear here after running the agent...</p>"
                )
            
            with gr.Group():
                gr.Markdown("<h3>🔬 Role Expectations</h3>")
                role_output = gr.HTML(
                    label="",
                    value="<p style='color: #888; padding: 20px;'>Results will appear here after running the agent...</p>"
                )
            
            with gr.Group():
                gr.Markdown("<h3>📚 Skill Growth Plan</h3>")
                skill_growth_output = gr.HTML(
                    label="",
                    value="<p style='color: #888; padding: 20px;'>Results will appear here after running the agent...</p>"
                )
            
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
            
            full_report_output.change(
                fn=prepare_download,
                inputs=[full_report_output],
                outputs=[download_btn]
            )
        
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


