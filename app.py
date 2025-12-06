"""Gradio UI for Resume Job Application Agent."""

import gradio as gr
from dotenv import load_dotenv
import os
from parser import parse_documents
from agent import run_agent

# Load environment variables
load_dotenv()

# Verify API key
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in .env file!")


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
        
        # Run agent
        output = run_agent(
            docs["resume"],
            docs["jd"],
            company_name or "the company"
        )
        
        return output
    
    except Exception as e:
        return f"❌ Error: {str(e)}\n\nPlease check your API key and try again."


# Create Gradio interface
with gr.Blocks(title="Resume Job Application Agent", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown(
        """
        # 🎯 Resume → Job Application Agent
        ### Powered by LangGraph + Google Gemini
        
        Upload your resume and job description to get:
        - ✅ ATS Match Score
        - ✍️ Tailored Cover Letter
        - 📝 Optimized Resume Bullets
        - 💼 Interview Prep Questions
        """
    )
    
    with gr.Row():
        with gr.Column():
            resume_input = gr.File(
                label="📄 Upload Resume (PDF)",
                file_types=[".pdf"]
            )
            jd_input = gr.File(
                label="📋 Upload Job Description (PDF)",
                file_types=[".pdf"]
            )
            company_input = gr.Textbox(
                label="🏢 Company Name (Optional)",
                placeholder="e.g., Google, Microsoft, Startup Inc."
            )
            
            submit_btn = gr.Button("🚀 Generate Job Application Package", variant="primary")
        
        with gr.Column():
            output = gr.Textbox(
                label="📦 Job-Ready Output",
                lines=25,
                max_lines=30,
                show_copy_button=True
            )
            
            #download_btn = gr.DownloadButton(
            #    label="💾 Download as .txt",
            #    visible=True
            #)
    
    # Event handlers
    submit_btn.click(
        fn=process_application,
        inputs=[resume_input, jd_input, company_input],
        outputs=output
    )
    
    # Update download button with output
    #output.change(
        #lambda text: gr.DownloadButton(
            #label="💾 Download Job Application Package",
            #value=text,
            #visible=bool(text)
        #),
        #inputs=output,
        #outputs=download_btn
    #)
    
    gr.Markdown(
        """
        ---
        **Note:** Processing takes 30-60 seconds. Make sure your PDFs are text-based (not scanned images).
        """
    )


if __name__ == "__main__":
    demo.launch(share=False)
