import PyPDF2
from typing import Dict


def extract_text_from_pdf(pdf_file) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text.strip()
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"


def parse_documents(resume_file, jd_file) -> Dict[str, str]:
    resume_text = extract_text_from_pdf(resume_file)
    jd_text = extract_text_from_pdf(jd_file)
    
    return {
        "resume": resume_text,
        "jd": jd_text
    }
