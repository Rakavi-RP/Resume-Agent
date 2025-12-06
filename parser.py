"""PDF Parser for extracting text from resume and job description files."""

import PyPDF2
from typing import Dict


def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract text content from a PDF file.
    
    Args:
        pdf_file: File object or path to PDF
        
    Returns:
        Extracted text as string
    """
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text.strip()
    except Exception as e:
        return f"Error extracting PDF: {str(e)}"


def parse_documents(resume_file, jd_file) -> Dict[str, str]:
    """
    Parse both resume and job description PDFs.
    
    Args:
        resume_file: Resume PDF file object
        jd_file: Job description PDF file object
        
    Returns:
        Dictionary with 'resume' and 'jd' text
    """
    resume_text = extract_text_from_pdf(resume_file)
    jd_text = extract_text_from_pdf(jd_file)
    
    return {
        "resume": resume_text,
        "jd": jd_text
    }
