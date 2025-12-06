"""Agent tools for resume analysis and job application assistance."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from typing import Dict


def create_llm():
    """Create Gemini LLM instance."""
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)


def calculate_ats_score(resume: str, jd: str) -> Dict[str, any]:
    """
    Calculate ATS match score between resume and job description.
    
    Returns:
        Dict with score, matched_skills, missing_skills
    """
    llm = create_llm()
    
    prompt = PromptTemplate(
        input_variables=["resume", "jd"],
        template="""You are an ATS (Applicant Tracking System) analyzer.

Resume:
{resume}

Job Description:
{jd}

Analyze the match between this resume and job description. Provide:
1. ATS Score (0-100)
2. Matched Skills (list)
3. Missing Skills (list)

Format your response EXACTLY as:
SCORE: [number]
MATCHED: [skill1, skill2, skill3]
MISSING: [skill1, skill2, skill3]
"""
    )
    
    response = llm.invoke(prompt.format(resume=resume, jd=jd))
    result = response.content
    
    # Parse response
    score = 0
    matched = []
    missing = []
    
    for line in result.split('\n'):
        if line.startswith('SCORE:'):
            try:
                score = int(line.split(':')[1].strip())
            except:
                score = 50
        elif line.startswith('MATCHED:'):
            matched = [s.strip() for s in line.split(':')[1].split(',')]
        elif line.startswith('MISSING:'):
            missing = [s.strip() for s in line.split(':')[1].split(',')]
    
    return {
        "score": score,
        "matched_skills": matched,
        "missing_skills": missing
    }


def generate_cover_letter(resume: str, jd: str, company_name: str = "the company") -> str:
    """Generate tailored cover letter."""
    llm = create_llm()
    
    prompt = PromptTemplate(
        input_variables=["resume", "jd", "company"],
        template="""You are a professional cover letter writer.

Based on this resume and job description, write a compelling cover letter.

Resume:
{resume}

Job Description:
{jd}

Company: {company}

Write a professional, tailored cover letter (250-300 words) that:
- Highlights relevant experience from the resume
- Addresses key requirements from the JD
- Shows enthusiasm for the role
- Is personalized and compelling

Cover Letter:
"""
    )
    
    response = llm.invoke(prompt.format(resume=resume, jd=jd, company=company_name))
    return response.content


def optimize_resume_bullets(resume: str, jd: str) -> str:
    """Generate improved resume bullet points."""
    llm = create_llm()
    
    prompt = PromptTemplate(
        input_variables=["resume", "jd"],
        template="""You are a resume optimization expert.

Current Resume:
{resume}

Target Job Description:
{jd}

Provide 5-7 improved bullet points that:
- Use action verbs
- Include quantifiable achievements
- Align with the job requirements
- Follow the STAR method (Situation, Task, Action, Result)

Format each bullet point starting with "•"

Improved Bullet Points:
"""
    )
    
    response = llm.invoke(prompt.format(resume=resume, jd=jd))
    return response.content


def generate_interview_questions(jd: str, resume: str) -> str:
    """Generate likely interview questions."""
    llm = create_llm()
    
    prompt = PromptTemplate(
        input_variables=["jd", "resume"],
        template="""You are an interview preparation coach.

Job Description:
{jd}

Candidate Resume:
{resume}

Generate 8-10 likely interview questions for this role, including:
- Technical questions based on required skills
- Behavioral questions
- Questions about gaps or concerns in the resume
- Company/role-specific questions

Format each question numbered (1., 2., etc.)

Interview Questions:
"""
    )
    
    response = llm.invoke(prompt.format(jd=jd, resume=resume))
    return response.content
