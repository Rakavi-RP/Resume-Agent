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


def generate_resume_improvements(resume: str, jd: str, matched_skills: list, missing_skills: list) -> str:
    """
    Generate resume improvement suggestions for low ATS scores.
    
    Args:
        resume: Current resume text
        jd: Job description text
        matched_skills: Skills that matched
        missing_skills: Skills that are missing
        
    Returns:
        Detailed improvement suggestions
    """
    llm = create_llm()
    
    matched_str = ", ".join(matched_skills[:10])
    missing_str = ", ".join(missing_skills[:10])
    
    prompt = PromptTemplate(
        input_variables=["resume", "jd", "matched", "missing"],
        template="""You are a resume improvement expert helping candidates with low ATS scores.

Current Resume:
{resume}

Target Job Description:
{jd}

Matched Skills: {matched}
Missing Skills: {missing}

The candidate's ATS score is below 60, indicating significant gaps. Provide detailed, actionable improvement suggestions:

1. **Skills to Add**: Specific missing skills to incorporate and where/how to add them
2. **Keywords to Include**: Important keywords from the JD that are missing
3. **Experience Reframing**: How to reframe existing experience to better match requirements
4. **Certifications/Training**: Recommended certifications or courses to bridge gaps
5. **Resume Structure**: Formatting or structural changes to improve ATS parsing

Format your response with clear sections using headers (###) and bullet points (•).

Resume Improvement Suggestions:
"""
    )
    
    response = llm.invoke(prompt.format(
        resume=resume,
        jd=jd,
        matched=matched_str,
        missing=missing_str
    ))
    return response.content


def self_review_output(final_output: str, resume: str, jd: str) -> str:
    """
    LLM critiques the final job application package.
    
    Args:
        final_output: The compiled job application package
        resume: Original resume text
        jd: Job description text
        
    Returns:
        Review notes with suggestions for improvement
    """
    llm = create_llm()
    
    prompt = PromptTemplate(
        input_variables=["output", "resume", "jd"],
        template="""You are a senior career advisor reviewing a job application package.

Original Resume:
{resume}

Job Description:
{jd}

Generated Job Application Package:
{output}

Critically review this package and provide specific improvement suggestions for:

1. **Cover Letter Quality**: Tone, personalization, impact, alignment with JD
2. **Resume Bullets**: Clarity, quantification, action verbs, relevance
3. **Interview Questions**: Completeness, difficulty level, relevance
4. **Overall Coherence**: Consistency across all sections

Format your response with clear sections (###) and specific, actionable suggestions (•).
Be constructive but critical - identify real weaknesses.

Review Notes:
"""
    )
    
    response = llm.invoke(prompt.format(
        output=final_output,
        resume=resume,
        jd=jd
    ))
    return response.content


def revise_content(cover_letter: str, optimized_bullets: str, review_notes: str, resume: str, jd: str) -> dict:
    """
    Revise cover letter and resume bullets based on review notes.
    
    Args:
        cover_letter: Original cover letter
        optimized_bullets: Original resume bullets
        review_notes: Critique from self-review
        resume: Original resume
        jd: Job description
        
    Returns:
        Dict with revised_cover_letter and revised_bullets
    """
    llm = create_llm()
    
    # Revise cover letter
    cover_letter_prompt = PromptTemplate(
        input_variables=["cover_letter", "review_notes", "resume", "jd"],
        template="""You are improving a cover letter based on expert feedback.

Original Cover Letter:
{cover_letter}

Review Feedback:
{review_notes}

Original Resume:
{resume}

Job Description:
{jd}

Rewrite the cover letter addressing the feedback. Make it:
- More compelling and personalized
- Better aligned with the job requirements
- More impactful and professional
- 250-300 words

Revised Cover Letter:
"""
    )
    
    revised_cover_letter = llm.invoke(cover_letter_prompt.format(
        cover_letter=cover_letter,
        review_notes=review_notes,
        resume=resume,
        jd=jd
    )).content
    
    # Revise resume bullets
    bullets_prompt = PromptTemplate(
        input_variables=["bullets", "review_notes", "resume", "jd"],
        template="""You are improving resume bullet points based on expert feedback.

Original Bullets:
{bullets}

Review Feedback:
{review_notes}

Original Resume:
{resume}

Job Description:
{jd}

Rewrite the bullet points addressing the feedback. Make them:
- More quantifiable and specific
- Better action verbs
- More aligned with job requirements
- Following STAR method

Format each bullet starting with "•"

Revised Bullet Points:
"""
    )
    
    revised_bullets = llm.invoke(bullets_prompt.format(
        bullets=optimized_bullets,
        review_notes=review_notes,
        resume=resume,
        jd=jd
    )).content
    
    return {
        "revised_cover_letter": revised_cover_letter,
        "revised_bullets": revised_bullets
    }


def research_role_expectations(jd_text: str, job_title: str = "this role") -> str:
    """
    Research common skills, responsibilities, and expectations for a role.
    
    Args:
        jd_text: Job description text
        job_title: Title of the job role
        
    Returns:
        Detailed role expectations and industry standards
    """
    llm = create_llm()
    
    prompt = PromptTemplate(
        input_variables=["jd", "title"],
        template="""You are a career research expert analyzing role expectations.

Job Description:
{jd}

Job Title: {title}

Research and provide detailed insights about this role:

1. **Common Skills**: Industry-standard skills for this position
2. **Key Responsibilities**: Typical day-to-day duties and expectations
3. **Career Level**: Junior/Mid/Senior level indicators
4. **Industry Trends**: Current trends affecting this role
5. **Success Metrics**: How performance is typically measured
6. **Growth Path**: Common career progression from this role

Format your response with clear sections (###) and bullet points (•).

Role Expectations Research:
"""
    )
    
    response = llm.invoke(prompt.format(jd=jd_text, title=job_title))
    return response.content


def generate_learning_plan(missing_skills: list, matched_skills: list = None) -> str:
    """
    Generate a skill improvement roadmap based on missing skills.
    
    Args:
        missing_skills: List of skills the candidate is missing
        matched_skills: List of skills the candidate already has (optional)
        
    Returns:
        Structured learning plan with resources and timeline
    """
    llm = create_llm()
    
    missing_str = ", ".join(missing_skills[:15])
    matched_str = ", ".join(matched_skills[:10]) if matched_skills else "None specified"
    
    prompt = PromptTemplate(
        input_variables=["missing", "matched"],
        template="""You are a career development coach creating a skill improvement roadmap.

Skills to Acquire: {missing}
Current Skills: {matched}

Create a comprehensive learning plan to bridge the skills gap:

1. **Priority Skills** (Learn First): Top 3-5 most critical skills with rationale
2. **Learning Resources**: 
   - Online courses (Coursera, Udemy, etc.)
   - Books and documentation
   - Practice projects
   - Certifications worth pursuing
3. **Timeline**: Realistic 3-6 month learning roadmap
4. **Practice Projects**: Hands-on projects to build each skill
5. **Milestones**: Checkpoints to track progress

Format with clear sections (###) and actionable items (•).
Be specific with course names, book titles, and project ideas.

Skill Growth Plan:
"""
    )
    
    response = llm.invoke(prompt.format(missing=missing_str, matched=matched_str))
    return response.content

