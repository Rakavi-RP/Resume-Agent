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
    
    # Step 1: Extract skills from resume (only from specific sections)
    resume_prompt = PromptTemplate(
        input_variables=["resume"],
        template="""Extract ONLY explicit technical skills/tools from the following resume sections:
- Skills / Technical Skills section
- Projects section (tools/tech used)
- Experience section (tools/tech used)

DO NOT extract from:
- Profile summary
- Objective
- General descriptive text
- Education section

Resume:
{resume}

Return ONLY a comma-separated list of technical skills/tools explicitly mentioned.
Do NOT infer or hallucinate skills not explicitly stated.

Skills:"""
    )
    
    resume_skills_response = llm.invoke(resume_prompt.format(resume=resume))
    resume_skills = [s.strip() for s in resume_skills_response.content.split(',') if s.strip()]
    
    # Step 2: Extract required skills from job description
    jd_prompt = PromptTemplate(
        input_variables=["jd"],
        template="""Extract all required technical skills and tools from this job description.
Include both must-have and nice-to-have skills.

Job Description:
{jd}

Return ONLY a comma-separated list of technical skills/tools.

Skills:"""
    )
    
    jd_skills_response = llm.invoke(jd_prompt.format(jd=jd))
    jd_skills = [s.strip() for s in jd_skills_response.content.split(',') if s.strip()]
    
    # Step 3: Calculate matched and missing skills
    resume_skills_lower = {skill.lower() for skill in resume_skills}
    jd_skills_lower = {skill.lower() for skill in jd_skills}
    
    matched_skills_lower = resume_skills_lower.intersection(jd_skills_lower)
    missing_skills_lower = jd_skills_lower - resume_skills_lower
    
    # Map back to original case
    matched_skills = [skill for skill in jd_skills if skill.lower() in matched_skills_lower]
    missing_skills = [skill for skill in jd_skills if skill.lower() in missing_skills_lower]
    
    # Step 4: Calculate ATS score
    if len(jd_skills) > 0:
        score = int((len(matched_skills) / len(jd_skills)) * 100)
    else:
        score = 50  # Default if no skills found
    
    # Clamp score between 0 and 100
    score = max(0, min(100, score))
    
    return {
        "score": score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills
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

IMPORTANT: Use PLAIN TEXT only. Do NOT use markdown syntax like **bold** or ###headers.
Format each question numbered (1., 2., etc.)

Interview Questions:
"""
    )
    
    response = llm.invoke(prompt.format(jd=jd, resume=resume))
    return response.content


def generate_resume_improvements(resume: str, jd: str, matched_skills: list, missing_skills: list, ats_score: int = 0) -> str:
    """
    Generate resume improvement suggestions based on ATS score.
    
    Args:
        resume: Current resume text
        jd: Job description text
        matched_skills: Skills that matched
        missing_skills: Skills that are missing
        ats_score: ATS score (0-100)
        
    Returns:
        Improvement suggestions (conditional based on score)
    """
    # Conditional logic based on ATS score
    if ats_score >= 90:
        return ""  # No suggestions needed
    
    if ats_score >= 85:
        return "No major improvements needed. Your resume shows strong alignment with the job requirements."
    
    # For scores < 85, provide 2-3 crisp suggestions
    llm = create_llm()
    
    matched_str = ", ".join(matched_skills[:10])
    missing_str = ", ".join(missing_skills[:10])
    
    prompt = PromptTemplate(
        input_variables=["resume", "jd", "matched", "missing"],
        template="""You are a resume improvement expert.

Current Resume:
{resume}

Target Job Description:
{jd}

Matched Skills: {matched}
Missing Skills: {missing}

Provide ONLY 2-3 short, crisp, actionable suggestions to improve the resume.
Each suggestion should be 1-2 sentences maximum.
Focus on the most impactful changes.

IMPORTANT: Use PLAIN TEXT only. Do NOT use markdown syntax like **bold** or ###headers.
Format as a numbered list (1., 2., 3.).
NO long paragraphs. NO essays. Be concise.

Suggestions:"""
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

1. <b>Cover Letter Quality</b>: Tone, personalization, impact, alignment with JD
2. <b>Resume Bullets</b>: Clarity, quantification, action verbs, relevance
3. <b>Interview Questions</b>: Completeness, difficulty level, relevance
4. <b>Overall Coherence</b>: Consistency across all sections

Format your response with clear sections using HTML headers (<h4>) and specific, actionable suggestions (\u2022).
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

1. Common Skills: Industry-standard skills for this position
2. Key Responsibilities: Typical day-to-day duties and expectations
3. Career Level: Junior/Mid/Senior level indicators
4. Industry Trends: Current trends affecting this role
5. Success Metrics: How performance is typically measured
6. Growth Path: Common career progression from this role

IMPORTANT: Use PLAIN TEXT only. Do NOT use markdown syntax like **bold** or ###headers.
Format with numbered sections and bullet points (•).

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

1. Priority Skills (Learn First): Top 3-5 most critical skills with rationale
2. Learning Resources: 
   - Online courses (Coursera, Udemy, etc.)
   - Books and documentation
   - Practice projects
   - Certifications worth pursuing
3. Timeline: Realistic 3-6 month learning roadmap
4. Practice Projects: Hands-on projects to build each skill
5. Milestones: Checkpoints to track progress

IMPORTANT: Use PLAIN TEXT only. Do NOT use markdown syntax like **bold** or ###headers.
Format with numbered sections and bullet points (\u2022).
Be specific with course names, book titles, and project ideas.

Skill Growth Plan:
"""
    )
    
    response = llm.invoke(prompt.format(missing=missing_str, matched=matched_str))
    return response.content

