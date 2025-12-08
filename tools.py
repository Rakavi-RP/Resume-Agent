from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from typing import Dict
import requests
import re


def create_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)


def extract_role_title(jd_text: str) -> str:
    patterns = [
        r'(?:Job Title|Position|Role):\s*([^\n]+)',
        r'(?:hiring|seeking|looking for)\s+(?:a\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})',
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})\s*(?:position|role)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, jd_text, re.IGNORECASE | re.MULTILINE)
        if match:
            title = match.group(1).strip()
            # Clean up common suffixes
            title = re.sub(r'\s*\(.*?\)\s*', '', title)
            return title
    
    role_keywords = [
        'Software Engineer', 'Data Scientist', 'Machine Learning Engineer',
        'Product Manager', 'Data Analyst', 'DevOps Engineer',
        'Full Stack Developer', 'Backend Developer', 'Frontend Developer'
    ]
    
    jd_lower = jd_text.lower()
    for role in role_keywords:
        if role.lower() in jd_lower:
            return role
    
    return None


def calculate_ats_score(resume: str, jd: str) -> Dict[str, any]:
    llm = create_llm()
    
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
    
    resume_skills_lower = {skill.lower() for skill in resume_skills}
    jd_skills_lower = {skill.lower() for skill in jd_skills}
    
    matched_skills_lower = resume_skills_lower.intersection(jd_skills_lower)
    missing_skills_lower = jd_skills_lower - resume_skills_lower
    
    matched_skills = [skill for skill in jd_skills if skill.lower() in matched_skills_lower]
    missing_skills = [skill for skill in jd_skills if skill.lower() in missing_skills_lower]
    
    if len(jd_skills) > 0:
        score = int((len(matched_skills) / len(jd_skills)) * 100)
    else:
        score = 50
    
    score = max(0, min(100, score))
    
    return {
        "score": score,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills
    }


def generate_cover_letter(resume: str, jd: str, company_name: str = "the company") -> str:
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

IMPORTANT: Use PLAIN TEXT only. Do NOT use markdown syntax like **bold** or ###headers.

Cover Letter:
"""
    )
    
    response = llm.invoke(prompt.format(resume=resume, jd=jd, company=company_name))
    content = response.content
    print(f"🔍 Cover letter before strip: {content[:100]}...")
    content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
    print(f"✅ Cover letter after strip: {content[:100]}...")
    return content


def optimize_resume_bullets(resume: str, jd: str) -> str:
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

IMPORTANT: Use PLAIN TEXT only. Do NOT use markdown syntax like **bold** or ###headers.
Format each bullet point starting with "•"

Improved Bullet Points:
"""
    )
    
    response = llm.invoke(prompt.format(resume=resume, jd=jd))
    content = response.content
    content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
    return content


def generate_interview_questions(jd: str, resume: str) -> str:
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
    if ats_score >= 90:
        return ""
    
    if ats_score >= 85:
        return "No major improvements needed. Your resume shows strong alignment with the job requirements."
    
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


def review_application_package(ats_score: int, matched_skills: list, missing_skills: list,
                               cover_letter: str, optimized_bullets: str, interview_questions: str,
                               role_expectations: str, learning_plan: str) -> str:
    llm = create_llm()
    
    package_summary = f"""
ATS Score: {ats_score}/100
Matched Skills: {len(matched_skills)} skills
Missing Skills: {len(missing_skills)} skills

Cover Letter Length: {len(cover_letter.split())} words
Resume Bullets: {len(optimized_bullets.split(chr(10)))} lines
Interview Questions: {len(interview_questions.split(chr(10)))} items
"""
    
    prompt = PromptTemplate(
        input_variables=["package_summary", "cover_letter", "bullets"],
        template="""You are a senior career advisor reviewing a job application package.

Package Summary:
{package_summary}

Cover Letter:
{cover_letter}

Resume Bullets:
{bullets}

Provide a CONCISE review (3-4 sentences max) covering:
1. STRENGTHS: What's working well
2. WEAKNESSES: What needs improvement
3. SUGGESTIONS: 1-2 quick actionable fixes

IMPORTANT: Use PLAIN TEXT only. Be brief and direct.

Review:"""
    )
    
    response = llm.invoke(prompt.format(
        package_summary=package_summary,
        cover_letter=cover_letter[:500],
        bullets=optimized_bullets[:300]
    ))
    
    return response.content


def self_review_output(final_output: str, resume: str, jd: str) -> str:
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
    llm = create_llm()
    
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


def refine_with_preference_tool(cover_letter: str, bullets: str, preference: str,
                                resume_text: str, jd_text: str) -> dict:
    llm = create_llm()

    if preference == "Looks good as is":
        return {"cover_letter": cover_letter, "bullets": bullets}

    prompt = PromptTemplate(
        input_variables=["cover_letter", "bullets", "preference", "resume", "jd"],
        template="""You are a professional resume editor. The user wants to refine their application materials.

Current Cover Letter:
{cover_letter}

Current Resume Bullets:
{bullets}

User Preference: "{preference}"

Original Resume (for fact-checking):
{resume}

Job Description:
{jd}

Task:
REWRITE the Cover Letter and Resume Bullets to STRONGLY emphasize the User Preference.
- Make SIGNIFICANT, VISIBLE changes to highlight the requested aspect
- Add specific keywords and phrases related to the preference throughout
- Reorganize content to put preference-related items FIRST
- Use stronger, more specific language for the emphasized areas
- Ensure all details are FACTUALLY SUPPORTED by the Original Resume
- Make the changes OBVIOUS and CLEAR to demonstrate the refinement

Output Format:
Return the result in exactly this format:

[COVER_LETTER]
<refined cover letter text here>

[BULLETS]
<refined bullets here, starting each bullet with "•">
"""
    )

    try:
        response = llm.invoke(prompt.format(
            cover_letter=cover_letter,
            bullets=bullets,
            preference=preference,
            resume=resume_text,
            jd=jd_text
        ))
        text = response.content or ""

        cover_part = ""
        bullets_part = ""

        if "[COVER_LETTER]" in text and "[BULLETS]" in text:
            before, after = text.split("[BULLETS]", 1)
            cover_part = before.replace("[COVER_LETTER]", "").strip()
            bullets_part = after.strip()
        else:
            cover_part = cover_letter
            bullets_part = bullets

        if not cover_part:
            cover_part = cover_letter
        if not bullets_part:
            bullets_part = bullets
        
        cover_part = re.sub(r'\*\*([^*]+)\*\*', r'\1', cover_part)
        bullets_part = re.sub(r'\*\*([^*]+)\*\*', r'\1', bullets_part)

        return {
            "cover_letter": cover_part,
            "bullets": bullets_part,
        }

    except Exception as e:
        print(f"Refine tool failed, returning original content: {e}")
        return {"cover_letter": cover_letter, "bullets": bullets}


def generate_refinement_options(resume_text: str, jd_text: str, cover_letter: str, bullets: str) -> list:
    llm = create_llm()
    
    prompt = PromptTemplate(
        input_variables=["resume", "jd"],
        template="""You are helping the user refine a job application package.

Here is the resume:
{resume}

Here is the job description:
{jd}

Based on these, suggest 3–4 SHORT refinement options the user could choose from
to adjust their application. Each option should be:

- 1 short sentence or phrase
- Specific to THIS role and THIS resume
- Focused on *how* to adjust emphasis or tone (not rewriting everything)

Examples of option styles:
- "Highlight backend and system design more"
- "Emphasize cloud, DevOps and reliability"
- "Focus more on front-end UI and UX impact"
- "Make the tone more concise and results-driven"

Return them EXACTLY in this format:

[OPTIONS]
1. ...
2. ...
3. ...
4. ...
"""
    )
    
    
    try:
        response = llm.invoke(prompt.format(
            resume=resume_text[:1500],
            jd=jd_text[:1500]
        ))
        
        options_text = response.content.strip()
        
        if '[OPTIONS]' in options_text:
            options_section = options_text.split('[OPTIONS]')[1].strip()
            options = [line.strip() for line in options_section.split('\n') if line.strip()]
        else:
            options = [line.strip() for line in options_text.split('\n') if line.strip()]
        
        cleaned_options = []
        for opt in options:
            cleaned = re.sub(r'^[\d\.\-\*\•\)]+\s*', '', opt)
            cleaned = cleaned.strip('"\'')
            if cleaned and len(cleaned) > 10:
                cleaned_options.append(cleaned)
        
        if len(cleaned_options) < 3:
            cleaned_options.extend([
                "Make tone more professional",
                "Increase technical depth",
                "Focus on quantifiable achievements"
            ])
        
        return cleaned_options[:5]
        
    except Exception as e:
        print(f"Error generating refinement options: {str(e)}")
        return [
            "Make tone more professional",
            "Increase technical depth",
            "Focus on quantifiable achievements",
            "Emphasize leadership and impact",
            "Make more concise"
        ]



