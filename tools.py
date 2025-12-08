"""Agent tools for resume analysis and job application assistance."""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from typing import Dict
import requests
import re


def create_llm():
    """Create Gemini LLM instance."""
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)


def github_company_research(jd_text: str, company_name: str = None) -> str:
    """
    Use GitHub API to research company's tech presence (no auth needed).
    
    Args:
        jd_text: Job description text
        company_name: Optional company name
        
    Returns:
        Plain text summary from GitHub
    """
    results = []
    
    # Extract probable role title from JD
    role_title = extract_role_title(jd_text)
    
    # Search for company on GitHub if provided
    if company_name:
        print(f"🔍 Searching GitHub for company: {company_name}")
        company_info = fetch_github_company_info(company_name)
        if company_info and "No GitHub organization" not in company_info:
            print(f"✅ Found GitHub info for {company_name}")
            results.append(company_info)
        else:
            print(f"❌ No GitHub org found for {company_name}")
    
    if results:
        return "\n\n".join(results)
    else:
        return "Could not find GitHub information for this company."


def extract_role_title(jd_text: str) -> str:
    """Extract probable role title from job description."""
    # Common role patterns
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
    
    # Fallback: look for common role keywords
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


def fetch_github_company_info(company_name: str) -> str:
    """
    Fetch company info from GitHub API (no auth needed).
    
    Args:
        company_name: Company name
        
    Returns:
        Plain text summary or None if not found
    """
    try:
        # Convert company name to likely GitHub org name
        org_name = company_name.lower().replace(' ', '')
        
        # Fetch organization info
        org_url = f"https://api.github.com/orgs/{org_name}"
        org_response = requests.get(org_url, timeout=10)
        
        if org_response.status_code == 200:
            org_data = org_response.json()
            
            # Fetch top repositories
            repos_url = f"https://api.github.com/orgs/{org_name}/repos?sort=stars&per_page=5"
            repos_response = requests.get(repos_url, timeout=10)
            
            if repos_response.status_code == 200:
                repos = repos_response.json()
                
                # Build summary
                summary_parts = []
                summary_parts.append(f"COMPANY: {company_name}")
                summary_parts.append(f"{org_data.get('name', company_name)} has {org_data.get('public_repos', 0)} public repositories on GitHub.")
                
                if org_data.get('description'):
                    summary_parts.append(org_data['description'])
                
                return "\n".join(summary_parts)
        
        return f"No GitHub organization found for {company_name}."
            
    except Exception:
        return None


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

IMPORTANT: Use PLAIN TEXT only. Do NOT use markdown syntax like **bold** or ###headers.

Cover Letter:
"""
    )
    
    response = llm.invoke(prompt.format(resume=resume, jd=jd, company=company_name))
    # Strip all bold markdown
    content = response.content
    print(f"🔍 Cover letter before strip: {content[:100]}...")
    content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
    print(f"✅ Cover letter after strip: {content[:100]}...")
    return content


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

IMPORTANT: Use PLAIN TEXT only. Do NOT use markdown syntax like **bold** or ###headers.
Format each bullet point starting with "•"

Improved Bullet Points:
"""
    )
    
    response = llm.invoke(prompt.format(resume=resume, jd=jd))
    # Strip all bold markdown
    content = response.content
    content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
    return content


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


def review_application_package(ats_score: int, matched_skills: list, missing_skills: list,
                               cover_letter: str, optimized_bullets: str, interview_questions: str,
                               role_expectations: str, learning_plan: str) -> str:
    """
    Review the complete job application package and provide critique.
    
    Args:
        ats_score: ATS match score
        matched_skills: Skills that matched
        missing_skills: Skills that are missing
        cover_letter: Generated cover letter
        optimized_bullets: Optimized resume bullets
        interview_questions: Interview questions
        role_expectations: Role expectations research
        learning_plan: Skill growth plan
        
    Returns:
        Review notes with strengths, weaknesses, and suggestions
    """
    llm = create_llm()
    
    # Build a concise summary of the package
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
        cover_letter=cover_letter[:500],  # First 500 chars to keep it efficient
        bullets=optimized_bullets[:300]   # First 300 chars
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


def refine_with_preference_tool(cover_letter: str, bullets: str, preference: str,
                                resume_text: str, jd_text: str) -> dict:
    """
    Refine cover letter and resume bullets based on a high-level user preference.
    This version avoids PydanticOutputParser to prevent JSON parsing errors.
    """
    llm = create_llm()

    # If user says it's fine, just return as-is
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

        # Simple parsing using the markers
        cover_part = ""
        bullets_part = ""

        # Split into two sections using our tags
        if "[COVER_LETTER]" in text and "[BULLETS]" in text:
            # Split once at [BULLETS]
            before, after = text.split("[BULLETS]", 1)
            # Remove the [COVER_LETTER] tag and strip
            cover_part = before.replace("[COVER_LETTER]", "").strip()
            bullets_part = after.strip()
        else:
            # Fallback: if model didn't follow format, keep original
            cover_part = cover_letter
            bullets_part = bullets

        # Safety fallback if any part is empty
        if not cover_part:
            cover_part = cover_letter
        if not bullets_part:
            bullets_part = bullets
        
        # Strip bold markdown from both
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
    """
    Generate context-specific refinement options using LLM.
    
    Args:
        resume_text: Original resume text
        jd_text: Job description text
        cover_letter: Generated cover letter
        bullets: Optimized resume bullets
        
    Returns:
        List of refinement option strings
    """
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
            resume=resume_text[:1500],  # Increased limit for better context
            jd=jd_text[:1500]
        ))
        
        # Parse response - look for [OPTIONS] section
        options_text = response.content.strip()
        
        # Try to extract options from [OPTIONS] section
        if '[OPTIONS]' in options_text:
            options_section = options_text.split('[OPTIONS]')[1].strip()
            options = [line.strip() for line in options_section.split('\n') if line.strip()]
        else:
            options = [line.strip() for line in options_text.split('\n') if line.strip()]
        
        # Filter and clean options
        cleaned_options = []
        for opt in options:
            # Remove leading numbers, bullets, or dashes
            cleaned = re.sub(r'^[\d\.\-\*\•\)]+\s*', '', opt)
            # Remove quotes if present
            cleaned = cleaned.strip('"\'')
            if cleaned and len(cleaned) > 10:  # Ensure meaningful options
                cleaned_options.append(cleaned)
        
        # Ensure we have at least 3 options, add defaults if needed
        if len(cleaned_options) < 3:
            cleaned_options.extend([
                "Make tone more professional",
                "Increase technical depth",
                "Focus on quantifiable achievements"
            ])
        
        # Return first 5 options
        return cleaned_options[:5]
        
    except Exception as e:
        print(f"Error generating refinement options: {str(e)}")
        # Fallback to generic options
        return [
            "Make tone more professional",
            "Increase technical depth",
            "Focus on quantifiable achievements",
            "Emphasize leadership and impact",
            "Make more concise"
        ]



