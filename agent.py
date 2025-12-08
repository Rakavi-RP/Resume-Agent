from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
import operator
from tools import (
    calculate_ats_score,
    generate_cover_letter,
    optimize_resume_bullets,
    generate_interview_questions,
    generate_resume_improvements,
    self_review_output,
    revise_content,
    research_role_expectations,
    generate_learning_plan,
    review_application_package
)


class AgentState(TypedDict):
    resume: str
    jd: str
    company_name: str
    ats_score: int
    matched_skills: list
    missing_skills: list
    improvement_suggestions: str
    cover_letter: str
    optimized_bullets: str
    interview_questions: str
    role_expectations: str
    learning_plan: str
    review_notes: str


def parse_node(state: AgentState) -> AgentState:
    print("📄 Documents parsed and ready")
    return state


def ats_analysis_node(state: AgentState) -> AgentState:
    print("🔍 Analyzing ATS score...")
    
    try:
        result = calculate_ats_score(state["resume"], state["jd"])
        
        state["ats_score"] = result["score"]
        state["matched_skills"] = result["matched_skills"]
        state["missing_skills"] = result["missing_skills"]
        
        print(f"✅ ATS Score: {result['score']}/100")
    except Exception as e:
        print(f"❌ Error in ATS analysis: {str(e)}")
        state["ats_score"] = 50
        state["matched_skills"] = ["Error analyzing skills"]
        state["missing_skills"] = [f"ATS analysis failed: {str(e)}"]
    
    return state


def cover_letter_node(state: AgentState) -> AgentState:
    print("✍️ Generating cover letter...")
    
    try:
        cover_letter = generate_cover_letter(
            state["resume"],
            state["jd"],
            state.get("company_name", "the company")
        )
        state["cover_letter"] = cover_letter
        print("✅ Cover letter generated")
    except Exception as e:
        print(f"❌ Error generating cover letter: {str(e)}")
        state["cover_letter"] = f"❌ Cover letter generation failed: {str(e)}\n\nPlease check your API key and try again."
    
    return state


def resume_optimizer_node(state: AgentState) -> AgentState:
    print("📝 Optimizing resume bullets...")
    
    try:
        bullets = optimize_resume_bullets(state["resume"], state["jd"])
        state["optimized_bullets"] = bullets
        print("✅ Resume bullets optimized")
    except Exception as e:
        print(f"❌ Error optimizing resume bullets: {str(e)}")
        state["optimized_bullets"] = f"❌ Resume optimization failed: {str(e)}"
    
    return state


def resume_improvement_node(state: AgentState) -> AgentState:
    print(f"⚠️ ATS Score {state['ats_score']} - generating improvement suggestions...")
    
    try:
        suggestions = generate_resume_improvements(
            state["resume"],
            state["jd"],
            state["matched_skills"],
            state["missing_skills"],
            state["ats_score"]
        )
        state["improvement_suggestions"] = suggestions
        if suggestions:
            print("✅ Improvement suggestions generated")
        else:
            print("✅ No suggestions needed (high ATS score)")
    except Exception as e:
        print(f"❌ Error generating improvement suggestions: {str(e)}")
        state["improvement_suggestions"] = ""
    
    return state


def interview_prep_node(state: AgentState) -> AgentState:
    print("💼 Generating interview questions...")
    
    try:
        questions = generate_interview_questions(state["jd"], state["resume"])
        state["interview_questions"] = questions
    except Exception as e:
        print(f"❌ Error generating interview questions: {str(e)}")
        state["interview_questions"] = f"❌ Interview questions generation failed: {str(e)}"
    
    print("🔬 Researching role expectations...")
    try:
        state["role_expectations"] = research_role_expectations(state["jd"], state.get("company_name", "this role"))
    except Exception as e:
        print(f"❌ Error researching role expectations: {str(e)}")
        state["role_expectations"] = f"❌ Role research failed: {str(e)}"
    
    print("✅ Interview prep and role research completed")
    return state


def compile_output_node(state: AgentState) -> AgentState:
    print("📚 Generating skill learning plan...")
    
    try:
        learning_plan = generate_learning_plan(state["missing_skills"], state["matched_skills"])
        state["learning_plan"] = learning_plan
        print("✅ Learning plan generated")
    except Exception as e:
        print(f"❌ Error generating learning plan: {str(e)}")
        state["learning_plan"] = f"❌ Learning plan generation failed: {str(e)}"
    
    return state


def self_review_node(state: AgentState) -> AgentState:
    print("🕵️ Running self-review on generated content...")
    
    try:
        from tools import review_application_package
        
        review_notes = review_application_package(
            ats_score=state["ats_score"],
            matched_skills=state["matched_skills"],
            missing_skills=state["missing_skills"],
            cover_letter=state["cover_letter"],
            optimized_bullets=state["optimized_bullets"],
            interview_questions=state["interview_questions"],
            role_expectations=state["role_expectations"],
            learning_plan=state["learning_plan"]
        )
        
        state["review_notes"] = review_notes
        print("✅ Review notes generated")
    except Exception as e:
        print(f"❌ Error in self-review: {str(e)}")
        state["review_notes"] = "Review skipped due to error."
    
    return state


def revise_output_node(state: AgentState) -> AgentState:
    print("✍️ Revising content based on review notes...")
    
    try:
        from tools import revise_content
        
        revisions = revise_content(
            cover_letter=state["cover_letter"],
            optimized_bullets=state["optimized_bullets"],
            review_notes=state["review_notes"],
            resume=state["resume"],
            jd=state["jd"]
        )
        
        state["cover_letter"] = revisions["revised_cover_letter"]
        state["optimized_bullets"] = revisions["revised_bullets"]
        print("✅ Content revised based on review feedback")
    except Exception as e:
        print(f"❌ Error revising content: {str(e)}")
    
    return state


def deep_resume_improvement_node(state: AgentState) -> AgentState:
    print(f"🚨 Low ATS Score {state['ats_score']} (<70) - generating deep restructuring suggestions...")
    
    try:
        suggestions = generate_resume_improvements(
            state["resume"],
            state["jd"],
            state["matched_skills"],
            state["missing_skills"],
            state["ats_score"]
        )
        
        if suggestions:
            suggestions = f"⚠️ DEEP RESUME RESTRUCTURING NEEDED (ATS Score: {state['ats_score']})\n\n{suggestions}"
        
        state["improvement_suggestions"] = suggestions
        print("✅ Deep improvement suggestions generated")
    except Exception as e:
        print(f"❌ Error generating deep improvement suggestions: {str(e)}")
        state["improvement_suggestions"] = ""
    
    return state


def route_after_ats(state: AgentState) -> str:
    score = state['ats_score']
    
    if score >= 90:
        print(f"🎯 High ATS Score {score} (≥90): Skipping to cover_letter")
        return "cover_letter"
    elif score >= 70:
        print(f"✅ Good ATS Score {score} (70-89): Routing to resume_improvement")
        return "resume_improvement"
    else:
        print(f"🚨 Low ATS Score {score} (<70): Routing to deep_resume_improvement")
        return "deep_resume_improvement"


def create_agent():
    workflow = StateGraph(AgentState)
    
    workflow.add_node("parse", parse_node)
    workflow.add_node("ats_analysis", ats_analysis_node)
    workflow.add_node("resume_improvement", resume_improvement_node)
    workflow.add_node("deep_resume_improvement", deep_resume_improvement_node)
    workflow.add_node("generate_cover_letter", cover_letter_node)
    workflow.add_node("resume_optimizer", resume_optimizer_node)
    workflow.add_node("interview_prep", interview_prep_node)
    workflow.add_node("compile_output", compile_output_node)
    workflow.add_node("self_review", self_review_node)
    workflow.add_node("revise_output", revise_output_node)
    
    workflow.set_entry_point("parse")
    workflow.add_edge("parse", "ats_analysis")
    
    workflow.add_conditional_edges(
        "ats_analysis",
        route_after_ats,
        {
            "cover_letter": "generate_cover_letter",
            "resume_improvement": "resume_improvement",
            "deep_resume_improvement": "deep_resume_improvement"
        }
    )
    
    workflow.add_edge("resume_improvement", "generate_cover_letter")
    workflow.add_edge("deep_resume_improvement", "generate_cover_letter")
    
    workflow.add_edge("generate_cover_letter", "resume_optimizer")
    workflow.add_edge("resume_optimizer", "interview_prep")
    workflow.add_edge("interview_prep", "compile_output")
    
    workflow.add_edge("compile_output", "self_review")
    workflow.add_edge("self_review", "revise_output")
    workflow.add_edge("revise_output", END)
    
    app = workflow.compile()
    return app


def run_agent(resume_text: str, jd_text: str, company_name: str = "the company") -> dict:
    agent = create_agent()
    
    initial_state = {
        "resume": resume_text,
        "jd": jd_text,
        "company_name": company_name,
        "ats_score": 0,
        "matched_skills": [],
        "missing_skills": [],
        "improvement_suggestions": "",
        "cover_letter": "",
        "optimized_bullets": "",
        "interview_questions": "",
        "role_expectations": "",
        "learning_plan": "",
        "review_notes": ""
    }
    
    final_state = agent.invoke(initial_state)
    
    matched_skills_text = "\n".join(f"  • {skill}" for skill in final_state['matched_skills'][:15])
    missing_skills_text = "\n".join(f"  • {skill}" for skill in final_state['missing_skills'][:15])
    
    ats_section = f"""ATS MATCH SCORE: {final_state['ats_score']}/100

MATCHED SKILLS:
{matched_skills_text}

MISSING SKILLS:
{missing_skills_text}
"""
    
    resume_suggestions_section = final_state.get("improvement_suggestions", "")
    
    cover_letter_section = final_state["cover_letter"]
    
    bullets_section = final_state["optimized_bullets"]
    
    interview_section = final_state["interview_questions"]
    
    role_expectations_section = final_state["role_expectations"]
    
    skill_growth_section = final_state["learning_plan"]
    
    full_report = f"""
{'='*80}
COMPLETE JOB APPLICATION PACKAGE
{'='*80}

{ats_section}

{'='*80}
RESUME IMPROVEMENT SUGGESTIONS
{'='*80}

{resume_suggestions_section if resume_suggestions_section else "No additional suggestions needed."}

{'='*80}
COVER LETTER
{'='*80}

{cover_letter_section}

{'='*80}
OPTIMIZED RESUME BULLETS
{'='*80}

{bullets_section}

{'='*80}
INTERVIEW PREPARATION
{'='*80}

{interview_section}

{'='*80}
ROLE EXPECTATIONS & RESEARCH
{'='*80}

{role_expectations_section}

{'='*80}
SKILL GROWTH PLAN
{'='*80}

{skill_growth_section}

{'='*80}
END OF REPORT
{'='*80}
"""
    
    return {
        "ats_section": ats_section,
        "resume_suggestions_section": resume_suggestions_section,
        "cover_letter_section": cover_letter_section,
        "bullets_section": bullets_section,
        "interview_section": interview_section,
        "role_expectations_section": role_expectations_section,
        "skill_growth_section": skill_growth_section,
        "full_report": full_report
    }

