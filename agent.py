"""LangGraph agent for resume job application workflow."""

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
    generate_learning_plan
)


class AgentState(TypedDict):
    """State for the agent workflow."""
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


def parse_node(state: AgentState) -> AgentState:
    """Initial node - documents already parsed."""
    print("📄 Documents parsed and ready")
    return state


def ats_analysis_node(state: AgentState) -> AgentState:
    """Analyze ATS score and skill matching."""
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
    """Generate tailored cover letter."""
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
    """Optimize resume bullet points."""
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
    """Generate resume improvement suggestions for low ATS scores."""
    print("⚠️ Low ATS score detected - generating improvement suggestions...")
    
    try:
        suggestions = generate_resume_improvements(
            state["resume"],
            state["jd"],
            state["matched_skills"],
            state["missing_skills"]
        )
        state["improvement_suggestions"] = suggestions
        print("✅ Improvement suggestions generated")
    except Exception as e:
        print(f"❌ Error generating improvement suggestions: {str(e)}")
        state["improvement_suggestions"] = f"❌ Improvement suggestions failed: {str(e)}"
    
    return state


def interview_prep_node(state: AgentState) -> AgentState:
    """Generate interview questions and research role expectations."""
    print("💼 Generating interview questions...")
    
    try:
        questions = generate_interview_questions(state["jd"], state["resume"])
        state["interview_questions"] = questions
    except Exception as e:
        print(f"❌ Error generating interview questions: {str(e)}")
        state["interview_questions"] = f"❌ Interview questions generation failed: {str(e)}"
    
    print("🔬 Researching role expectations...")
    try:
        role_research = research_role_expectations(state["jd"], state.get("company_name", "this role"))
        state["role_expectations"] = role_research
    except Exception as e:
        print(f"❌ Error researching role expectations: {str(e)}")
        state["role_expectations"] = f"❌ Role research failed: {str(e)}"
    
    print("✅ Interview prep and role research completed")
    return state


def compile_output_node(state: AgentState) -> AgentState:
    """Generate learning plan from missing skills."""
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
    """Self-review node - skipped since we return structured data."""
    print("✅ Self-review skipped (using structured output)")
    return state


def revise_output_node(state: AgentState) -> AgentState:
    """Revise cover letter and bullets based on self-review."""
    print("✍️ Revising content for quality improvements...")
    
    try:
        # Simple revision prompt without full review
        revisions = revise_content(
            state["cover_letter"],
            state["optimized_bullets"],
            "Improve clarity, impact, and alignment with job requirements",
            state["resume"],
            state["jd"]
        )
        
        # Update state with revised content
        state["cover_letter"] = revisions["revised_cover_letter"]
        state["optimized_bullets"] = revisions["revised_bullets"]
        print("✅ Content revised")
    except Exception as e:
        print(f"❌ Error revising content: {str(e)}")
        # Keep original content if revision fails
    
    return state


def route_after_ats(state: AgentState) -> str:
    """Conditional routing based on ATS score."""
    if state["ats_score"] < 60:
        print(f"⚠️ ATS Score {state['ats_score']} < 60: Routing to resume_improvement")
        return "resume_improvement"
    else:
        print(f"✅ ATS Score {state['ats_score']} >= 60: Proceeding to cover_letter")
        return "generate_cover_letter"


def create_agent():
    """Create the LangGraph agent workflow with conditional routing."""
    
    # Create graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("parse", parse_node)
    workflow.add_node("ats_analysis", ats_analysis_node)
    workflow.add_node("resume_improvement", resume_improvement_node)
    workflow.add_node("generate_cover_letter", cover_letter_node)
    workflow.add_node("resume_optimizer", resume_optimizer_node)
    workflow.add_node("interview_prep", interview_prep_node)
    workflow.add_node("compile_output", compile_output_node)
    workflow.add_node("self_review", self_review_node)
    workflow.add_node("revise_output", revise_output_node)
    
    # Define edges (workflow)
    workflow.set_entry_point("parse")
    workflow.add_edge("parse", "ats_analysis")
    
    # Conditional routing after ATS analysis
    workflow.add_conditional_edges(
        "ats_analysis",
        route_after_ats,
        {
            "resume_improvement": "resume_improvement",
            "generate_cover_letter": "generate_cover_letter"
        }
    )
    
    # Both paths converge at cover_letter
    workflow.add_edge("resume_improvement", "generate_cover_letter")
    workflow.add_edge("generate_cover_letter", "resume_optimizer")
    workflow.add_edge("resume_optimizer", "interview_prep")
    workflow.add_edge("interview_prep", "compile_output")
    
    # Self-review and revision chain
    workflow.add_edge("compile_output", "self_review")
    workflow.add_edge("self_review", "revise_output")
    workflow.add_edge("revise_output", END)
    
    # Compile
    app = workflow.compile()
    return app


def run_agent(resume_text: str, jd_text: str, company_name: str = "the company") -> dict:
    """
    Run the complete agent workflow.
    
    Args:
        resume_text: Extracted resume text
        jd_text: Extracted job description text
        company_name: Company name for cover letter
        
    Returns:
        Dictionary with separate sections for UI display
    """
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
        "learning_plan": ""
    }
    
    # Run the agent
    final_state = agent.invoke(initial_state)
    
    # Build ATS section
    improvement_section = ""
    if final_state.get("improvement_suggestions"):
        improvement_section = f"\n\n### 📋 Improvement Suggestions\n\n{final_state['improvement_suggestions']}"
    
    ats_section = f"""### 📊 ATS Match Score: {final_state['ats_score']}/100

#### ✅ Matched Skills
{chr(10).join(f"- {skill}" for skill in final_state['matched_skills'][:10])}

#### ❌ Skills Gap
{chr(10).join(f"- {skill}" for skill in final_state['missing_skills'][:10])}
{improvement_section}
"""
    
    # Return structured dictionary
    return {
        "ats": ats_section,
        "cover_letter": final_state["cover_letter"],
        "bullets": final_state["optimized_bullets"],
        "interview": final_state["interview_questions"],
        "role_expectations": final_state["role_expectations"],
        "skill_growth": final_state["learning_plan"]
    }

