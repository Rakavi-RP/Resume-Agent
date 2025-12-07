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
    final_output: str
    review_notes: str


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
    """Compile final job-ready output."""
    print("📦 Compiling final output...")
    
    # Generate learning plan from missing skills
    print("📚 Generating skill learning plan...")
    try:
        learning_plan = generate_learning_plan(state["missing_skills"], state["matched_skills"])
        state["learning_plan"] = learning_plan
    except Exception as e:
        print(f"❌ Error generating learning plan: {str(e)}")
        state["learning_plan"] = f"❌ Learning plan generation failed: {str(e)}"
    
    # Include improvement suggestions if ATS score is low
    improvement_section = ""
    if state.get("improvement_suggestions"):
        improvement_section = f"""
═══════════════════════════════════════════════════════════════
                   RESUME IMPROVEMENT SUGGESTIONS
═══════════════════════════════════════════════════════════════

{state['improvement_suggestions']}

"""
    
    output = f"""
═══════════════════════════════════════════════════════════════
                    JOB APPLICATION PACKAGE
═══════════════════════════════════════════════════════════════

📊 ATS MATCH SCORE: {state['ats_score']}/100

✅ MATCHED SKILLS:
{chr(10).join(f"  • {skill}" for skill in state['matched_skills'][:10])}

❌ SKILLS GAP:
{chr(10).join(f"  • {skill}" for skill in state['missing_skills'][:10])}
{improvement_section}
═══════════════════════════════════════════════════════════════
                        COVER LETTER
═══════════════════════════════════════════════════════════════

{state['cover_letter']}

═══════════════════════════════════════════════════════════════
                   OPTIMIZED RESUME BULLETS
═══════════════════════════════════════════════════════════════

{state['optimized_bullets']}

═══════════════════════════════════════════════════════════════
                    INTERVIEW PREPARATION
═══════════════════════════════════════════════════════════════

{state['interview_questions']}

═══════════════════════════════════════════════════════════════
                      ROLE EXPECTATIONS
═══════════════════════════════════════════════════════════════

{state['role_expectations']}

═══════════════════════════════════════════════════════════════
                      SKILL GROWTH PLAN
═══════════════════════════════════════════════════════════════

{state['learning_plan']}

═══════════════════════════════════════════════════════════════
                    Generated by Resume Job Agent
                    Powered by LangGraph + Gemini
═══════════════════════════════════════════════════════════════
"""
    
    state["final_output"] = output
    print("✅ Initial output compiled")
    return state


def self_review_node(state: AgentState) -> AgentState:
    """Self-review the final output and generate improvement notes."""
    print("🔍 Reviewing output for quality improvements...")
    
    try:
        review_notes = self_review_output(
            state["final_output"],
            state["resume"],
            state["jd"]
        )
        state["review_notes"] = review_notes
        print("✅ Review notes generated")
    except Exception as e:
        print(f"❌ Error in self-review: {str(e)}")
        state["review_notes"] = f"Self-review skipped due to error: {str(e)}"
    
    return state


def revise_output_node(state: AgentState) -> AgentState:
    """Revise cover letter and bullets based on review notes."""
    print("✍️ Revising content based on review feedback...")
    
    try:
        revisions = revise_content(
            state["cover_letter"],
            state["optimized_bullets"],
            state["review_notes"],
            state["resume"],
            state["jd"]
        )
        
        # Update state with revised content
        state["cover_letter"] = revisions["revised_cover_letter"]
        state["optimized_bullets"] = revisions["revised_bullets"]
    except Exception as e:
        print(f"❌ Error revising content: {str(e)}")
        # Keep original content if revision fails
    
    # Recompile final output with revised content
    improvement_section = ""
    if state.get("improvement_suggestions"):
        improvement_section = f"""
═══════════════════════════════════════════════════════════════
                   RESUME IMPROVEMENT SUGGESTIONS
═══════════════════════════════════════════════════════════════

{state['improvement_suggestions']}

"""
    
    revised_output = f"""
═══════════════════════════════════════════════════════════════
                    JOB APPLICATION PACKAGE
                         (REVISED VERSION)
═══════════════════════════════════════════════════════════════

📊 ATS MATCH SCORE: {state['ats_score']}/100

✅ MATCHED SKILLS:
{chr(10).join(f"  • {skill}" for skill in state['matched_skills'][:10])}

❌ SKILLS GAP:
{chr(10).join(f"  • {skill}" for skill in state['missing_skills'][:10])}
{improvement_section}
═══════════════════════════════════════════════════════════════
                        COVER LETTER
═══════════════════════════════════════════════════════════════

{state['cover_letter']}

═══════════════════════════════════════════════════════════════
                   OPTIMIZED RESUME BULLETS
═══════════════════════════════════════════════════════════════

{state['optimized_bullets']}

═══════════════════════════════════════════════════════════════
                    INTERVIEW PREPARATION
═══════════════════════════════════════════════════════════════

{state['interview_questions']}

═══════════════════════════════════════════════════════════════
                      ROLE EXPECTATIONS
═══════════════════════════════════════════════════════════════

{state['role_expectations']}

═══════════════════════════════════════════════════════════════
                      SKILL GROWTH PLAN
═══════════════════════════════════════════════════════════════

{state['learning_plan']}

═══════════════════════════════════════════════════════════════
                    Generated by Resume Job Agent
                    Powered by LangGraph + Gemini
                         Self-Reviewed & Revised
═══════════════════════════════════════════════════════════════
"""
    
    state["final_output"] = revised_output
    print("✅ Content revised and output recompiled")
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


def run_agent(resume_text: str, jd_text: str, company_name: str = "the company") -> str:
    """
    Run the complete agent workflow.
    
    Args:
        resume_text: Extracted resume text
        jd_text: Extracted job description text
        company_name: Company name for cover letter
        
    Returns:
        Final compiled output text
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
        "learning_plan": "",
        "final_output": "",
        "review_notes": ""
    }
    
    # Run the agent
    final_state = agent.invoke(initial_state)
    
    return final_state["final_output"]
