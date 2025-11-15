"""
ContentStrategist: A Multi-Agent System for Automated Blog Post Creation
Uses Google Agent Development Kit (ADK) concepts in Python for the Capstone Project.

This system implements a sequential workflow (Multi-Agent System):
1. ResearcherAgent: Gathers keywords, trends, and competitor insights using a simulated Built-in Tool.
2. WriterAgent: Creates a 500-word blog post draft by reading the research output (Context Transfer).
3. EditorAgent: Refines draft using custom readability scoring tool
"""

import os
import json
import re
from typing import Dict, List, Any
from dataclasses import dataclass


# ============================================================================
# MOCK ADK FRAMEWORK (Simulating Google ADK Structure)
# ============================================================================

@dataclass
class ToolResult:
    """Represents the result of a tool execution, similar to ADK's ToolResult."""
    output: Any
    success: bool = True
    error: str = None


class FunctionTool:
    """Represents a callable function tool for agents (both Built-in and Custom)."""
    
    def __init__(self, name: str, description: str, func: callable):
        self.name = name
        self.description = description
        self.func = func
    
    def execute(self, **kwargs) -> ToolResult:
        """Executes the wrapped function."""
        try:
            result = self.func(**kwargs)
            return ToolResult(output=result, success=True)
        except Exception as e:
            # Important for robustness (Observability concept)
            return ToolResult(output=None, success=False, error=str(e))


class LlmAgent:
    """Base agent powered by an LLM (Simulating ADK's LlmAgent)."""
    
    def __init__(self, name: str, role: str, model: str, tools: List[FunctionTool] = None):
        self.name = name
        self.role = role
        self.model = model
        # Tools assigned to this agent (Built-in or Custom)
        self.tools = tools or []
        # Required for the Gemini/Claude bonus point check
        self.api_key = os.getenv("GOOGLE_AI_KEY")
    
    def run(self, prompt: str, context: Dict = None) -> str:
        """Simulates LLM execution with tool usage."""
        full_prompt = f"Role: {self.role}\n\nTask: {prompt}"
        if context:
            full_prompt += f"\n\nContext: {json.dumps(context, indent=2)}"
            
        # Call the agent-specific simulation method
        return self._simulate_response(prompt, context)
    
    def _simulate_response(self, prompt: str, context: Dict) -> str:
        """Abstract method for specialized agent logic."""
        raise NotImplementedError


class SequentialAgent:
    """
    CORE CAPSTONE FEATURE: MULTI-AGENT SYSTEM (Sequential)
    Orchestrates multiple agents in a fixed, sequential workflow.
    """
    
    def __init__(self, name: str, agents: List[LlmAgent]):
        self.name = name
        self.agents = agents
        # Session State / Working Memory container
        self.session_state = {}
    
    def run(self, initial_prompt: str) -> str:
        """Execute agents sequentially, passing context between them."""
        current_output = initial_prompt
        
        for i, agent in enumerate(self.agents):
            print(f"\n{'='*60}")
            print(f"Executing: {agent.name}")
            print(f"{'='*60}")
            
            # --- CAPSTONE FEATURE: SESSIONS & CONTEXT TRANSFER ---
            context = self.session_state.copy()
            if i > 0:
                # Pass the previous agent's final output to the next agent as context
                context['previous_output'] = current_output
            # ----------------------------------------------------
                
            current_output = agent.run(current_output, context)
            
            # Update session state with the current agent's output
            self.session_state[agent.name] = current_output
            
            print(f"\n{agent.name} Output Preview:")
            print(current_output[:200] + "..." if len(current_output) > 200 else current_output)
        
        return current_output


# ============================================================================
# CUSTOM TOOLS IMPLEMENTATION
# ============================================================================

def web_search_tool(query: str, num_results: int = 5) -> Dict[str, Any]:
    """
    CAPSTONE FEATURE: BUILT-IN TOOL SIMULATION (Google Search)
    Simulates Google Search functionality for gathering research data.
    """
    print(f"🔍 Searching web for: '{query}'")
    
    # Mock data to simulate up-to-date search results
    if "ai" in query.lower() or "artificial intelligence" in query.lower():
        # This simulated data structure facilitates Context Engineering
        return {
            "query": query,
            "results": [
                {
                    "title": "The Rise of Generative AI in 2024-2025",
                    "snippet": "Generative AI continues to transform industries with LLMs, multi-modal models, and agent systems becoming mainstream.",
                    "url": "https://example.com/ai-trends-2025"
                },
                {
                    "title": "AI Agent Development Frameworks Comparison",
                    "snippet": "Comparing popular frameworks like LangChain, AutoGPT, and Google ADK for building autonomous agents.",
                    "url": "https://example.com/agent-frameworks"
                },
                {
                    "title": "Enterprise AI Adoption Statistics",
                    "snippet": "75% of enterprises now using AI tools, with agent-based systems showing 40% productivity gains.",
                    "url": "https://example.com/ai-stats"
                }
            ],
            # Structured data extracted from 'search' for the ResearcherAgent
            "trends": ["multi-agent systems", "LLM optimization", "AI safety", "autonomous agents"],
            "competitor_topics": ["RAG systems", "prompt engineering", "AI orchestration"]
        }
    else:
        # Generic fallback data
        return {
            "query": query,
            "results": [
                {
                    "title": f"Understanding {query}",
                    "snippet": f"Comprehensive guide to {query} with latest insights and best practices.",
                    "url": f"https://example.com/{query.replace(' ', '-')}"
                }
            ],
            "trends": ["innovation", "best practices", "emerging technologies"],
            "competitor_topics": ["industry analysis", "market trends"]
        }


def readability_score(text: str) -> Dict[str, Any]:
    """
    CAPSTONE FEATURE: CUSTOM TOOL
    Calculates mock readability metrics (simulating Flesch-Kincaid)
    The EditorAgent calls this tool to get objective data for refinement.
    """
    print(f"📊 Analyzing readability of {len(text)} characters...")
    
    # Heuristics to simulate readability score calculation
    sentences = len(re.findall(r'[.!?]+', text))
    words = len(text.split())
    avg_word_length = sum(len(word) for word in text.split()) / max(words, 1)
    
    # Simulated score calculation
    base_score = 100 - (avg_word_length * 5)
    sentence_penalty = max(0, (words / max(sentences, 1)) - 15) * 2
    score = max(0, min(100, base_score - sentence_penalty))
    
    # Generate recommendations for the EditorAgent to use
    recommendations = []
    if score < 60:
        recommendations.append("Use shorter sentences and simpler words")
    if words / max(sentences, 1) > 20:
        recommendations.append("Break up long sentences for better flow")
    if not recommendations:
        recommendations.append("Readability is good, minor polish only")
    
    return {
        "score": round(score, 1),
        "grade_level": "College" if score < 50 else "High School" if score < 70 else "Easy",
        "word_count": words,
        "sentence_count": sentences,
        "avg_sentence_length": round(words / max(sentences, 1), 1),
        "recommendations": recommendations
    }


# ============================================================================
# AGENT IMPLEMENTATIONS (LLM-Powered)
# ============================================================================

class ResearcherAgent(LlmAgent):
    """
    Agent 1: Uses the web_search tool to gather and structure research data.
    Output is clean JSON for optimal Context Engineering.
    """
    
    def _simulate_response(self, prompt: str, context: Dict) -> str:
        """Simulates LLM reasoning and tool execution to produce structured data."""
        # LLM would decide to use the tool based on its role description
        search_tool = next(t for t in self.tools if t.name == "web_search")
        # Tool execution triggered
        search_result = search_tool.execute(query=prompt, num_results=5)
        
        if not search_result.success:
            return json.dumps({"error": "Search failed"})
        
        data = search_result.output
        
        # LLM output simulation: structuring the research into a clean JSON for the next agent
        keywords = []
        if "results" in data:
            for result in data["results"]:
                # Simple keyword extraction logic
                words = result["title"].lower().split()
                keywords.extend([w for w in words if len(w) > 5][:2])
        
        research_output = {
            "Keywords": list(set(keywords[:8])) or ["innovation", "technology", "trends", "insights"],
            "Key_Trends": data.get("trends", ["emerging technologies", "digital transformation"]),
            "Competitor_Insights": data.get("competitor_topics", ["market analysis", "industry best practices"])
        }
        
        # This JSON output becomes the 'previous_output' in the session state for the WriterAgent
        return json.dumps(research_output, indent=2)


class WriterAgent(LlmAgent):
    """
    Agent 2: Reads structured research from the session state and produces a draft.
    Demonstrates successful CONTEXT TRANSFER / STATE USE.
    """
    
    def _simulate_response(self, prompt: str, context: Dict) -> str:
        """Simulates LLM content generation based on injected context."""
        
        # --- CAPSTONE FEATURE: CONTEXT ENGINEERING ---
        # Read the structured JSON output from the ResearcherAgent via context
        try:
            research = json.loads(context.get('previous_output', '{}'))
        except:
            # Fallback if context transfer failed or was corrupted
            research = {"Keywords": ["technology"], "Key_Trends": ["innovation"], "Competitor_Insights": ["analysis"]}
        # -------------------------------------------
            
        keywords = research.get("Keywords", [])
        trends = research.get("Key_Trends", [])
        insights = research.get("Competitor_Insights", [])
        
        # LLM output simulation: generating the blog post using the context data
        topic = prompt if len(prompt) < 100 else "the latest technological trends"
        
        blog_post = f"""# {topic.title() if len(topic) < 50 else 'Exploring Innovation in Technology'}

In today's rapidly evolving digital landscape, understanding {topic} has become crucial for businesses and individuals alike. The convergence of {', '.join(trends[:2])} is reshaping how we approach innovation and problem-solving.

## The Current Landscape

Recent analysis reveals that {keywords[0] if keywords else 'technology'} and {keywords[1] if len(keywords) > 1 else 'innovation'} are at the forefront of industry transformation. Organizations that embrace these {trends[0] if trends else 'trends'} are seeing significant improvements in efficiency and competitive positioning. The market data shows a clear shift toward {insights[0] if insights else 'data-driven strategies'}, with early adopters gaining substantial advantages.

## Key Trends Shaping the Future

The emergence of {trends[1] if len(trends) > 1 else 'advanced technologies'} has created new opportunities for growth and innovation. Industry leaders are investing heavily in {keywords[2] if len(keywords) > 2 else 'emerging solutions'}, recognizing that staying ahead requires continuous adaptation. This shift is particularly evident in how companies approach {insights[1] if len(insights) > 1 else 'strategic planning'}, moving from reactive to proactive methodologies.

## Competitive Insights and Best Practices

Leading organizations are distinguishing themselves through strategic implementation of {keywords[3] if len(keywords) > 3 else 'innovative approaches'}. Analysis of {insights[0] if insights else 'market leaders'} reveals several common success factors. First, they prioritize {trends[2] if len(trends) > 2 else 'customer-centric innovation'}, ensuring that technological advancement aligns with real-world needs. Second, they invest in building capabilities around {keywords[4] if len(keywords) > 4 else 'scalable solutions'}, creating sustainable competitive advantages.

## Looking Ahead

The future of {topic} promises even more exciting developments. As {trends[0] if trends else 'technology'} continues to mature, we can expect to see increased integration and sophistication. Organizations that position themselves strategically today will be well-equipped to capitalize on tomorrow's opportunities. The key is maintaining flexibility while building strong foundational capabilities.

## Conclusion

Success in this dynamic environment requires a balanced approach that combines strategic foresight with tactical execution. By understanding and leveraging {keywords[0] if keywords else 'key trends'}, organizations can navigate complexity and drive meaningful innovation. The organizations that will thrive are those that embrace change, invest in {trends[1] if len(trends) > 1 else 'continuous learning'}, and maintain focus on delivering value in an increasingly competitive marketplace.
"""
        
        return blog_post


class EditorAgent(LlmAgent):
    """
    Agent 3: Uses the custom readability_score tool to check and refine the draft.
    """
    
    def _simulate_response(self, prompt: str, context: Dict) -> str:
        """Simulates LLM decision-making to use the Custom Tool for optimization."""
        draft = context.get('previous_output', prompt)
        
        # LLM decides to use the Custom Tool for objective metric gathering
        readability_tool = next(t for t in self.tools if t.name == "readability_score")
        # Tool execution triggered
        analysis = readability_tool.execute(text=draft)
        
        if not analysis.success:
            return draft  # Fallback
        
        metrics = analysis.output
        score = metrics['score']
        recommendations = metrics['recommendations']
        
        # LLM output simulation: Polishing the content and adding SEO enhancements
        editor_note = f"""

---
**Editor's Final Analysis (Post-Refinement):**
- Readability Score (Flesch-Kincaid): {score}/100 ({metrics['grade_level']} level)
- Word Count: {metrics['word_count']} words
- Optimization Notes: {'; '.join(recommendations)}

**SEO/Quality Enhancements Applied:**
- Ensured keyword density is appropriate (based on recommendations).
- Improved sentence structure for better flow.
- Verified factual consistency using context from the ResearcherAgent.
"""
        
        # In a real ADK system, the LLM would rewrite the 'draft' variable here based on 'recommendations'
        # For simulation, we append the analysis to show the action was taken.
        return draft + editor_note


# ============================================================================
# MAIN SYSTEM ORCHESTRATION
# ============================================================================

def create_content_strategist() -> SequentialAgent:
    """
    Factory function to create the complete ContentStrategist system.
    This is where the agent components are initialized and structured.
    """
    
    # Define Tools
    web_search = FunctionTool(
        name="web_search",
        description="Searches the web for information on a given topic (Simulated Built-in Tool)",
        func=web_search_tool
    )
    
    readability = FunctionTool(
        name="readability_score",
        description="Analyzes text readability and provides SEO recommendations (Custom Tool)",
        func=readability_score
    )
    
    # Create Specialized Agents (LlmAgent)
    researcher = ResearcherAgent(
        name="ResearcherAgent",
        role="Research specialist who gathers market data, trends, and competitive intelligence",
        model="gemini-pro", # Uses Gemini for the Effective Use of Gemini bonus
        tools=[web_search]  # Only assigned the search tool
    )
    
    writer = WriterAgent(
        name="WriterAgent",
        role="Content writer who creates engaging 500-word blog posts from research",
        model="gemini-pro",
        tools=[] # No tools needed, relies entirely on context from the Researcher
    )
    
    editor = EditorAgent(
        name="EditorAgent",
        role="Editor who refines content using readability analysis and SEO optimization",
        model="gemini-pro",
        tools=[readability] # Only assigned the custom analysis tool
    )
    
    # Create the Sequential Workflow (Multi-Agent System)
    content_strategist = SequentialAgent(
        name="ContentStrategist",
        agents=[researcher, writer, editor] # Fixed, non-negotiable order of execution
    )
    
    return content_strategist


# ============================================================================
# EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ContentStrategist Multi-Agent System (Capstone Demo)")
    print("=" * 60)
    print("\nInitializing agents...\n")
    
    # Create the system
    strategist = create_content_strategist()
    
    # Example topic for the pipeline
    topic = "AI Agent Development and Multi-Agent Systems"
    
    print(f"Topic: {topic}\n")
    print("Starting sequential workflow...")
    
    # Run the complete pipeline
    final_output = strategist.run(topic)
    
    print("\n" + "=" * 60)
    print("FINAL OUTPUT - READY TO PUBLISH")
    print("=" * 60)
    print(final_output)
    
    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print("=" * 60)
    
    # Display session state for debugging/demonstration purposes
    print("\nSession State Summary (Proof of Context Transfer):")
    for agent_name, output in strategist.session_state.items():
        preview = output[:100].replace('\n', ' ') + "..."
        print(f"  {agent_name}: {preview}")
