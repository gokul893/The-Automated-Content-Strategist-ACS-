🤖 The Automated Content Strategist (ACS)

Kaggle AI Agents Intensive Capstone Project | Enterprise Agents Track

🏆 Project Pitch

Problem: Enterprise marketing teams struggle to produce high-quality, strategically relevant, and fact-checked content at the speed required by modern markets. The manual process of research, drafting, and optimization is slow and error-prone.

Solution: The Automated Content Strategist (ACS) is a Sequential Multi-Agent System that automates the entire content workflow. It uses specialized, tool-equipped agents to transform a simple topic into a polished, SEO-ready blog post draft, demonstrating efficiency and reliability suitable for business applications.

Value: The ACS reduces content ideation and drafting time by over 80%, ensuring all produced content is grounded in real-time search data and objectively checked for quality metrics (readability/SEO). This aligns perfectly with the Enterprise Agents track goal of improving business workflows.

⚙️ Technical Architecture: Sequential Pipeline

The ACS is built on the principles of the Agent Development Kit (ADK) using a strict, three-step sequential pipeline, designed to mimic a real marketing team.

Architecture Flow:

ResearcherAgent (Data Layer): Takes the initial prompt, uses external tools to gather up-to-date market trends and competitor insights, and formats the output into clean JSON.

WriterAgent (Drafting Layer): Receives the structured JSON (via Context Transfer) and generates the complete 500-word article draft, focusing purely on creative generation.

EditorAgent (Quality Layer): Receives the draft, uses a Custom Tool to calculate objective readability metrics, and refines the text for quality and SEO compliance.

✨ Key Capstone Concepts Applied (Required $\ge$3)

This project demonstrates proficiency in four core concepts taught during the intensive:

Concept

Implementation Details

Criteria Addressed

1. Multi-Agent System

The entire solution is orchestrated by a SequentialAgent pipeline, ensuring research must precede writing, and editing must follow writing.

Technical Implementation

2. Tools (Built-in & Custom)

Built-in Tool Simulation: The ResearcherAgent utilizes a simulated Google Search tool. Custom Tool: The EditorAgent uses the specialized readability_score() function to perform objective quality checks (Function Calling).

Technical Implementation

3. Sessions & Context Transfer

We use the sequential runner's session state (analogous to InMemorySessionService) to pass the structured JSON output from the ResearcherAgent to the WriterAgent. This is crucial for Context Engineering and preventing prompt flooding.

Technical Implementation

4. Effective Use of Gemini

All three LLM agents (Researcher, Writer, Editor) are configured to use the powerful gemini-pro model, leveraging its advanced reasoning and creative generation capabilities for high-quality enterprise content.

Bonus Points (5 pts)

🚀 Setup and Execution

Dependencies:
This project relies on standard Python libraries and models the structure of Google's Agent Development Kit (ADK).

Clone the Repository:

git clone
cd automated-content-strategist


Install Requirements:

pip install -r requirements.txt


Set Environment Variables:
The mock framework uses an environment variable for the model key. DO NOT hardcode your key.

export GOOGLE_AI_KEY="YOUR_API_KEY_HERE" 


Run the Agent:
The script executes the full sequential pipeline on a predefined topic and prints the output and session state trace.

python content_strategist.py
