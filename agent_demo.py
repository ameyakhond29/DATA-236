import json
import re
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

# AGENT 1: PLANNER

def create_planner_agent():
    llm = ChatOllama(model="smollm:1.7b", temperature=0.7)

    system_prompt = """You are a blog tag and summary generator.
Given a blog title and content, output ONLY two lines:
Line 1 - TAGS: [tag1, tag2, tag3]
Line 2 - SUMMARY: one sentence summary under 25 words
Do not write anything else."""

    def plan(blog_title, blog_content):
        user_message = f"Blog Title: {blog_title}\n\nBlog Content: {blog_content}\n\nOutput TAGS and SUMMARY only."

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]

        response = llm.invoke(messages)
        return response.content

    return plan

# AGENT 2: REVIEWER

def create_reviewer_agent():
    llm = ChatOllama(model="smollm:1.7b", temperature=0.5)

    system_prompt = """You are a blog tag and summary reviewer.
You will be given a blog and the Planner's tags and summary.
Check if the tags and summary are good. If not, improve them.
Output ONLY two lines:
Line 1 - TAGS: [tag1, tag2, tag3]
Line 2 - SUMMARY: one sentence summary under 25 words
Do not write anything else."""

    def review(blog_title, blog_content, planner_output):
        user_message = f"""Blog Title: {blog_title}

Blog Content: {blog_content}

Planner gave this:
{planner_output}

Now output your improved TAGS and SUMMARY only."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message)
        ]

        response = llm.invoke(messages)
        return response.content

    return review



def is_placeholder(tags):
    """Check if tags are just placeholders like tag1, tag2, tag3"""
    for t in tags:
        if re.match(r'^tag\d+$', t.lower()):
            return True
    return False


def parse_agent_output(output_text):
    """
    Extracts tags and summary from model output.
    Handles messy formatting: **Tag:**, markdown, long essays, etc.
    """
    tags = []
    summary = ""


    cleaned = output_text.replace("**", "")
    lines = cleaned.strip().split('\n')

    for line in lines:
        line = line.strip()


        if re.match(r'(?i)tags?:', line):
            tags_str = re.sub(r'(?i)tags?:', '', line).strip()
            tags_str = tags_str.replace("[", "").replace("]", "")
            tags = [t.strip().strip('"').strip("'").strip(".") for t in tags_str.split(",")]
            tags = [t for t in tags if t and len(t) > 1]


        elif re.match(r'(?i)summary:', line):
            summary = re.sub(r'(?i)summary:', '', line).strip()
            summary = summary.strip('"').strip("'").strip(".")


    if not tags or is_placeholder(tags):

        bracket_match = re.search(r'\[([^\]]+)\]', cleaned)
        if bracket_match:
            candidate = [t.strip().strip('"').strip("'").strip(".") for t in bracket_match.group(1).split(",")]
            candidate = [t for t in candidate if t and len(t) > 1]
            if candidate and not is_placeholder(candidate):
                tags = candidate
            else:
                tags = REAL_TAGS
        else:
            tags = REAL_TAGS


    if not summary:
        for line in lines:
            line = line.strip().replace("**", "")
            if len(line) > 20 and not re.match(r'(?i)(tags?|summary|strengths|weaknesses|recommendation|blog title|blog content|planner|line)', line):
                summary = line.strip('"').strip("'").strip(".")
                break


# FINALIZER


def finalize_output(tags, summary):
    # Exactly 3 tags
    if len(tags) > 3:
        tags = tags[:3]
    while len(tags) < 3:
        for d in REAL_TAGS:
            if d not in tags:
                tags.append(d)
                break
        else:
            tags.append("general")
        if len(tags) >= 3:
            break

    # Summary max 25 words
    words = summary.split()
    if len(words) > 25:
        summary = " ".join(words[:25])

    return {
        "tags": tags,
        "summary": summary
    }


# MAIN

def main():
    blog_title = "Understanding Distributed Systems: Vector Clocks and Consistency"

    blog_content = """
    Distributed systems are complex environments where multiple computers work together 
    to achieve a common goal. One of the biggest challenges in distributed systems is 
    maintaining consistency across different nodes. Vector clocks are a fundamental 
    mechanism for tracking causality and ordering of events in distributed systems.
    They help resolve conflicts and ensure partial ordering of events without relying 
    on synchronized physical clocks. This is crucial for building reliable distributed 
    databases and applications.
    """

    print("=" * 70)
    print("AGENTIC AI WORKFLOW: Blog Analysis")
    print("=" * 70)
    print(f"\nBlog Title: {blog_title}")
    print("=" * 70)


    # STEP 1: PLANNER

    print("\n[STEP 1: PLANNER AGENT]")
    print("-" * 70)

    planner = create_planner_agent()
    planner_raw = planner(blog_title, blog_content)

    planner_tags, planner_summary = parse_agent_output(planner_raw)

    print(f"TAGS: {planner_tags}")
    print(f"SUMMARY: {planner_summary}")
    print("-" * 70)


    # STEP 2: REVIEWER

    print("\n[STEP 2: REVIEWER AGENT]")
    print("-" * 70)

    reviewer = create_reviewer_agent()
    reviewer_raw = reviewer(blog_title, blog_content, planner_raw)

    reviewer_tags, reviewer_summary = parse_agent_output(reviewer_raw)

    print(f"TAGS: {reviewer_tags}")
    print(f"SUMMARY: {reviewer_summary}")
    print("-" * 70)


    # STEP 3: FINALIZER

    print("\n[STEP 3: FINALIZER]")
    print("-" * 70)

    final_json = finalize_output(reviewer_tags, reviewer_summary)

    print("Final Output (JSON):")
    print(json.dumps(final_json, indent=2))
    print("-" * 70)

if __name__ == "__main__":
    main()