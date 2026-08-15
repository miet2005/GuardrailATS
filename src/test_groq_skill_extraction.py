"""
End-to-end test script for the Groq-backed skill extraction path.
Same test as test_skill_extraction.py, but confirms the LLM_BACKEND=groq
path works correctly before switching the full dashboard over.
"""

import os
from dotenv import load_dotenv

from pdf_inspector import extract_text_spans
from rag_engine import generate_skill_analysis, LLM_BACKEND


def load_resume_text(pdf_path):
    """Extracts all visible text from a resume PDF as one string."""
    spans = extract_text_spans(pdf_path)
    return "\n".join(span["text"] for span in spans)


def load_jd_text(jd_path):
    """Reads a job description from a plain text file."""
    with open(jd_path, "r", encoding="utf-8") as f:
        return f.read()


def print_skill_analysis(resume_path, jd_path):
    """Runs the LLM skill extraction pipeline and prints a readable report."""
    resume_text = load_resume_text(resume_path)
    jd_text = load_jd_text(jd_path)

    print(f"\n{'=' * 65}")
    print(f"SKILL ANALYSIS: {resume_path} vs {jd_path}")
    print(f"Active LLM_BACKEND: {LLM_BACKEND}")
    print(f"{'=' * 65}")
    print("Calling LLM...")

    result = generate_skill_analysis(resume_text, jd_text)

    if not result["parse_success"]:
        print("\nWARNING: Failed to parse model output as JSON.")
        if "error" in result:
            print(f"Error: {result['error']}")
        print(f"\nRaw model output was:\n{result['raw_model_output']}")
        return

    print(f"\nMATCHED SKILLS ({len(result['matched_skills'])}):")
    for skill in result["matched_skills"]:
        print(f"  + {skill}")

    print(f"\nMISSING SKILLS ({len(result['missing_skills'])}):")
    for skill in result["missing_skills"]:
        print(f"  - {skill}")

    print(f"\nSUMMARY:\n  {result['summary']}")


if __name__ == "__main__":
    print_skill_analysis("../data/clean_resume.pdf", "../data/sample_jd.txt")