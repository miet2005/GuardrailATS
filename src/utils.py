"""
Shared utility functions used across multiple modules in the GuardrailATS
pipeline (guardrail.py, rag_engine.py).
"""

import re


def is_heading_line(line, max_heading_words=4):
    """
    Detects whether a line looks like a section heading/label (e.g.,
    "Requirements:", "Nice to have:") rather than actual content.

    Heuristic: short (at or under max_heading_words words) AND ends with
    a colon. This is a simple, imperfect rule (not 100% accurate on every
    possible document), but works well for typical JD/resume formatting
    and avoids more complex logic that isn't justified for this use case.

    Args:
        line (str): a single, already-stripped line of text.
        max_heading_words (int): maximum word count to still count as a heading.

    Returns:
        bool: True if this line looks like a heading.
    """
    if not line.endswith(":"):
        return False

    word_count = len(line.split())
    return word_count <= max_heading_words


def chunk_text_for_analysis(text, max_chunk_chars=300):
    """
    Splits text into small, independently-analyzable chunks, matching how
    resumes and job descriptions are naturally structured (line/bullet based).

    Strategy:
      1. Split on line breaks (resumes and JDs are naturally line/bullet
         based).
      2. Detect short, colon-ending heading lines (e.g., "Requirements:",
         "Nice to have:") and merge them into the NEXT line, instead of
         leaving them as standalone chunks. Standalone headings produce
         noisy, near-meaningless similarity/injection scores when compared
         against unrelated content elsewhere - merging preserves their
         useful context (e.g., "required" vs "optional") without leaving
         noise behind.
      3. Any resulting chunk still longer than max_chunk_chars gets further
         split into sentences.

    Used by:
      - guardrail.py (Tier 2 injection detection)
      - rag_engine.py (embedding-based resume/JD matching)

    Args:
        text (str): the full text to chunk.
        max_chunk_chars (int): if a line exceeds this length, split it
            further into sentences.

    Returns:
        list[str]: non-empty text chunks, stripped of leading/trailing whitespace.
    """
    raw_lines = [line.strip() for line in text.split("\n") if line.strip()]

    # --- Pass 1: merge heading lines into the line that follows them ---
    merged_lines = []
    pending_heading = None

    for line in raw_lines:
        if is_heading_line(line):
            # Hold onto this heading; don't emit it as its own chunk yet.
            # If a heading is immediately followed by another heading,
            # we simply overwrite pending_heading - only the most recent
            # heading gets attached, which is the reasonable behavior.
            pending_heading = line
            continue

        if pending_heading:
            merged_lines.append(f"{pending_heading} {line}")
            pending_heading = None
        else:
            merged_lines.append(line)

    # Edge case: if the very last line in the document was itself a heading
    # (nothing followed it to merge into), keep it as its own chunk rather
    # than silently dropping it.
    if pending_heading:
        merged_lines.append(pending_heading)

    # --- Pass 2: split any overly long merged line into sentences ---
    chunks = []
    for line in merged_lines:
        if len(line) <= max_chunk_chars:
            chunks.append(line)
        else:
            sentences = re.split(r'(?<=[.!?])\s+', line)
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    chunks.append(sentence)

    return chunks