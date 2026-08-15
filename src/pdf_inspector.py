"""
PDF Inspector Module
Extracts text and structural metadata (font size, color, position) from PDF resumes
to enable detection of hidden/invisible content used in prompt injection attacks.
"""

import re

import fitz  # PyMuPDF


# Common phrasings used in prompt injection attempts. Matched case-insensitively
# against span text as a fast, deterministic first line of defense - catches
# obvious/lazy attacks without needing any ML inference. This list is not
# exhaustive and is meant as defense-in-depth alongside Tier 2's ML-based
# semantic detection, which can catch paraphrased or novel phrasing this
# list would miss. Kept intentionally small and high-precision rather than
# broad, to avoid false-positiving on normal resume language.
INJECTION_KEYWORD_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"system override",
    r"disregard (all )?(previous|prior|above)",
    r"you are now",
    r"new instructions?:",
    r"forget (everything|what) (you were told|above)",
    r"act as (if|though)",
    r"do not follow (your|the) (original|previous) instructions",
]


def extract_text_spans(pdf_path):
    """
    Opens a PDF and extracts every text span along with its formatting metadata.

    A 'span' is the smallest unit of text in a PDF that shares consistent
    formatting (same font, size, color). This granular extraction lets us
    later detect anomalies like tiny fonts or invisible text colors that
    would be invisible to a human but readable by a naive text extractor.

    Args:
        pdf_path (str): Path to the PDF file to inspect.

    Returns:
        list[dict]: One dict per text span, containing:
            - text (str): the actual text content
            - font_size (float): font size in points
            - color (int): color as a packed RGB integer
            - page_number (int): which page (0-indexed)
            - bbox (tuple): (x0, y0, x1, y1) bounding box coordinates
            - page_width (float): width of the page in points
            - page_height (float): height of the page in points
    """
    doc = fitz.open(pdf_path)
    spans = []

    for page_number, page in enumerate(doc):
        page_width = page.rect.width
        page_height = page.rect.height

        # get_text("dict") returns the full structural breakdown:
        # blocks -> lines -> spans. By default, PyMuPDF clips extraction to
        # the page's own rectangle, silently excluding any text positioned
        # outside the visible page boundary - which means an attacker hiding
        # a payload off-page would go completely undetected, since we'd
        # never even see the text to check it. We explicitly expand the
        # extraction area well beyond the page bounds so off-page content
        # is captured and can still be flagged by our off-page positioning
        # check (Check 3 in detect_anomalies).
        expanded_clip = fitz.Rect(
            -2000, -2000, page_width + 2000, page_height + 2000
        )
        page_content = page.get_text("dict", clip=expanded_clip)

        for block in page_content.get("blocks", []):
            # Skip image blocks (type 1) — we only care about text blocks (type 0)
            if block.get("type") != 0:
                continue

            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    span_text = span.get("text", "")

                    # Skip completely empty spans (just whitespace artifacts)
                    if not span_text.strip():
                        continue

                    spans.append({
                        "text": span_text,
                        "font_size": span.get("size", 0.0),
                        "color": span.get("color", 0),
                        "page_number": page_number,
                        "bbox": span.get("bbox", (0, 0, 0, 0)),
                        "page_width": page_width,
                        "page_height": page_height,
                    })

    doc.close()
    return spans



def unpack_rgb_color(color_int):
    """
    Converts PyMuPDF's packed integer color format into separate R, G, B values.

    PyMuPDF stores color as a single integer that encodes red, green, and blue
    channels together (a common technique to save space). This function unpacks
    it back into the three separate 0-255 values so we can compare against
    known colors like white (255, 255, 255).

    Args:
        color_int (int): packed RGB color as returned by PyMuPDF.

    Returns:
        tuple[int, int, int]: (red, green, blue), each 0-255.
    """
    red = (color_int >> 16) & 255
    green = (color_int >> 8) & 255
    blue = color_int & 255
    return (red, green, blue)


def is_color_near_white(rgb, tolerance=10):
    """
    Checks whether an RGB color is close enough to pure white to be considered
    invisible against a typical white resume background.

    Args:
        rgb (tuple[int, int, int]): the (red, green, blue) color to check.
        tolerance (int): how far from 255 each channel is allowed to be and
            still count as "near white". Default 10 catches near-invisible
            off-white tricks, not just pure (255,255,255).

    Returns:
        bool: True if the color is within tolerance of pure white.
    """
    red, green, blue = rgb
    return (
        red >= (255 - tolerance)
        and green >= (255 - tolerance)
        and blue >= (255 - tolerance)
    )


def detect_anomalies(spans, font_size_threshold=5.0, color_tolerance=10):
    """
    Scans extracted text spans for signs of hidden/invisible content commonly
    used to smuggle prompt injection payloads into resumes.

    Three anomaly types are checked per span:
      1. Font size below `font_size_threshold` (near-invisible text size)
      2. Color near-white, i.e. blending into a typical white page background
      3. Bounding box positioned outside the visible page area

    Args:
        spans (list[dict]): output from extract_text_spans().
        font_size_threshold (float): font sizes below this (in points) are flagged.
        color_tolerance (int): how close to pure white counts as "invisible".

    Returns:
        list[dict]: one dict per flagged span, containing all original span
            data plus a "reasons" list explaining why it was flagged.
    """
    flagged_spans = []

    for span in spans:
        reasons = []

        # Check 1: tiny font size
        if span["font_size"] < font_size_threshold:
            reasons.append(
                f"Font size {span['font_size']:.1f}pt is below the "
                f"{font_size_threshold}pt visibility threshold"
            )

        # Check 2: color blending into white background
        rgb = unpack_rgb_color(span["color"])
        if is_color_near_white(rgb, tolerance=color_tolerance):
            reasons.append(
                f"Text color RGB{rgb} is near-white and likely invisible "
                f"against a white page background"
            )

        # Check 3: positioned outside the visible page
        x0, y0, x1, y1 = span["bbox"]
        page_w = span["page_width"]
        page_h = span["page_height"]
        if x0 < 0 or y0 < 0 or x1 > page_w or y1 > page_h:
            reasons.append(
                f"Text positioned at bbox({x0:.1f}, {y0:.1f}, {x1:.1f}, {y1:.1f}) "
                f"falls outside the visible page area "
                f"({page_w:.1f} x {page_h:.1f})"
            )

        # Check 4: known injection keyword/phrase patterns (case-insensitive).
        # This check applies regardless of formatting - a span can be flagged
        # here even with completely normal font/color/position, since a lazy
        # attacker might not bother hiding text structurally at all. Fast,
        # deterministic, zero ML cost - a useful first line of defense
        # alongside Tier 2's more sophisticated semantic detection.
        for pattern in INJECTION_KEYWORD_PATTERNS:
            if re.search(pattern, span["text"], re.IGNORECASE):
                reasons.append(
                    f"Text matches known injection pattern: \"{pattern}\""
                )
                break  # one keyword match is enough evidence; avoid duplicate reasons

        # If any check triggered, record this span as flagged
        if reasons:
            flagged_span = dict(span)  # copy original data
            flagged_span["reasons"] = reasons
            flagged_spans.append(flagged_span)

    return flagged_spans



if __name__ == "__main__":
    # Quick manual test — we'll point this at a real PDF in the next step
    print("pdf_inspector.py loaded successfully. Run via test script once a sample PDF exists.")
