import os
import re
import html
from openai import OpenAI
from dotenv import load_dotenv

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether
)

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# ---------------------------------------------------------
# 0. UNICODE FONT REGISTRATION
# ---------------------------------------------------------
# FIX: the script never registered a real font, so every style used
# Helvetica. Helvetica's built-in encoding has no glyph for the Indian
# Rupee sign (\u20b9) or several other characters the LLM commonly
# outputs, so ReportLab silently replaced them with "\u25a0" boxes in
# the final PDF. We register a Unicode-capable TTF (DejaVu Sans, which
# does contain the Rupee sign) and use it everywhere instead. If it
# isn't installed on a given machine we fall back to Helvetica rather
# than crashing.

FONT_REGULAR = "Helvetica"
FONT_BOLD = "Helvetica-Bold"

_UNICODE_FONT_CANDIDATES = [
    (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ),
    (
        "C:\\Windows\\Fonts\\arialuni.ttf",
        "C:\\Windows\\Fonts\\arialbd.ttf",
    ),
    (
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ),
]


def register_fonts():
    """Register a Unicode-capable font so currency symbols like Rupee
    render correctly. Sets the module-level FONT_REGULAR/FONT_BOLD
    names used by build_styles() and add_page_number(). Safe to call
    more than once; falls back to Helvetica if no candidate is found.
    """

    global FONT_REGULAR, FONT_BOLD

    for regular_path, bold_path in _UNICODE_FONT_CANDIDATES:

        if os.path.exists(regular_path):

            try:
                pdfmetrics.registerFont(TTFont("ReportUnicode", regular_path))

                if os.path.exists(bold_path):
                    pdfmetrics.registerFont(TTFont("ReportUnicode-Bold", bold_path))
                    FONT_BOLD = "ReportUnicode-Bold"
                else:
                    FONT_BOLD = "ReportUnicode"

                FONT_REGULAR = "ReportUnicode"
                return

            except Exception:
                continue

    # No Unicode font found on this machine - keep the Helvetica
    # defaults rather than failing the whole report.


register_fonts()


# ---------------------------------------------------------
# 1. COLLECT EVIDENCE
# ---------------------------------------------------------

def collect_evidence(retriever):

    questions = [
        "What are the company's major financial achievements and financial performance?",
        "What are the company's major business segments and how are they performing?",
        "What are the company's major risks and risk mitigation measures?",
        "What are the company's major growth opportunities and future strategies?",
        "Are there any financial warning signs, red flags or areas requiring investigation?",
        "Are there any contradictions or inconsistencies in the reported information?",
        "What important information is missing, unclear or requires further investigation?"
    ]

    evidence = []

    for question in questions:

        docs = retriever.invoke(question)

        evidence.append(
            f"\nQUESTION: {question}\n"
        )

        for doc in docs:

            page = doc.metadata.get("page", "Unknown")
            source = doc.metadata.get("source", "Document")

            # LangChain PDF loaders often use zero-based page numbers.
            # Convert to human-readable page numbering when numeric.
            if isinstance(page, int):
                page = page + 1

            evidence.append(
                f"""
SOURCE: {source}
PAGE: {page}

{doc.page_content}
"""
            )

    return "\n".join(evidence)


# ---------------------------------------------------------
# 2. GENERATE REPORT CONTENT
# ---------------------------------------------------------

def generate_report(evidence):

    prompt = f"""
You are a senior due diligence analyst writing a formal report for a
reviewer who will act on it. Your job is accuracy, traceability and
clarity, not persuasion.

Base the report ONLY on the EVIDENCE at the end of this message.

=====================================================================
PART 1 - ANALYSIS RULES
=====================================================================

Before writing, silently work out: the company name, the reporting
periods covered, the currency and unit used (for example "INR crore"),
and whether each figure is standalone or consolidated. Use these
consistently throughout.

Grounding
- Never invent facts, figures, dates, page numbers or sources.
- Never fill in a missing value. If something is not in the evidence,
  write exactly: Not clearly stated in the provided documents.
- Do not draw conclusions the evidence does not support.
- Do not give investment advice, buy/sell/hold views or valuation views.
- Keep three things clearly separate: (a) reported facts, (b) potential
  concerns inferred from the facts, (c) items needing further
  investigation. Never present (b) or (c) as confirmed problems.

Numbers
- Copy reported figures exactly, with the currency symbol, unit and
  period label (for example "₹78,089 crore", "FY2026").
- Write every percentage with one decimal place (for example 41.1%).
- Any figure you derive yourself (growth %, margins, shares of total)
  must be computed from reported figures, double-checked, and marked
  "(Calculated)". Never mark a reported figure as calculated.
- Standalone and consolidated figures are different reporting scopes.
  Never compare them as if they were the same thing, and do not call
  a difference between them a contradiction unless the evidence shows
  a real conflict within the same scope and period.
- Compare like with like: same metric definition, same scope, same
  period. If definitions differ (for example "revenue from operations"
  vs "revenue including other income"), say so.

Citations
- Cite the page for every material fact, inline in plain parentheses,
  for example (p. 331) or (pp. 431-432). In tables use the Source
  column. Only cite pages that appear in the evidence.
- Never put a citation on a line by itself and never bold it.

Explicit vs inferred
- Risks the company itself states are "Reported by company".
- Risks you infer from the numbers are "Inferred". Label each one and
  never blur the two.

=====================================================================
PART 2 - OUTPUT FORMAT CONTRACT (strict, the output is machine parsed)
=====================================================================

The output is converted to a PDF by a program, so formatting mistakes
break the document. Follow these rules exactly.

Headings
- Use only these levels: "# " once for the title, "## " for the twelve
  numbered sections, "### " for sub-sections. Never use "####" or deeper.
- A heading is always on its own line, followed by a blank line.
- Never use a bold line as a substitute for a heading.
- Never use horizontal rules ("---", "***" or "___").

Tables
- Put a blank line before and after every table.
- Every row, including the header row and the separator row, is on
  its own single line that starts with "|" and ends with "|".
- Never put two rows on one line. Never split a row across lines.
- Never put a line break, "<br>", or a bullet list inside a cell.
- Every row must have exactly the same number of cells as the header.
- Keep each cell under about 25 words. If a point needs more, split it
  into another row.
- Use at most about 12 rows per table. If there is more, keep the most
  material rows.
- Use exactly the column headers given below, in the same order.
- The separator row uses "---" for text columns and "---:" for numeric
  columns, with the same number of cells as the header.
- If a required table has no supporting evidence at all, still output
  the table with a single row whose first cell is
  Not clearly stated in the provided documents. and "-" in the rest.

Text
- One bullet per line, starting with "- ". Write each bullet as a
  single line with no manual line wrapping. No nested bullets.
- Numbered items are single lines starting with "1. ", "2. ", etc.
- Paragraphs are 1-3 sentences, written as a single line each.
- Use **bold** sparingly and only inside a sentence, never for a whole
  line.
- No emojis, no HTML, no code fences, no block quotes.
- Do not add anything before the title or after the last section.

=====================================================================
PART 3 - REPORT TEMPLATE
=====================================================================

Start with this exact structure. Replace text in angle brackets.

# Due Diligence Report: <Company Name>

One or two sentences stating the documents and periods covered, the
currency unit, and that the report relies only on the evidence supplied.

## 1. Executive Summary

Write 5-7 bullets. Each bullet states one key finding with its
figures, period and page. Cover: overall performance, main growth
driver, main concern, and the most important information gap.

| Key Area | Finding | Evidence / Source |
|---|---|---|

Give 5-8 rows covering performance, profitability, segments, risks,
red flags and data gaps.

## 2. Company Overview

Two or three short paragraphs or bullets: what the business does, its
main segments, reporting scope (standalone / consolidated), and
notable organisational facts stated in the evidence.

| Area | Reported Information | Source |
|---|---|---|

## 3. Financial Analysis

Use a "### 3.1 Consolidated financial performance" and a
"### 3.2 Standalone financial performance" sub-section when both are
available; otherwise use a single sub-section for the scope available.
Each sub-section has one table:

| Metric | Current Period | Previous Period | Change/Growth | Observation | Source Page |
|---|---:|---:|---:|---|---|

Include every available metric among: revenue, operating profit,
profit before tax, profit after tax, finance costs, exceptional items,
margins, assets, liabilities, debt, cash, cash flow, EPS. Do not
include a metric that has no figure in the evidence. Mark derived
growth and margin values "(Calculated)".

After the tables, add "### 3.3 Key financial observations" with 3-6
bullets interpreting the numbers (trends, margin movement, finance
cost movement, exceptional items, what cannot be assessed and why).

## 4. Business Segment Analysis

Use one table per reporting scope available, in sub-sections
"### 4.1 <scope> segment performance":

| Segment | Current Revenue | Previous Revenue | Revenue Growth | Current Profit | Previous Profit | Key Observation | Source Page |
|---|---:|---:|---:|---:|---:|---|---|

Then, for each material segment, add a "### 4.x <Segment name>"
sub-section with 2-4 bullets: what changed, how profit moved relative
to revenue, share of total (Calculated), and what the evidence does
not explain.

## 5. Key Business Insights

### 5.1 Reported facts

Bullets containing only facts directly stated in the evidence.

### 5.2 Potential concerns

Bullets of evidence-based concerns. Each must say what in the
evidence prompted it and must not be worded as a confirmed problem.

### 5.3 Areas requiring investigation

Bullets of issues that need more information.

| Insight | Type | Evidence | Source Page |
|---|---|---|---|

Type must be exactly one of: Reported fact, Potential concern,
Requires investigation. Include the most material 6-10 insights.

## 6. Growth Opportunities

| Opportunity | Supporting Evidence | Potential Business Impact | Limitations / Unknowns | Source Page |
|---|---|---|---|---|

Only opportunities supported by the evidence. Do not invent targets,
forecasts or expected returns. If a strategy or target is not stated,
say so in the Limitations column.

## 7. Major Risks

| Risk | Basis | Evidence | Potential Impact | Reported Mitigation | Further Investigation | Source Page |
|---|---|---|---|---|---|---|

Basis must be exactly "Reported by company" or "Inferred". If no
mitigation is stated, write Not clearly stated in the provided
documents. in that cell.

## 8. Warnings and Potential Red Flags

| Warning / Potential Red Flag | Supporting Evidence | Why It Requires Attention | Confirmed or Requires Investigation | Source Page |
|---|---|---|---|---|

Use "Confirmed" only when the evidence itself establishes the point.
Otherwise use "Requires investigation". After the table, add one
sentence stating whether the evidence shows any confirmed misstatement,
qualification, going-concern issue or enforcement action.

## 9. Contradictions or Inconsistencies

| Topic | Information A | Information B | Source Pages | Assessment |
|---|---|---|---|---|

Assessment must be one of: Confirmed contradiction, Reporting-scope
difference, Presentation or definition difference, Requires
reconciliation. If there is no confirmed contradiction, add one
sentence saying so explicitly.

## 10. Missing or Unclear Information

| Missing / Unclear Information | Why It Matters | Available Evidence | Suggested Investigation |
|---|---|---|---|

Cover financial, operational, risk and governance gaps. Prioritise the
most material gaps first (for example debt, cash flow, liquidity,
nature of exceptional items, reconciliation of reporting scopes).

## 11. Questions for Further Investigation

| # | Investigation Question | Reason | Relevant Source/Page |
|---:|---|---|---|

Give 10-20 questions, numbered 1, 2, 3 in the first column, grouped in
order of importance. Each question must trace to a specific gap or
concern found above. Do not also repeat them as a separate list.

## 12. Overall Due Diligence Observations

Use these four sub-sections, each with 2-5 bullets:

### 12.1 Evidence-based observations
### 12.2 Strengths supported by the documents
### 12.3 Areas requiring further investigation
### 12.4 Limitations of the available evidence

End with one sentence: No investment recommendation is made.

=====================================================================
PART 4 - FINAL SELF-CHECK (do silently, do not print)
=====================================================================

Before you answer, verify:
1. Every table row is on its own line and has the same number of cells
   as its header.
2. There are no horizontal rules, no "####" headings, no bold-only
   lines, no citations on their own line.
3. Every derived number is marked "(Calculated)" and is arithmetically
   correct.
4. No figure, page number or fact appears that is not in the evidence.
5. Standalone and consolidated figures are never mixed in one row.
6. The response contains only the report.

EVIDENCE:
{evidence}
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    return response.output_text


# ---------------------------------------------------------
# 3. PDF STYLES
# ---------------------------------------------------------

def build_styles():

    styles = getSampleStyleSheet()

    return {

        "title": ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontName=FONT_BOLD,
            fontSize=22,
            leading=27,
            alignment=TA_CENTER,
            spaceAfter=8
        ),

        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=styles["Normal"],
            fontName=FONT_REGULAR,
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"),
            spaceAfter=20
        ),

        "h1": ParagraphStyle(
            "H1",
            parent=styles["Heading1"],
            fontName=FONT_BOLD,
            fontSize=15,
            leading=19,
            spaceBefore=15,
            spaceAfter=9,
            textColor=colors.HexColor("#17365D"),
            keepWithNext=True
        ),

        "h2": ParagraphStyle(
            "H2",
            parent=styles["Heading2"],
            fontName=FONT_BOLD,
            fontSize=11.5,
            leading=15,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor("#2F5597"),
            keepWithNext=True
        ),

        "body": ParagraphStyle(
            "Body",
            parent=styles["BodyText"],
            fontName=FONT_REGULAR,
            fontSize=9.2,
            leading=13,
            spaceAfter=6
        ),

        "bullet": ParagraphStyle(
            "Bullet",
            parent=styles["BodyText"],
            fontName=FONT_REGULAR,
            fontSize=9.2,
            leading=13,
            leftIndent=13,
            firstLineIndent=-7,
            spaceAfter=4
        ),

        "table_header": ParagraphStyle(
            "TableHeader",
            parent=styles["BodyText"],
            fontName=FONT_BOLD,
            fontSize=7.2,
            leading=9,
            textColor=colors.white
        ),

        "table_cell": ParagraphStyle(
            "TableCell",
            parent=styles["BodyText"],
            fontName=FONT_REGULAR,
            fontSize=7.1,
            leading=9,
            spaceAfter=0
        ),

        "small": ParagraphStyle(
            "Small",
            parent=styles["BodyText"],
            fontName=FONT_REGULAR,
            fontSize=7.5,
            leading=10,
            textColor=colors.HexColor("#666666")
        )
    }


# ---------------------------------------------------------
# 4. MARKDOWN HELPERS
# ---------------------------------------------------------

def clean_markdown(text):

    text = text.strip()

    # Remove markdown bold/italic markers while preserving text.
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.*?)__", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*(?!\s)(.*?)(?<!\s)\*", r"<i>\1</i>", text)

    # Escape ampersands that are not already entities.
    text = re.sub(
        r"&(?!amp;|lt;|gt;|quot;|apos;)",
        "&amp;",
        text
    )

    text = text.replace("<br>", "<br/>")

    return text


def paragraph_text(text, style):

    text = clean_markdown(text)

    return Paragraph(text, style)


def is_table_separator(line):

    cells = [c.strip() for c in line.strip().strip("|").split("|")]

    if not cells:
        return False

    # FIX: was "-{3,}" (required 3+ dashes). Some models emit shorter
    # separators such as "|-|-|"; loosened to 1+ so real separator
    # rows aren't missed and the table that follows isn't dropped.
    return all(
        re.fullmatch(r":?-+:?", c.replace(" ", ""))
        for c in cells
    )


def parse_table(lines, start_index):

    header_line = lines[start_index].strip()
    separator_line = lines[start_index + 1].strip()

    headers = [
        c.strip()
        for c in header_line.strip("|").split("|")
    ]

    column_count = len(headers)

    rows = []

    i = start_index + 2

    while i < len(lines):

        line = lines[i].strip()

        if not line or "|" not in line:
            break

        # FIX: the LLM sometimes runs two or more data rows together on
        # one line with no newline between them, joined as
        # "...| 553 crore || Eyecare | 907 crore ...". That used to
        # leave us with more cells than headers on that line, and the
        # code silently truncated to the first `column_count` cells -
        # discarding every row after the first (this is why segments
        # like "Others" were vanishing from the segment table).
        #
        # A squashed boundary always shows up as a literal "||" (the
        # closing pipe of one row butted against the opening pipe of
        # the next, with nothing in between), whereas a genuine empty
        # cell is written as "| |" with a space. So we un-squash by
        # inserting a line break at every "||" before splitting into
        # cells, recovering each row instead of merging/dropping it.
        for sub_line in line.replace("||", "|\n|").split("\n"):

            sub_line = sub_line.strip()

            if not sub_line or "|" not in sub_line:
                continue

            cells = [
                c.strip()
                for c in sub_line.strip("|").split("|")
            ]

            if len(cells) < column_count:
                cells += [""] * (column_count - len(cells))
            elif len(cells) > column_count:
                cells = cells[:column_count]

            rows.append(cells)

        i += 1

    return headers, rows, i


def make_table(headers, rows, styles):

    data = []

    data.append([
        Paragraph(clean_markdown(h), styles["table_header"])
        for h in headers
    ])

    for row in rows:
        data.append([
            Paragraph(clean_markdown(cell), styles["table_cell"])
            for cell in row
        ])

    column_count = len(headers)

    # A4 usable width with 15 mm margins.
    usable_width = 180 * mm

    if column_count <= 4:
        col_widths = [usable_width / column_count] * column_count
    else:
        # Give the last/observation columns more room.
        base = usable_width / column_count
        col_widths = [base] * column_count

        for index, header in enumerate(headers):
            h = header.lower()

            if any(
                word in h
                for word in [
                    "observation",
                    "evidence",
                    "impact",
                    "investigation",
                    "explanation",
                    "information",
                    "question"
                ]
            ):
                col_widths[index] = base * 1.45

        total = sum(col_widths)
        col_widths = [
            width * usable_width / total
            for width in col_widths
        ]

    table = Table(
        data,
        colWidths=col_widths,
        repeatRows=1,
        hAlign="LEFT"
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#17365D")
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                colors.white
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.35,
                colors.HexColor("#B7B7B7")
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),
            (
                "ROWBACKGROUNDS",
                (0, 1),
                (-1, -1),
                [
                    colors.white,
                    colors.HexColor("#F5F7FA")
                ]
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                5
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                5
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                5
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                5
            )
        ])
    )

    return table


# ---------------------------------------------------------
# 5. HEADER / FOOTER
# ---------------------------------------------------------

def add_page_number(canvas, doc):

    canvas.saveState()

    width, height = A4

    canvas.setStrokeColor(colors.HexColor("#D9E1F2"))
    canvas.line(
        15 * mm,
        12 * mm,
        width - 15 * mm,
        12 * mm
    )

    canvas.setFont(FONT_REGULAR, 7.5)
    canvas.setFillColor(colors.HexColor("#666666"))

    canvas.drawString(
        15 * mm,
        7 * mm,
        "AI Due Diligence Report"
    )

    canvas.drawRightString(
        width - 15 * mm,
        7 * mm,
        f"Page {doc.page}"
    )

    canvas.restoreState()


# ---------------------------------------------------------
# 6. CREATE PROFESSIONAL PDF
# ---------------------------------------------------------

def create_pdf(report_text, output_path):

    styles = build_styles()

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="AI Due Diligence Report",
        author="AI Due Diligence Copilot",
        subject="Due Diligence Analysis"
    )

    story = []

    # Cover/title area.
    story.append(
        Spacer(1, 20 * mm)
    )

    story.append(
        Paragraph(
            "AI DUE DILIGENCE REPORT",
            styles["title"]
        )
    )

    story.append(
        Paragraph(
            "Evidence-Based Business and Financial Analysis",
            styles["subtitle"]
        )
    )

    story.append(
        Spacer(1, 8 * mm)
    )

    story.append(
        Paragraph(
            "<b>Report basis:</b> Analysis generated only from the "
            "documents and retrieved evidence supplied to the system.",
            styles["body"]
        )
    )

    story.append(
        Spacer(1, 10 * mm)
    )

    # FIX (structure): give the cover its own page instead of letting
    # body text run on underneath it.
    story.append(PageBreak())

    # FIX (structure): track whether we've reached a top-level "## N."
    # section yet, so each one after the first starts on a fresh page
    # instead of the whole report reading as one unbroken wall of text.
    first_top_level_section_seen = False

    # FIX (structure): a heading that lands at the very bottom of a
    # page with its content pushed to the next page reads as broken
    # structure. We hold each heading back until we know what follows
    # it, then emit heading+content together via KeepTogether so a
    # heading is never left dangling alone at a page break.
    pending_heading = None

    def flush_pending(with_flowables=None):
        nonlocal pending_heading
        if pending_heading is not None:
            group = [pending_heading] + (with_flowables or [])
            story.append(KeepTogether(group))
            pending_heading = None
        elif with_flowables:
            story.extend(with_flowables)

    # Report content.
    lines = report_text.splitlines()

    i = 0

    while i < len(lines):

        raw_line = lines[i].strip()

        if not raw_line:
            flush_pending()
            story.append(Spacer(1, 4))
            i += 1
            continue

        # Ignore an LLM-generated duplicate title.
        if raw_line.lower() in {
            "# ai due diligence report",
            "ai due diligence report"
        }:
            i += 1
            continue

        # Markdown headings.
        if raw_line.startswith("### "):
            heading = raw_line[4:].strip()

            flush_pending()
            pending_heading = Paragraph(
                clean_markdown(heading),
                styles["h2"]
            )

            i += 1
            continue

        if raw_line.startswith("## "):
            heading = raw_line[3:].strip()

            flush_pending()

            if first_top_level_section_seen:
                story.append(PageBreak())

            first_top_level_section_seen = True

            pending_heading = Paragraph(
                clean_markdown(heading),
                styles["h1"]
            )

            i += 1
            continue

        if raw_line.startswith("# "):
            heading = raw_line[2:].strip()

            flush_pending()
            pending_heading = Paragraph(
                clean_markdown(heading),
                styles["h1"]
            )

            i += 1
            continue

        # Markdown table.
        if (
            "|" in raw_line
            and i + 1 < len(lines)
            and is_table_separator(lines[i + 1])
        ):

            headers, rows, next_index = parse_table(
                lines,
                i
            )

            if headers:
                table = make_table(
                    headers,
                    rows,
                    styles
                )

                flush_pending([
                    Spacer(1, 4),
                    table,
                    Spacer(1, 10)
                ])

            i = next_index
            continue

        # "Source: ..." captions read better as a small muted note
        # directly under the table/paragraph they refer to.
        if raw_line.lower().startswith("source:"):

            flush_pending([
                Paragraph(
                    clean_markdown(raw_line),
                    styles["small"]
                )
            ])

            i += 1
            continue

        # Markdown bullet.
        if raw_line.startswith("- "):

            content = raw_line[2:].strip()

            flush_pending([
                Paragraph(
                    "\u2022 " + clean_markdown(content),
                    styles["bullet"]
                )
            ])

            i += 1
            continue

        # Numbered list.
        if re.match(r"^\d+\.\s+", raw_line):

            content = re.sub(
                r"^\d+\.\s+",
                "",
                raw_line
            )

            # Keep investigation questions compact.
            flush_pending([
                Paragraph(
                    clean_markdown(raw_line),
                    styles["bullet"]
                )
            ])

            i += 1
            continue

        # Bold-only subsection text.
        if (
            raw_line.startswith("**")
            and raw_line.endswith("**")
            and raw_line.count("**") == 2
        ):

            heading = raw_line[2:-2].strip()

            flush_pending()
            pending_heading = Paragraph(
                f"<b>{html.escape(heading)}</b>",
                styles["h2"]
            )

            i += 1
            continue

        # Normal paragraph.
        flush_pending([
            Paragraph(
                clean_markdown(raw_line),
                styles["body"]
            )
        ])

        i += 1

    # Flush any heading left pending at the very end of the document.
    flush_pending()

    doc.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number
    )

    return output_path