"""Export generated job postings to downloadable PDF files."""
import re
from html import escape
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


def create_job_posting_pdf(title: str, text: str) -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=title,
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="JobTitle",
        parent=styles["Title"],
        alignment=TA_LEFT,
        textColor=colors.HexColor("#17324D"),
        spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        name="JobSection",
        parent=styles["Heading2"],
        textColor=colors.HexColor("#17324D"),
        spaceBefore=10,
        spaceAfter=5,
    ))
    styles.add(ParagraphStyle(
        name="JobSubsection",
        parent=styles["Heading3"],
        textColor=colors.HexColor("#315B7D"),
        spaceBefore=7,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        name="JobBody",
        parent=styles["BodyText"],
        leading=15,
        spaceAfter=5,
    ))

    content = []
    heading_seen = False
    for line in _clean_lines(text):
        line = line.strip()
        if not line:
            if content and not isinstance(content[-1], Spacer):
                content.append(Spacer(1, 4))
        elif heading := re.match(r"^#{1,6}\s+(.+)$", line):
            if not heading_seen:
                style = styles["JobTitle"]
                heading_seen = True
            elif len(line) - len(line.lstrip("#")) >= 3:
                style = styles["JobSubsection"]
            else:
                style = styles["JobSection"]
            content.append(Paragraph(_format_inline(heading.group(1)), style))
        elif line.startswith(("- ", "* ")):
            content.append(Paragraph(
                _format_inline(line[2:]), styles["JobBody"], bulletText="-"
            ))
        elif re.match(r"^\d+[.)]\s+", line):
            number, item = re.match(r"^(\d+[.)])\s+(.+)$", line).groups()
            content.append(Paragraph(
                _format_inline(item), styles["JobBody"], bulletText=number
            ))
        else:
            content.append(Paragraph(_format_inline(line), styles["JobBody"]))

    document.build(content)
    return buffer.getvalue()


def _clean_lines(text: str) -> list[str]:
    lines = text.splitlines()
    heading_index = next(
        (index for index, line in enumerate(lines) if re.match(r"^\s{0,3}#{1,6}\s+\S", line)),
        None,
    )
    if heading_index is not None:
        lines = lines[heading_index:]
    else:
        while lines and (
            not lines[0].strip()
            or re.match(r"^(?:voici|bien sûr|certainement)\b", lines[0].strip(), re.IGNORECASE)
        ):
            lines.pop(0)
    return [
        line for line in lines
        if not re.match(r"^\s*(?:\*{3,}|-{3,}|_{3,})\s*$", line)
    ]


def _format_inline(text: str) -> str:
    escaped_text = escape(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped_text)
