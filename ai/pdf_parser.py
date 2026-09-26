from io import BytesIO

from pypdf import PdfReader
from docx import Document


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    Extract text from a PDF resume.
    """

    reader = PdfReader(BytesIO(file_bytes))

    text = []

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text.append(page_text)

    return "\n".join(text).strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    """
    Extract text from a DOCX resume.
    """

    document = Document(BytesIO(file_bytes))

    text = []

    # Extract normal paragraphs
    for paragraph in document.paragraphs:
        content = paragraph.text.strip()

        if content:
            text.append(content)

    # Extract text from tables
    for table in document.tables:
        for row in table.rows:
            for cell in row.cells:
                content = cell.text.strip()

                if content:
                    text.append(content)

    return "\n".join(text).strip()


def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Automatically detect the resume file type
    and extract its text.
    """

    filename = filename.lower()

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)

    if filename.endswith(".docx"):
        return extract_text_from_docx(file_bytes)

    raise ValueError(
        "Unsupported file type. Please upload a PDF or DOCX file."
    )


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("AI RESUME ANALYZER")
    print("PDF / DOCX PARSER TEST")
    print("=" * 60)

    print("\nParser module loaded successfully.")

    print("\nSUPPORTED FORMATS:")
    print("- PDF")
    print("- DOCX")

    print("\nAVAILABLE FUNCTIONS:")
    print("- extract_text_from_pdf()")
    print("- extract_text_from_docx()")
    print("- extract_text()")

    print("\nTest completed successfully.")