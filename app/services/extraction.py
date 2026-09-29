import pymupdf  # PyMuPDF
import io

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extracts text from a PDF file using PyMuPDF."""
    text = ""
    # Open the PDF from bytes in memory
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text.strip()

def extract_text_from_txt(file_bytes: bytes) -> str:
    """
    Extracts text from a TXT file.
    Tries multiple encodings in order: UTF-8-sig (handles BOM), UTF-16, Latin-1.
    UTF-8-sig automatically strips the BOM if present.
    Latin-1 is a catch-all since it can decode any byte sequence.
    """
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            return file_bytes.decode(encoding).strip()
        except (UnicodeDecodeError, Exception):
            continue
    raise ValueError("Could not decode the text file with any supported encoding.")

def extract_text(file_bytes: bytes, filename: str) -> str:
    """Router function to extract text based on the file extension."""
    if filename.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif filename.lower().endswith(".txt"):
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError("Unsupported file type. Only .pdf and .txt are allowed.")
