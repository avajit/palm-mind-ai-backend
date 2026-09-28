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
    """Extracts text from a standard TXT file."""
    # Assuming UTF-8 encoding for standard text files
    return file_bytes.decode("utf-8").strip()

def extract_text(file_bytes: bytes, filename: str) -> str:
    """Router function to extract text based on the file extension."""
    if filename.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif filename.lower().endswith(".txt"):
        return extract_text_from_txt(file_bytes)
    else:
        raise ValueError("Unsupported file type. Only .pdf and .txt are allowed.")
