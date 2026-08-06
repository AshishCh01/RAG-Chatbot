import io
from pathlib import Path
from typing import List

from PIL import Image
import pytesseract
from langchain_core.documents import Document

# Image extensions that will be routed through OCR
OCR_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def _run_ocr(image: Image.Image) -> str:
    """Runs Tesseract OCR on a Pillow Image object and returns cleaned text."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    try:
        return pytesseract.image_to_string(image).strip()
    except pytesseract.TesseractNotFoundError as e:
        raise RuntimeError(
            "Tesseract OCR engine is not installed or not on PATH. "
            "Install it (e.g. `brew install tesseract`, `sudo apt install tesseract-ocr`, "
            "or the Windows installer from https://github.com/UB-Mannheim/tesseract/wiki) "
            "and make sure the `tesseract` command is available."
        ) from e


def load_image_with_ocr(file_path: str) -> List[Document]:
    """Runs OCR on an image FILE ON DISK and wraps the extracted text in a LangChain
    Document so it can flow through the ingestion pipeline (used by /upload)."""
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        image = Image.open(path)
        extracted_text = _run_ocr(image)
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to run OCR on {path.name}: {e}") from e

    if not extracted_text:
        print(f"[WARN] OCR found no readable text in {path.name}")

    print(f"[SUCCESS] OCR extracted {len(extracted_text)} character(s) from {path.name}")

    return [
        Document(
            page_content=extracted_text,
            metadata={"source": str(path), "type": "image_ocr"},
        )
    ]


def extract_text_from_image_bytes(image_bytes: bytes, filename: str = "chat_image") -> str:
    """Runs OCR on raw image BYTES (no file saved to disk) and returns plain text.
    Used for images attached directly in the chat input."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        extracted_text = _run_ocr(image)
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Failed to run OCR on {filename}: {e}") from e

    if not extracted_text:
        print(f"[WARN] OCR found no readable text in {filename}")

    print(f"[SUCCESS] OCR extracted {len(extracted_text)} character(s) from {filename}")
    return extracted_text