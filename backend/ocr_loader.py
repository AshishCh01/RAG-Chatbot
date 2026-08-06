from pathlib import Path
from typing import List

from PIL import Image
import pytesseract
from langchain_core.documents import Document

# Image extensions that will be routed through OCR
OCR_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def load_image_with_ocr(file_path: str) -> List[Document]:
    """Runs OCR on an image file and wraps the extracted text in a LangChain Document
    so it can flow through the same chunking/embedding/ingestion pipeline as any
    other document type (pdf, docx, csv, txt).
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        extracted_text = pytesseract.image_to_string(image).strip()
    except pytesseract.TesseractNotFoundError as e:
        raise RuntimeError(
            "Tesseract OCR engine is not installed or not on PATH. "
            "Install it (e.g. `brew install tesseract`, `sudo apt install tesseract-ocr`, "
            "or the Windows installer from https://github.com/UB-Mannheim/tesseract/wiki) "
            "and make sure the `tesseract` command is available."
        ) from e
    except Exception as e:
        raise RuntimeError(f"Failed to run OCR on {path.name}: {e}") from e

    if not extracted_text:
        print(f"[WARN] OCR found no readable text in {path.name}")
        extracted_text = ""

    print(f"[SUCCESS] OCR extracted {len(extracted_text)} character(s) from {path.name}")

    return [
        Document(
            page_content=extracted_text,
            metadata={"source": str(path), "type": "image_ocr"},
        )
    ]




# from pathlib import Path
# from typing import List

# from PIL import Image
# import pytesseract
# from langchain_core.documents import Document

# # Image extensions that get routed through OCR instead of a LangChain loader
# OCR_SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


# def load_image_with_ocr(file_path: str) -> List[Document]:
#     """Runs OCR on an image file and wraps the extracted text in a LangChain Document
#     so it can flow through the same chunking/embedding/ingestion pipeline as any
#     other document type (pdf, docx, csv, txt)."""
#     path = Path(file_path)
#     if not path.exists() or not path.is_file():
#         raise FileNotFoundError(f"File not found: {file_path}")

#     try:
#         image = Image.open(path)
#         if image.mode != "RGB":
#             image = image.convert("RGB")

#         extracted_text = pytesseract.image_to_string(image).strip()

#     except pytesseract.TesseractNotFoundError as e:
#         raise RuntimeError(
#             "Tesseract OCR engine is not installed or not on PATH. "
#             "Install it (e.g. `brew install tesseract`, `sudo apt install tesseract-ocr`, "
#             "or the Windows installer from https://github.com/UB-Mannheim/tesseract/wiki) "
#             "and make sure the `tesseract` command is available."
#         ) from e

#     except Exception as e:
#         raise RuntimeError(f"Failed to run OCR on {path.name}: {e}") from e

#     if not extracted_text:
#         print(f"[WARN] OCR found no readable text in {path.name}")

#     print(f"[SUCCESS] OCR extracted {len(extracted_text)} character(s) from {path.name}")

#     return [
#         Document(
#             page_content=extracted_text,
#             metadata={"source": str(path), "type": "image_ocr"}
#         )
#     ]


# if __name__ == "__main__":
#     # Independent execution check
#     test_path = "./documents/sample.png"
#     result = load_image_with_ocr(test_path)
#     print(f"[SUCCESS] Extracted {len(result[0].page_content)} character(s) from {test_path}")