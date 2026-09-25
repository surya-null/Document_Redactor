
import io
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
from typing import Tuple, Dict, Any

class DualPathPDFExtractor:
    def __init__(self, min_char_threshold: int = 50):
        """
        :param min_char_threshold: Total character count below which a document
                                  is flagged as a scan requiring OCR fallback.
        """
        self.min_char_threshold = min_char_threshold

    def extract(self, file_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
        """
        Extracts text from PDF bytes via digital text layer or OCR fallback.
        
        Returns:
            text: Extracted string content.
            metadata: Diagnostic dict containing extraction method, page count, and char count.
        """
        # Open PDF directly from memory buffer (no disk writes needed)
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        total_pages = len(doc)
        
        # 1. First Pass: Attempt digital text extraction
        digital_text_parts = []
        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            text = page.get_text("text").strip()
            if text:
                digital_text_parts.append(text)

        combined_digital_text = "\n".join(digital_text_parts)
        char_count = len(combined_digital_text)

        # 2. Check if digital extraction succeeded
        if char_count >= self.min_char_threshold:
            metadata = {
                "method": "digital",
                "is_scanned": False,
                "total_pages": total_pages,
                "char_count": char_count
            }
            return combined_digital_text, metadata

        # 3. Fallback: Document is an image/scan -> Run Tesseract OCR
        ocr_text_parts = []
        for page_num in range(total_pages):
            page = doc.load_page(page_num)
            
            # Render page to an RGB image at 300 DPI for reliable OCR accuracy
            # Default PDF resolution is 72 DPI; 300/72 ≈ 4.16x zoom
            zoom_matrix = fitz.Matrix(300 / 72, 300 / 72)
            pixmap = page.get_pixmap(matrix=zoom_matrix)
            
            # Convert raw pixmap bytes to PIL Image in-memory
            img = Image.open(io.BytesIO(pixmap.tobytes("png")))
            
            # Run local Tesseract OCR engine
            page_ocr_text = pytesseract.image_to_string(img)
            ocr_text_parts.append(page_ocr_text)

        combined_ocr_text = "\n".join(ocr_text_parts)

        metadata = {
            "method": "ocr_tesseract",
            "is_scanned": True,
            "total_pages": total_pages,
            "char_count": len(combined_ocr_text)
        }
        return combined_ocr_text, metadata