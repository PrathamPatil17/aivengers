import fitz 

from typing import List, Optional

def extract_text_from_pdf(pdf_path:str) -> List[str]:
    """
    Extracts text from each page in a PDF.
    Returns a list where each item is the text of one page.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text").replace("\u00ad", "")
        pages.append(text.strip())
    return pages


        
        
    