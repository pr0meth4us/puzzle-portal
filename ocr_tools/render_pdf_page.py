import fitz  # PyMuPDF

doc = fitz.open("/Users/nicksng/code/puzzle-portal/ocr_tools/reports/13-June-2026.pdf")
page = doc[0]
pix = page.get_pixmap(dpi=150)
pix.save("/Users/nicksng/code/puzzle-portal/ocr_tools/results/test_pdf_page1.jpg")
print("Rendered page 1 of PDF successfully.")
