"""Financial PDF extraction package.

Pipeline
--------
  announcements (SQL, tags=financial results)
      → PDFDownloader  (urllib, NSE/BSE archive)
      → FinancialExtractor  (Claude claude-opus-4-7, PDF base64 → structured JSON)
      → FinancialStorage  (SQL financial_results table + ep_financials ChromaDB)
      → PDFAgent  (orchestrates the full run)
"""
from financial.models import FinancialResult, ExtractionStatus
from financial.pdf_downloader import PDFDownloader
from financial.extractor import FinancialExtractor
from financial.storage import FinancialStorage
from financial.pdf_agent import PDFAgent

__all__ = [
    "FinancialResult", "ExtractionStatus",
    "PDFDownloader", "FinancialExtractor",
    "FinancialStorage", "PDFAgent",
]
