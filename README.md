# trade-doc-ocr-checker

AI-powered checker that OCRs bilingual (Chinese/English) trade documents and flags field mismatches across invoices, packing lists, and bills of lading.

**Status:** In development — built during the Build North hackathon (October 2026).
**Planned stack:** Streamlit + vision LLM (OCR/document understanding) + Python.

## The problem

Every international shipment travels with a set of documents — commercial invoice, packing list, bill of lading — that must agree with each other exactly. A wrong line amount, a mistyped consignee name, or a carton count that differs between the packing list and the B/L can mean customs delays, bank document rejection under a letter of credit, or a shipment held at destination. Today, operators still check these documents line by line, by eye. This tool automates that check.

## What it does

1. **Upload** the invoice, packing list and/or bill of lading for one shipment (images, Chinese/English).
2. **Extract** — a vision LLM (Claude) reads each document into structured fields: parties, dates, line items, quantities, unit prices, carton counts, net/gross weights, volume, ports, vessel/voyage.
3. **Verify arithmetic inside each document** — every line (quantity × unit price vs. stated amount) and every column total is recomputed.
4. **Verify agreement across documents** — consignee name, total cartons, gross weight, volume, currency, and date logic (invoice/packing list must not postdate the B/L) are compared; every disagreement is flagged in plain language.

## Run it locally

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here   # or put it in .streamlit/secrets.toml
streamlit run app.py
```

Then click **Load sample documents** to try it on the bundled synthetic samples, or upload your own document images.

## Sample documents

`samples/` contains three synthetic documents for one fictional shipment (Shanghai → Vancouver → Toronto), with an answer key (`samples/README.md`) listing the 4 deliberate errors planted in them — a wrong invoice line amount, a wrong packing-list gross-weight total, a consignee name mismatch, and a carton-count mismatch. A correct run flags exactly those 4 and passes everything else. All sample data is fictional.

## How it works

- `app.py` — Streamlit UI: uploads, sample loader, results (mismatches first, then the full check table and extracted JSON).
- `checker.py` — extraction prompts/schemas per document type plus the verification engine. The check functions are pure Python over the extracted dicts, so they are unit-testable without API calls and never "trust" the LLM's math.

## Roadmap

- PDF uploads
- Letter-of-credit (UCP 600) document rules
- Exportable discrepancy report (PDF/Excel) for the customer file

## Disclosure

Built during Build North (October 16–25, 2026). The three sample documents and their answer key were generated for this project; all data in them is synthetic. No real customer documents are used anywhere in this repository.
