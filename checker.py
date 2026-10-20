"""Core logic for the Trade Document OCR Checker.

Extracts structured fields from bilingual (Chinese/English) trade documents
(commercial invoice, packing list, bill of lading) with a vision LLM, then runs
deterministic consistency checks:

- within-document arithmetic (line amounts, column totals)
- cross-document agreement (consignee, cartons, weights, volume, dates)

The check functions are pure Python and take the extracted dicts, so they can
be unit-tested without any API calls.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass

DEFAULT_MODEL = "claude-sonnet-4-5"

TOL_MONEY = 0.05
TOL_WEIGHT = 0.11
TOL_CBM = 0.011
TOL_QTY = 0.51


@dataclass
class CheckResult:
    name: str
    status: str  # "pass" | "fail" | "info"
    detail: str


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _num(value):
    """Best-effort float parse: accepts numbers and strings like '6,650.00' or '640.0 kg'."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        m = re.search(r"-?\d[\d,]*\.?\d*", value.replace("，", ","))
        if m:
            try:
                return float(m.group(0).replace(",", ""))
            except ValueError:
                return None
    return None


def _norm_name(value):
    """Normalize a company name for comparison: case/punctuation/space-insensitive."""
    if not value:
        return ""
    s = str(value).lower()
    s = re.split(r"[，,（(]", s)[0]  # drop Chinese name / parenthetical suffixes
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def _close(a, b, tol):
    return a is not None and b is not None and abs(a - b) <= tol


def _fmt(x):
    if x is None:
        return "—"
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x)):,}"
    return f"{x:,.2f}"


# --------------------------------------------------------------------------
# extraction (vision LLM)
# --------------------------------------------------------------------------

_SCHEMAS = {
    "invoice": """{
  "doc_no": string,
  "date": "YYYY-MM-DD",
  "currency": string (e.g. "USD"),
  "seller": string (company name only),
  "buyer": string (company name only),
  "items": [{"description": string, "hs_code": string or null, "quantity": number, "unit": string, "unit_price": number, "amount": number}],
  "total_amount": number
}""",
    "packing_list": """{
  "doc_no": string,
  "date": "YYYY-MM-DD",
  "ref_invoice": string or null,
  "seller": string (company name only),
  "buyer": string (company name only),
  "items": [{"description": string, "cartons": number, "net_weight_kg": number, "gross_weight_kg": number}],
  "total_cartons": number,
  "total_net_weight_kg": number,
  "total_gross_weight_kg": number,
  "total_volume_cbm": number
}""",
    "bill_of_lading": """{
  "bl_no": string,
  "date": "YYYY-MM-DD",
  "shipper": string (company name only),
  "consignee": string (company name only),
  "vessel": string or null,
  "voyage": string or null,
  "port_of_loading": string or null,
  "port_of_discharge": string or null,
  "final_destination": string or null,
  "packages": number,
  "package_unit": string (e.g. "cartons"),
  "gross_weight_kg": number,
  "volume_cbm": number
}""",
}

_DOC_LABELS = {
    "invoice": "commercial invoice (商业发票)",
    "packing_list": "packing list (装箱单)",
    "bill_of_lading": "bill of lading (海运提单)",
}


def build_extraction_prompt(doc_type: str) -> str:
    return (
        f"This image is a bilingual (Chinese/English) {_DOC_LABELS[doc_type]} used in "
        "international trade. Extract the fields into JSON matching exactly this schema:\n\n"
        f"{_SCHEMAS[doc_type]}\n\n"
        "Rules:\n"
        "- Output ONLY the JSON object, no markdown fences, no commentary.\n"
        "- All amounts, quantities and weights must be plain numbers (no thousands "
        "separators, no units, no currency symbols).\n"
        "- Weights in kilograms, volume in CBM.\n"
        "- Company names: use the English name as printed; if both English and Chinese "
        "names are printed, return the English one.\n"
        "- Use null for any field that is not present on the document.\n"
        "- Copy numbers exactly as printed, even if they look wrong. Do not correct "
        "arithmetic; a separate checker verifies the math."
    )


def extract_document(image_bytes: bytes, media_type: str, doc_type: str,
                     api_key: str, model: str = DEFAULT_MODEL) -> dict:
    """Send one document image to the vision model and return the extracted dict."""
    import anthropic  # imported lazily so the check engine stays dependency-free

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.standard_b64encode(image_bytes).decode("utf-8"),
                }},
                {"type": "text", "text": build_extraction_prompt(doc_type)},
            ],
        }],
    )
    text = "".join(block.text for block in message.content if block.type == "text")
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    return json.loads(text)


# --------------------------------------------------------------------------
# within-document checks
# --------------------------------------------------------------------------

def check_invoice(data: dict) -> list[CheckResult]:
    results = []
    items = data.get("items") or []
    stated_sum = 0.0
    for i, item in enumerate(items, start=1):
        qty = _num(item.get("quantity"))
        price = _num(item.get("unit_price"))
        amount = _num(item.get("amount"))
        if amount is not None:
            stated_sum += amount
        desc = item.get("description") or f"line {i}"
        if qty is None or price is None or amount is None:
            results.append(CheckResult(
                f"Invoice line {i} amount", "info",
                f"{desc}: quantity/unit price/amount not fully readable, skipped."))
            continue
        expected = qty * price
        if _close(expected, amount, TOL_MONEY):
            results.append(CheckResult(
                f"Invoice line {i} amount", "pass",
                f"{desc}: {_fmt(qty)} × {_fmt(price)} = {_fmt(expected)} ✓"))
        else:
            results.append(CheckResult(
                f"Invoice line {i} amount", "fail",
                f"{desc}: {_fmt(qty)} × {_fmt(price)} = {_fmt(expected)}, "
                f"but the document states {_fmt(amount)}."))
    total = _num(data.get("total_amount"))
    if total is not None and items:
        if _close(stated_sum, total, TOL_MONEY):
            results.append(CheckResult(
                "Invoice total", "pass",
                f"Line amounts add up to the stated total {_fmt(total)}."))
        else:
            results.append(CheckResult(
                "Invoice total", "fail",
                f"Line amounts add up to {_fmt(stated_sum)}, but the stated total is {_fmt(total)}."))
    return results


def check_packing_list(data: dict) -> list[CheckResult]:
    results = []
    items = data.get("items") or []
    sums = {
        "total_cartons": ("cartons", TOL_QTY, "Packing list total cartons"),
        "total_net_weight_kg": ("net_weight_kg", TOL_WEIGHT, "Packing list total net weight"),
        "total_gross_weight_kg": ("gross_weight_kg", TOL_WEIGHT, "Packing list total gross weight"),
    }
    for total_key, (item_key, tol, name) in sums.items():
        stated = _num(data.get(total_key))
        if stated is None or not items:
            continue
        computed = sum(v for v in (_num(it.get(item_key)) for it in items) if v is not None)
        if _close(computed, stated, tol):
            results.append(CheckResult(
                name, "pass",
                f"Rows add up to the stated total {_fmt(stated)}."))
        else:
            results.append(CheckResult(
                name, "fail",
                f"Rows add up to {_fmt(computed)}, but the stated total is {_fmt(stated)}."))
    return results


def check_bill_of_lading(data: dict) -> list[CheckResult]:
    # The B/L fixtures carry a single summary row, so there is nothing internal
    # to recompute; presence of the key fields is cross-checked instead.
    missing = [k for k in ("consignee", "packages", "gross_weight_kg") if data.get(k) in (None, "")]
    if missing:
        return [CheckResult("B/L key fields", "info",
                            f"Not readable on the B/L: {', '.join(missing)}.")]
    return [CheckResult("B/L key fields", "pass",
                        "Consignee, packages and gross weight are present.")]


_WITHIN = {
    "invoice": check_invoice,
    "packing_list": check_packing_list,
    "bill_of_lading": check_bill_of_lading,
}


# --------------------------------------------------------------------------
# cross-document checks
# --------------------------------------------------------------------------

def cross_checks(docs: dict) -> list[CheckResult]:
    """docs maps doc_type -> extracted dict. Only checks fields present in 2+ docs."""
    results = []
    inv, pl, bl = docs.get("invoice"), docs.get("packing_list"), docs.get("bill_of_lading")

    # Consignee / buyer name across all documents that name one.
    names = {}
    if inv and inv.get("buyer"):
        names["invoice buyer"] = inv["buyer"]
    if pl and pl.get("buyer"):
        names["packing list buyer"] = pl["buyer"]
    if bl and bl.get("consignee"):
        names["B/L consignee"] = bl["consignee"]
    if len(names) >= 2:
        normalized = {_norm_name(v) for v in names.values()}
        detail = "; ".join(f"{k}: “{v}”" for k, v in names.items())
        if len(normalized) == 1:
            results.append(CheckResult("Consignee name consistency", "pass", detail))
        else:
            results.append(CheckResult("Consignee name consistency", "fail",
                                        f"Names do not match — {detail}"))

    # Total cartons: packing list vs B/L packages.
    if pl and bl:
        a, b = _num(pl.get("total_cartons")), _num(bl.get("packages"))
        if a is not None and b is not None:
            if _close(a, b, TOL_QTY):
                results.append(CheckResult("Total cartons (PL vs B/L)", "pass",
                                           f"Both state {_fmt(a)} cartons."))
            else:
                results.append(CheckResult("Total cartons (PL vs B/L)", "fail",
                                           f"Packing list states {_fmt(a)} cartons, B/L states {_fmt(b)}."))

    # Gross weight: packing list stated total vs B/L.
    if pl and bl:
        a, b = _num(pl.get("total_gross_weight_kg")), _num(bl.get("gross_weight_kg"))
        if a is not None and b is not None:
            if _close(a, b, TOL_WEIGHT):
                results.append(CheckResult("Gross weight (PL vs B/L)", "pass",
                                           f"Both state {_fmt(a)} kg."))
            else:
                results.append(CheckResult("Gross weight (PL vs B/L)", "fail",
                                           f"Packing list states {_fmt(a)} kg, B/L states {_fmt(b)} kg."))

    # Volume: packing list vs B/L.
    if pl and bl:
        a, b = _num(pl.get("total_volume_cbm")), _num(bl.get("volume_cbm"))
        if a is not None and b is not None:
            if _close(a, b, TOL_CBM):
                results.append(CheckResult("Volume (PL vs B/L)", "pass",
                                           f"Both state {_fmt(a)} CBM."))
            else:
                results.append(CheckResult("Volume (PL vs B/L)", "fail",
                                           f"Packing list states {_fmt(a)} CBM, B/L states {_fmt(b)} CBM."))

    # Currency consistency wherever stated.
    currencies = {d.get("currency") for d in (inv, pl, bl) if d and d.get("currency")}
    if len(currencies) > 1:
        results.append(CheckResult("Currency consistency", "fail",
                                   f"Documents use different currencies: {sorted(currencies)}."))
    elif len(currencies) == 1:
        results.append(CheckResult("Currency consistency", "pass",
                                   f"Currency is {next(iter(currencies))} where stated."))

    # Date logic: invoice and packing list should not be dated after the B/L.
    def _date(d):
        v = (d or {}).get("date")
        return v if isinstance(v, str) and re.match(r"^\d{4}-\d{2}-\d{2}$", v) else None

    bl_date = _date(bl)
    if bl_date:
        bad = []
        for label, d in (("Invoice", inv), ("Packing list", pl)):
            d_date = _date(d)
            if d_date and d_date > bl_date:
                bad.append(f"{label} is dated {d_date}, after the B/L date {bl_date}.")
        if bad:
            results.append(CheckResult("Date logic", "fail", " ".join(bad)))
        else:
            results.append(CheckResult("Date logic", "pass",
                                       f"No document is dated after the B/L ({bl_date})."))
    return results


def run_all_checks(docs: dict) -> list[CheckResult]:
    results: list[CheckResult] = []
    for doc_type, data in docs.items():
        fn = _WITHIN.get(doc_type)
        if fn and data:
            results.extend(fn(data))
    if len([d for d in docs.values() if d]) >= 2:
        results.extend(cross_checks(docs))
    return results
