"""Trade Document OCR Checker — Streamlit UI.

Upload a commercial invoice, packing list and/or bill of lading (bilingual
Chinese/English images). A vision LLM extracts the key fields, then a
deterministic checker verifies the arithmetic inside each document and the
agreement across documents — the two checks a trade operator does by hand
on every shipment.

Built during Build North (October 2026) by Jian (Nick) Wang.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import streamlit as st

import checker

st.set_page_config(page_title="Trade Document Checker", page_icon="📦", layout="wide")

SAMPLES_DIR = Path(__file__).parent / "samples"
DOC_SLOTS = [
    ("invoice", "Commercial Invoice 商业发票", "sample-01-commercial-invoice.png"),
    ("packing_list", "Packing List 装箱单", "sample-02-packing-list.png"),
    ("bill_of_lading", "Bill of Lading 海运提单", "sample-03-bill-of-lading.png"),
]
MEDIA_TYPES = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg",
               "webp": "image/webp", "gif": "image/gif"}


def get_api_key() -> str | None:
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        key = None
    return key or os.environ.get("ANTHROPIC_API_KEY")


def media_type_of(name: str) -> str:
    return MEDIA_TYPES.get(name.rsplit(".", 1)[-1].lower(), "image/png")


st.title("📦 Trade Document Checker 贸易单证智能核查")
st.caption(
    "Upload the invoice, packing list and bill of lading for one shipment. "
    "The checker extracts the key fields with a vision model, re-does the math "
    "inside each document, and flags every field that disagrees across documents."
)

api_key = get_api_key()
model = os.environ.get("CHECKER_MODEL", checker.DEFAULT_MODEL)

with st.sidebar:
    st.header("Settings")
    if api_key:
        st.success("Vision model API key: configured")
    else:
        st.warning("No API key configured. Add ANTHROPIC_API_KEY to the app "
                   "secrets (Streamlit Cloud) or environment to run extraction.")
    st.text_input("Vision model", value=model, disabled=True)
    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. Upload 1–3 document images\n"
        "2. Claude Vision extracts structured fields\n"
        "3. A rule engine verifies arithmetic (qty × price, column totals) "
        "and cross-document agreement (consignee, cartons, weight, volume, dates)\n\n"
        "Sample documents are synthetic — no real customer data."
    )

if "docs" not in st.session_state:
    st.session_state.docs = {}       # slot -> {"name", "bytes"}
if "extracted" not in st.session_state:
    st.session_state.extracted = {}  # slot -> extracted dict
if "results" not in st.session_state:
    st.session_state.results = None

cols = st.columns(3)
for col, (slot, label, sample_name) in zip(cols, DOC_SLOTS):
    with col:
        st.subheader(label)
        uploaded = st.file_uploader(label, type=list(MEDIA_TYPES), key=f"up_{slot}",
                                    label_visibility="collapsed")
        if uploaded is not None:
            st.session_state.docs[slot] = {"name": uploaded.name,
                                           "bytes": uploaded.getvalue()}
        current = st.session_state.docs.get(slot)
        if current:
            st.image(current["bytes"], caption=current["name"], use_container_width=True)
        else:
            st.info("No document uploaded yet.")

load_col, run_col = st.columns([1, 2])
with load_col:
    if st.button("📂 Load sample documents", use_container_width=True):
        for slot, _label, sample_name in DOC_SLOTS:
            path = SAMPLES_DIR / sample_name
            if path.exists():
                st.session_state.docs[slot] = {"name": sample_name,
                                               "bytes": path.read_bytes()}
        st.session_state.extracted = {}
        st.session_state.results = None
        st.rerun()

with run_col:
    run_clicked = st.button("🔍 Run checks", type="primary", use_container_width=True,
                            disabled=not st.session_state.docs)

if run_clicked:
    if not api_key:
        st.error("Extraction needs an Anthropic API key. Configure ANTHROPIC_API_KEY "
                 "and try again.")
    else:
        extracted = {}
        for slot, _label, _sample in DOC_SLOTS:
            doc = st.session_state.docs.get(slot)
            if not doc:
                continue
            digest = hashlib.sha256(doc["bytes"]).hexdigest()
            cache_key = f"{slot}:{digest}"
            if st.session_state.get("cache_key") == cache_key and slot in st.session_state.extracted:
                extracted[slot] = st.session_state.extracted[slot]
                continue
            with st.spinner(f"Extracting {_label}…"):
                try:
                    extracted[slot] = checker.extract_document(
                        doc["bytes"], media_type_of(doc["name"]), slot,
                        api_key=api_key, model=model)
                except Exception as exc:  # surface API/parse errors, keep going
                    st.error(f"Extraction failed for {_label}: {exc}")
            st.session_state["cache_key"] = cache_key
        st.session_state.extracted = extracted
        st.session_state.results = checker.run_all_checks(extracted) if extracted else []

results = st.session_state.results
if results is not None:
    fails = [r for r in results if r.status == "fail"]
    passes = [r for r in results if r.status == "pass"]
    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Documents processed", len(st.session_state.extracted))
    m2.metric("Checks passed", len(passes))
    m3.metric("⚠️ Mismatches found", len(fails))

    if fails:
        st.subheader("⚠️ Mismatches to fix before shipping")
        for r in fails:
            st.error(f"**{r.name}** — {r.detail}")
    else:
        st.success("No mismatches found. All checks passed.")

    st.subheader("All checks")
    icon = {"pass": "✅", "fail": "❌", "info": "ℹ️"}
    st.table([{"Check": r.name, "Result": f"{icon[r.status]} {r.status.upper()}",
               "Detail": r.detail} for r in results])

    st.subheader("Extracted data")
    for slot, _label, _sample in DOC_SLOTS:
        data = st.session_state.extracted.get(slot)
        if data:
            with st.expander(_label):
                st.json(data)
