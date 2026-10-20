import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from checker import run_all_checks

inv = {"doc_no": "INV-2026-1018", "date": "2026-10-18", "currency": "USD",
  "seller": "Shenzhen Bright Tech Co., Ltd.", "buyer": "Maple Leaf Trading Inc.",
  "items": [
    {"description": "Silicone Kitchenware Set (12pcs)", "hs_code": "3924.10", "quantity": 500, "unit": "sets", "unit_price": 8.50, "amount": 4250.00},
    {"description": "Bone Conduction Headphones", "hs_code": "8518.30", "quantity": 300, "unit": "pcs", "unit_price": 7.90, "amount": 2400.00}],
  "total_amount": 6650.00}
pl = {"doc_no": "PL-2026-1018", "date": "2026-10-18", "ref_invoice": "INV-2026-1018",
  "seller": "Shenzhen Bright Tech Co., Ltd.", "buyer": "Maple Leaf Trading Inc.",
  "items": [
    {"description": "Silicone Kitchenware Set", "cartons": 25, "net_weight_kg": 462.5, "gross_weight_kg": 500.0},
    {"description": "Bone Conduction Headphones", "cartons": 15, "net_weight_kg": 120.0, "gross_weight_kg": 138.0}],
  "total_cartons": 40, "total_net_weight_kg": 582.5, "total_gross_weight_kg": 640.0, "total_volume_cbm": 4.80}
bl = {"bl_no": "HBL-SH-TOR-1018", "date": "2026-10-20",
  "shipper": "Shenzhen Bright Tech Co., Ltd.", "consignee": "Maple Trading Inc.",
  "vessel": "MV PACIFIC TRADER", "voyage": "V.012E",
  "port_of_loading": "Shanghai", "port_of_discharge": "Vancouver", "final_destination": "Toronto",
  "packages": 42, "package_unit": "cartons", "gross_weight_kg": 640.0, "volume_cbm": 4.80}

results = run_all_checks({"invoice": inv, "packing_list": pl, "bill_of_lading": bl})
for r in results:
    print(f"{r.status.upper():4}  {r.name}: {r.detail}")
fails = {r.name for r in results if r.status == "fail"}
expected = {"Invoice line 2 amount", "Packing list total gross weight",
            "Consignee name consistency", "Total cartons (PL vs B/L)"}
print("\nFAILS:", sorted(fails))
assert fails == expected, f"MISMATCH: got {sorted(fails)} expected {sorted(expected)}"
print("ACCEPTANCE OK: exactly the 4 planted anomalies flagged, no false positives")
