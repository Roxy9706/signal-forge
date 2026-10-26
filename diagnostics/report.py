import json
from diagnostics.d7_diagnostic import D7_DIAGNOSTIC_CATEGORIES
def build_report(records):
    bd = {c: 0 for c in D7_DIAGNOSTIC_CATEGORIES}
    p = 0
    for r in records:
        if r.get("d7_pass") is True: p += 1
        c = r.get("d7_diagnostic_category")
        if c in bd: bd[c] += 1
    return {"total_records": len(records), "d7_pass": p, "d7_diagnostic_breakdown": bd}
def format_text(report):
    bd = report.get("d7_diagnostic_breakdown", {})
    lines = ["D7 diagnostic category breakdown:"]
    pad = {"RETEST_LEVEL_UNAVAILABLE": 29, "D7_PASS": 29}
    for c in D7_DIAGNOSTIC_CATEGORIES:
        k = c + ":"
        lines.append(f"  {k:<{pad.get(c, 30)}}{bd.get(c, 0)}")
    d = "D7_PASS:"
    lines.append(f"  {d:<29}{report.get('d7_pass', 0)}")
    return "\n".join(lines)
def format_json(report):
    return json.dumps(report, indent=2)
