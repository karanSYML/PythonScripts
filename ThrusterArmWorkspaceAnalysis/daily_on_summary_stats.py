import openpyxl
from collections import defaultdict

XLSX_PATH = "pdsk_6month_maneuverPlan.xlsx"
SHEET_NAME = "Daily_ON_Summary"

wb = openpyxl.load_workbook(XLSX_PATH, data_only=True)
ws = wb[SHEET_NAME]

rows = list(ws.iter_rows(values_only=True))
header = rows[0]  # ('Direction', 'Date', 'ON_Count', 'ON_Duration_Minutes')
data = [r for r in rows[1:] if any(c is not None for c in r)]

# Accumulate per-direction totals
totals = defaultdict(lambda: {"count_sum": 0.0, "duration_sum": 0.0, "days": 0})

for direction, date, on_count, on_duration in data:
    t = totals[direction]
    t["count_sum"] += on_count or 0.0
    t["duration_sum"] += on_duration or 0.0
    t["days"] += 1

print(f"{'Direction':<12} {'Days w/ Data':>12} {'Avg Events/Day':>15} {'Avg Duration/Day (s)':>22}")
print("-" * 65)

for direction in sorted(totals):
    t = totals[direction]
    avg_events = t["count_sum"] / 180.0 #t["days"]
    avg_duration = 60 * t["duration_sum"] / 180.0 #t["days"]
    print(f"{direction:<12} {t['days']:>12} {avg_events:>15.2f} {avg_duration:>22.1f}")
