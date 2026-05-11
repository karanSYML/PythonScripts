"""
Mass budget calculator for STEP assembly files (SolidWorks exports).

Uses OCC/XCAF to load the assembly with names preserved, traverses the
hierarchy to find all unique parts and their instance counts, computes
solid volumes from B-rep geometry, then applies material densities to
produce a mass budget CSV.

Usage:
    python step_mass_budget.py [STEP_FILE] [--output OUTPUT_CSV] [--materials JSON]

Material density overrides can be supplied via JSON:
    { "PartName": 4430, "AnotherPart": 1600 }
"""

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cadquery as cq
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TCollection import TCollection_ExtendedString
from OCP.TDF import TDF_Label, TDF_LabelSequence
from OCP.TDataStd import TDataStd_Name
from OCP.TDocStd import TDocStd_Document
from OCP.XCAFDoc import XCAFDoc_DocumentTool

# ---------------------------------------------------------------------------
# Material density map (kg/m³).
# Keys are substrings matched case-insensitively against part names.
# First match wins — put more specific patterns first.
# ---------------------------------------------------------------------------
MATERIAL_DENSITY_MAP = {
    # Stainless steel fasteners
    "316 ss":       7900,
    "a4-70":        7900,
    "stainless":    7900,
    # Titanium
    "ti-":          4430,
    "titanium":     4430,
    # CFRP / composites
    "cfrp":         1600,
    "composite":    1600,
    # Flexible / solar array
    "flex":         1400,
    "solar":        1400,
    # PCBs / electronics (Al housing assumed for enclosures)
    "pcb":          1850,
    # General spacecraft-grade aluminium (most structural parts)
    "bracket":      2700,
    "panel":        2700,
    "shear":        2700,
    "washer":       7900,   # assume SS unless overridden
    "thruster":     2700,
    "star_tracker": 2700,
    "rfim":         2700,
    "umb":          2700,
    "cots":         2700,
    "al":           2700,
    "aluminium":    2700,
    "aluminum":     2700,
    # Fallback
    "_default_":    2700,
}


def get_density(part_name: str) -> tuple[float, str]:
    lower = part_name.lower()
    for key, density in MATERIAL_DENSITY_MAP.items():
        if key == "_default_":
            continue
        if key in lower:
            return density, key
    return MATERIAL_DENSITY_MAP["_default_"], "_default_ (Al assumed)"


def label_key(lbl: TDF_Label) -> tuple:
    """Unique path-based key for a TDF_Label (hashable, document-wide unique)."""
    tags = []
    cur = lbl
    for _ in range(20):   # max depth guard
        tags.append(cur.Tag())
        if cur.IsRoot():
            break
        cur = cur.Father()
    return tuple(reversed(tags))


def get_label_name(label: TDF_Label) -> str:
    name_attr = TDataStd_Name()
    if label.FindAttribute(TDataStd_Name.GetID_s(), name_attr):
        return name_attr.Get().ToExtString()
    return ""


def clean_name(raw: str) -> str:
    """SolidWorks STEP exports often duplicate: 'PartA_PartA' → 'PartA'."""
    if "_" in raw:
        mid = raw.find("_")
        left, right = raw[:mid], raw[mid + 1:]
        if left == right:
            return left
    return raw


def volume_mm3(shape) -> float:
    """Compute volume in mm³ from a TopoDS_Shape."""
    props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape, props, True)
    return abs(props.Mass())


def load_step(path: str):
    """Load a STEP file via STEPCAFControl_Reader; return (doc, shape_tool)."""
    doc = TDocStd_Document(TCollection_ExtendedString("XmXCAF"))
    reader = STEPCAFControl_Reader()
    reader.SetNameMode(True)
    reader.SetColorMode(False)
    reader.SetLayerMode(False)
    Interface_Static.SetIVal_s("read.stepcaf.subshapes.name", 1)

    print(f"Reading {path}")
    print("  (59s expected for 152MB file — please wait...)")
    t0 = time.time()
    if reader.ReadFile(path) != IFSelect_RetDone:
        raise RuntimeError("STEPCAFControl_Reader.ReadFile failed")
    reader.Transfer(doc)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    return doc, XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())


def collect_unique_parts(shape_tool):
    """
    Walk the assembly hierarchy and accumulate unique parts.

    Returns:
        parts: dict  label_hash → {"name": str, "shape": TopoDS_Shape}
        counts: dict label_hash → int (total instance count in assembly)
    """
    parts: dict[int, dict] = {}
    counts: dict[int, int] = defaultdict(int)

    def visit(label: TDF_Label):
        if shape_tool.IsReference_s(label):
            ref = TDF_Label()
            shape_tool.GetReferredShape_s(label, ref)

            if shape_tool.IsAssembly_s(ref):
                _descend(ref)
            elif shape_tool.IsSimpleShape_s(ref):
                _record(label, ref)

        elif shape_tool.IsAssembly_s(label):
            _descend(label)

        elif shape_tool.IsSimpleShape_s(label):
            _record(label, label)

    def _descend(assy_label: TDF_Label):
        comp_seq = TDF_LabelSequence()
        shape_tool.GetComponents_s(assy_label, comp_seq)
        for i in range(comp_seq.Length()):
            visit(comp_seq.Value(i + 1))

    def _record(comp_label: TDF_Label, ref_label: TDF_Label):
        h = label_key(ref_label)          # unique path tuple, document-wide
        counts[h] += 1
        if h not in parts:
            # ref_label holds the PRODUCT name; comp_label holds the NAUO instance name
            name = (get_label_name(ref_label)
                    or get_label_name(comp_label)
                    or f"Part_{h}")
            name = clean_name(name)
            shape = shape_tool.GetShape_s(ref_label)
            parts[h] = {"name": name, "shape": shape}

    # Start traversal from all free (top-level) shapes
    free = TDF_LabelSequence()
    shape_tool.GetFreeShapes(free)
    print(f"  Top-level shapes in document: {free.Length()}")

    for i in range(free.Length()):
        top = free.Value(i + 1)
        if shape_tool.IsReference_s(top):
            ref = TDF_Label()
            shape_tool.GetReferredShape_s(top, ref)
            if shape_tool.IsAssembly_s(ref):
                _descend(ref)
            else:
                visit(top)
        elif shape_tool.IsAssembly_s(top):
            _descend(top)
        else:
            _record(top, top)

    return parts, counts


def build_budget(parts, counts, extra_densities: dict) -> list[dict]:
    n = len(parts)
    print(f"\nComputing volumes for {n} unique parts...")
    rows = []
    for idx, (h, info) in enumerate(parts.items(), 1):
        name = info["name"]
        shape = info["shape"]
        qty = counts[h]

        if idx % 5 == 0 or idx == n:
            print(f"  [{idx:>3}/{n}] {name[:55]}")

        vol = volume_mm3(shape)          # mm³

        if name in extra_densities:
            density, material = extra_densities[name], "user-specified"
        else:
            density, material = get_density(name)

        mass_unit_g = vol * 1e-3 * density * 1e-3   # mm³ → cm³ → L → kg → g
        # simpler: vol_mm3 * density_kg_m3 * 1e-6 = mass_g
        mass_unit_g = vol * density * 1e-6
        total_g = mass_unit_g * qty

        rows.append({
            "Part Name":         name,
            "Qty":               qty,
            "Vol/unit (cm³)":    round(vol * 1e-3, 4),
            "Density (kg/m³)":   density,
            "Material guess":    material,
            "Mass/unit (g)":     round(mass_unit_g, 3),
            "Total mass (g)":    round(total_g, 3),
        })

    rows.sort(key=lambda r: r["Total mass (g)"], reverse=True)
    return rows


def write_csv(rows: list[dict], out_path: str) -> float:
    fields = [
        "Part Name", "Qty", "Vol/unit (cm³)", "Density (kg/m³)",
        "Material guess", "Mass/unit (g)", "Total mass (g)",
    ]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    total_g = sum(r["Total mass (g)"] for r in rows)
    total_vol = sum(r["Vol/unit (cm³)"] * r["Qty"] for r in rows)
    with open(out_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writerow({
            "Part Name":        "TOTAL",
            "Qty":              sum(r["Qty"] for r in rows),
            "Vol/unit (cm³)":   round(total_vol, 4),
            "Density (kg/m³)":  "",
            "Material guess":   "",
            "Mass/unit (g)":    "",
            "Total mass (g)":   round(total_g, 3),
        })
    return total_g


def main():
    p = argparse.ArgumentParser(description="STEP assembly mass budget")
    p.add_argument("step_file", nargs="?",
                   default="OrbitGuard_Assembly_3.2.2_Thruster.STEP")
    p.add_argument("--output", default=None)
    p.add_argument("--materials", default=None,
                   help="JSON file: {\"PartName\": density_kg_m3, ...}")
    args = p.parse_args()

    step_path = str(Path(args.step_file).resolve())
    out_path = args.output or str(Path(step_path).with_suffix("")) + "_mass_budget.csv"

    extra = {}
    if args.materials:
        with open(args.materials) as f:
            extra = json.load(f)

    doc, shape_tool = load_step(step_path)
    parts, counts = collect_unique_parts(shape_tool)

    print(f"  Unique parts:    {len(parts)}")
    print(f"  Total instances: {sum(counts.values())}")

    rows = build_budget(parts, counts, extra)
    total_g = write_csv(rows, out_path)

    bar = "=" * 62
    print(f"\n{bar}")
    print(f"  Output:       {out_path}")
    print(f"  Unique parts: {len(rows)}")
    print(f"  Total mass:   {total_g/1000:.3f} kg  ({total_g:.0f} g)")
    print(bar)

    print("\nTop 10 heaviest (by total mass):")
    print(f"  {'Part Name':<44} {'Qty':>4} {'g/unit':>9} {'Total g':>9}")
    print("  " + "-" * 70)
    for r in rows[:10]:
        print(f"  {r['Part Name']:<44} {r['Qty']:>4} "
              f"{r['Mass/unit (g)']:>9.1f} {r['Total mass (g)']:>9.1f}")

    print(f"\n  Tip: review the 'Material guess' column in the CSV and")
    print(f"  override wrong densities via --materials densities.json")


if __name__ == "__main__":
    main()
