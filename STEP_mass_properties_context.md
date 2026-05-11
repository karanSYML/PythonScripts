# STEP Assembly Mass Properties — Project Context

**Purpose:** Feed this file into a new Claude session to resume work on mass, CoG, and MOI computation for the OrbitGuard assembly. It covers what was done, what failed, why, and what to do next.

---

## 1. Project Overview

**Goal:** Compute mass, centre of gravity (CoG), and moments of inertia (MOI) for the OrbitGuard 3.2.2 spacecraft assembly from a SolidWorks STEP export. Eventually track these properties across design iterations and compute statistical variation (Monte Carlo on density/dimensional tolerances).

**Assembly file:** `OrbitGuard_Assembly_3.2.2_Thruster.STEP`
- Created with SolidWorks 2026, exported as STEP AP214
- 152 MB, ~1.8M lines
- 67 unique parts, 320 total instances
- Preliminary total mass: ~45.2 kg (all-aluminium assumption)

**Output so far:**
- `step_mass_budget.py` — working script that computes part volumes and mass budget
- `OrbitGuard_Assembly_3.2.2_Thruster_mass_budget.csv` — full mass budget with material guesses
- `OrbitGuard_3.2.2_Mass_Budget_Assessment.docx` — engineering summary for mechanical team
- `generate_mass_budget_report.py` — generates the Word document

---

## 2. Environment

**Python:** 3.12 (via miniforge3)

**Key libraries installed:**
```
cadquery==2.7.0          # wraps OCP; needed to pull in cadquery-ocp
cadquery-ocp==7.8.1.1    # OpenCASCADE Python bindings (the actual geometry kernel)
python-docx==1.2.0       # Word document generation
numpy, scipy             # already present
```

**Install command:**
```bash
pip install cadquery python-docx
```

**Important:** `pythonocc-core` is NOT available via pip for Python 3.12 on Linux. Use `cadquery-ocp` instead — it exposes the same OpenCASCADE API under the `OCP` namespace (not `OCC`).

**Module namespace:** `from OCP.XCAFDoc import ...` not `from OCC.Core.XCAFDoc import ...`

---

## 3. Approach: Using XCAF to Read STEP Assemblies

STEP files store geometry but not material properties. The XCAF (eXtended CAF) framework in OpenCASCADE is the correct way to read STEP assemblies because it preserves:
- Part names (from PRODUCT entities)
- Assembly hierarchy (parent–child relationships)
- Instance transforms (location/placement of each instance)

**Do not use** `STEPControl_Reader` for assemblies — it flattens everything and loses names and structure.

**Use instead:**
```python
from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.TDocStd import TDocStd_Document
from OCP.TCollection import TCollection_ExtendedString
from OCP.XCAFDoc import XCAFDoc_DocumentTool

doc = TDocStd_Document(TCollection_ExtendedString("XmXCAF"))
reader = STEPCAFControl_Reader()
reader.SetNameMode(True)
reader.SetColorMode(False)
reader.SetLayerMode(False)
reader.ReadFile(path)
reader.Transfer(doc)
shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
```

**Load time:** ~60 seconds for the 152 MB file. This is unavoidable — do not re-load between runs; cache the doc object if running in a notebook or interactive session.

---

## 4. OCP API Pitfalls (Differences from Classic OCC)

These caused failures and cost significant debugging time.

### 4.1 Method naming: `_s` suffix for static methods

In OCP (cadquery-ocp), static methods on XCAF classes have an `_s` suffix. Instance methods do not.

```python
# WRONG — these do not exist as instance or unsuffixed methods:
shape_tool.GetComponents(label, seq)
shape_tool.IsAssembly(label)
shape_tool.GetShape(label)

# CORRECT:
shape_tool.GetComponents_s(label, seq)   # static
shape_tool.IsAssembly_s(label)           # static
shape_tool.GetShape_s(label)             # static
shape_tool.GetFreeShapes(seq)            # instance method — no _s
```

### 4.2 `GetFreeShapes` takes `TDF_LabelSequence`, not a Python list

```python
# WRONG:
free_labels = []
shape_tool.GetFreeShapes(free_labels)   # TypeError

# CORRECT:
from OCP.TDF import TDF_LabelSequence
free_labels = TDF_LabelSequence()
shape_tool.GetFreeShapes(free_labels)
for i in range(free_labels.Length()):
    label = free_labels.Value(i + 1)    # 1-indexed
```

Same applies to `GetComponents_s`, `GetSubShapes_s`, etc.

### 4.3 `GetReferredShape_s` modifies its second argument in-place

```python
ref = TDF_Label()                              # must pre-allocate an empty label
ok = shape_tool.GetReferredShape_s(comp, ref)  # ref is now populated
```

### 4.4 `TDF_Label` has no `HashCode` method in OCP

`TDF_Label.HashCode(upper)` exists in classic OCC but not in OCP. To get a document-wide unique key for a label:

```python
def label_key(lbl: TDF_Label) -> tuple:
    """Walk up to root collecting tags — gives a unique path tuple."""
    tags = []
    cur = lbl
    for _ in range(20):        # depth guard
        tags.append(cur.Tag())
        if cur.IsRoot():
            break
        cur = cur.Father()
    return tuple(reversed(tags))
```

Use this tuple as a dictionary key for deduplication across instances.

### 4.5 Calling methods on a null `TDF_Label` causes a segfault

```python
l = TDF_Label()   # null label
l.IsNull()        # SEGFAULT — do not do this
```

Always guard: only call methods on labels obtained from `GetFreeShapes`, `GetComponents_s`, etc., or after checking the return value of `GetReferredShape_s`.

### 4.6 `BRepGProp.VolumeProperties_s`, not `brepgprop_VolumeProperties`

```python
# WRONG (classic OCC style):
from OCP.BRepGProp import brepgprop_VolumeProperties   # ImportError

# CORRECT:
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps

props = GProp_GProps()
BRepGProp.VolumeProperties_s(shape, props, True)
volume_mm3 = props.Mass()              # returns mm³ when shape is in mm
cog = props.CentreOfMass()            # gp_Pnt in mm — for future CoG work
inertia = props.MatrixOfInertia()     # gp_Mat — for future MOI work
```

**One call gives volume, CoG, and inertia simultaneously** — do not call it three times.

### 4.7 NAUO naming problem

SolidWorks STEP exports name component instance labels "NAUO1", "NAUO2", etc. (Next Assembly Usage Occurrence). The actual part name (from the PRODUCT entity) lives on the *referred* label, not the component label.

```python
ref = TDF_Label()
shape_tool.GetReferredShape_s(comp_label, ref)
name = get_label_name(ref)      # ← this is the PRODUCT name
# NOT:
name = get_label_name(comp_label)   # ← this gives "NAUO14" etc.
```

### 4.8 Duplicated part names in SolidWorks AP214 exports

SolidWorks appends the part name to itself: `Z-_Panel_-(2)_Z-_Panel_-(2)`. Clean with:

```python
def clean_name(raw: str) -> str:
    if "_" in raw:
        mid = raw.find("_")
        left, right = raw[:mid], raw[mid + 1:]
        if left == right:
            return left
    return raw
```

This handles simple cases. Longer duplications (e.g. four-fold repetition in some fastener names) require a more robust deduplication.

---

## 5. Assembly Traversal Logic

The structure in XCAF for a SolidWorks assembly:

```
Free shape (top-level assembly label)
  └─ Component labels (NAUO1, NAUO2, ...) — IsReference_s = True
       └─ Referred labels (actual part/sub-assembly definitions)
            ├─ IsSimpleShape_s = True  →  leaf part, has geometry
            └─ IsAssembly_s = True     →  sub-assembly, recurse into GetComponents_s
```

Working traversal pattern (from `step_mass_budget.py`):

```python
def visit(label):
    if shape_tool.IsReference_s(label):
        ref = TDF_Label()
        shape_tool.GetReferredShape_s(label, ref)
        if shape_tool.IsAssembly_s(ref):
            descend(ref)
        elif shape_tool.IsSimpleShape_s(ref):
            record(label, ref)   # label=NAUO, ref=part definition
    elif shape_tool.IsAssembly_s(label):
        descend(label)
    elif shape_tool.IsSimpleShape_s(label):
        record(label, label)

def descend(assy_label):
    seq = TDF_LabelSequence()
    shape_tool.GetComponents_s(assy_label, seq)
    for i in range(seq.Length()):
        visit(seq.Value(i + 1))
```

---

## 6. Current Mass Budget Results

| Category | Mass (g) | % |
|---|---|---|
| Structural panels (6× + shear) | 33,882 | 75.0% |
| Propulsion (2× BHT-350 T1) | 4,240 | 9.4% |
| Brackets and structure | 3,965 | 8.8% |
| Electronics / RFIM | 752 | 1.7% |
| Fasteners | 116 | 0.3% |
| Other COTS | 230 | 0.5% |
| **TOTAL** | **45,185** | |

**Known issues with current numbers:**
- Panels assumed solid Al (2700 kg/m³). If CFRP sandwich, total drops ~8–12 kg.
- BHT-350 T1 thruster geometry may be a simplified placeholder — verify against Busek datasheet.
- RS-00000821, RS-00000873, RS-00000450 not resolved — default Al assumed.
- All COTS-000xxx assigned default Al density.

---

## 7. Extending to CoG and MOI

### 7.1 What `GProp_GProps` already provides

`BRepGProp.VolumeProperties_s` already computes all three in one shot:

```python
props = GProp_GProps()
BRepGProp.VolumeProperties_s(shape, props, True)

volume_mm3  = props.Mass()
cog_local   = props.CentreOfMass()           # gp_Pnt — in part's local frame
inertia_mat = props.MatrixOfInertia()        # gp_Mat — about CoG, local frame
```

### 7.2 Transforming CoG into assembly frame

Each part instance has a placement (location) in the assembly. To get CoG in the assembly frame:

```python
from OCP.BRep import BRep_Builder
from OCP.TopLoc import TopLoc_Location

# Get the shape with its assembly-level transform already applied:
shape_in_assembly = shape_tool.GetShape_s(comp_label)   # comp_label = NAUO label
# VolumeProperties on this already-placed shape gives CoG in assembly frame
props = GProp_GProps()
BRepGProp.VolumeProperties_s(shape_in_assembly, props, True)
cog_assembly_frame = props.CentreOfMass()
```

**Key insight:** `GetShape_s(ref_label)` returns the shape in the part's own local frame. `GetShape_s(comp_label)` (using the component/NAUO label, not the referred label) returns the shape with its assembly transform applied. Use the comp_label version for CoG and MOI in the assembly frame.

The current `step_mass_budget.py` uses `ref_label` for volume (correct, because volume is frame-independent). CoG/MOI work needs to use `comp_label` to pick up the placement.

### 7.3 Assembly-level CoG (weighted average)

```python
total_mass = sum(part_mass_i)
cog_x = sum(mass_i * cog_i.X()) / total_mass
cog_y = sum(mass_i * cog_i.Y()) / total_mass
cog_z = sum(mass_i * cog_i.Z()) / total_mass
```

Note: if the same part definition appears multiple times (e.g. BHT-350 at two locations), each *instance* must be processed separately using its own comp_label to get the correct assembly-frame placement. The current deduplication logic in the mass script records only one shape per unique part — this must be changed for CoG/MOI: iterate over all instances, not unique definitions.

### 7.4 Assembly-level MOI (parallel axis theorem)

For each part instance:
```
I_assembly += I_cog_i + m_i * [d²_i]
```
where `[d²_i]` is the parallel axis correction matrix (distance from part CoG to assembly CoG).

`GProp_GProps` can also combine results from multiple shapes using `Add()`:

```python
total_props = GProp_GProps()
for shape_instance in all_instances:
    part_props = GProp_GProps()
    BRepGProp.VolumeProperties_s(shape_instance, part_props, True)
    # Scale by density to convert volume-weighted to mass-weighted:
    # OCC VolumeProperties treats "mass" as volume — need to scale
    # The simplest approach is to set a density-scaled point of inertia system
    # using GProp_GProps(gp_Pnt, density) constructor — see OCC docs
    total_props.Add(part_props, density)   # density scales the contribution
```

This is the cleanest path — let OCC handle the parallel axis theorem internally.

### 7.5 MOI frame conventions

Be explicit about which frame MOI is reported in:
- **Body frame** (assembly origin, assembly axes) — most useful for ACS/dynamics
- **Principal axes** — diagonalises the inertia tensor; eigenvalue decomposition
- **CoG frame** (parallel to body frame but origin at CoG) — standard in spacecraft dynamics

OCC returns inertia in the coordinate system passed to `VolumeProperties_s`. To get principal axes, diagonalise the 3×3 inertia matrix with `numpy.linalg.eigh`.

---

## 8. Variation / Monte Carlo Analysis

Once deterministic mass/CoG/MOI work, the natural extension is uncertainty propagation.

### 8.1 Density uncertainty

For each part, assign a density distribution (e.g. Al 7075: 2810 ± 10 kg/m³). Monte Carlo:

```python
import numpy as np

N = 1000
total_masses = []
for _ in range(N):
    total = 0
    for part in parts:
        rho = np.random.normal(part['density_mean'], part['density_std'])
        total += part['volume_m3'] * rho * 1000   # grams
    total_masses.append(total)

# Report: mean ± 3σ
```

### 8.2 COTS mass uncertainty

COTS items should be modelled with a mass distribution, not a computed volume. Replace the geometric computation with:

```python
cots_db = {
    'BHT-350 T1': {'mass_mean': 2100, 'mass_std': 50},   # g, from datasheet ± margin
    ...
}
```

### 8.3 Dimensional tolerances (advanced)

For parts where dimensional tolerances affect mass meaningfully (e.g. thick panels with ±0.2 mm tolerance), the volume must be perturbed. This requires either:
- Parametric CAD (not feasible from a static STEP file)
- Volume sensitivity: `∂V/∂t` estimated by re-scaling the shape or using analytic geometry for simple prismatic parts

This is only worth pursuing if the panel thickness tolerance contribution to mass uncertainty is significant compared to density uncertainty — check this first with a simple sensitivity calculation.

---

## 9. Limitations and Known Issues

| Issue | Impact | Resolution |
|---|---|---|
| STEP load time ~60s | High — slows iteration | Cache the document; avoid re-loading between parameter sweeps |
| Material not in STEP | High — all densities guessed | Assign materials in SolidWorks; use JSON override file short-term |
| Defeatured/simplified COTS geometry | High for COTS mass | Replace computed volumes with datasheet masses in override JSON |
| CoG requires comp_label, not ref_label | Medium — current script uses ref_label | Change instance loop for CoG/MOI work |
| AP214 duplicate name export | Low — cosmetic | Fix in SolidWorks export settings or use AP242 |
| Null label segfault | Low — only triggered by bad traversal | Existing traversal avoids it; guard any new label operations |
| `TDF_Label.Tag()` not unique across document | Low — fixed with path tuple | `label_key()` function handles this |

---

## 10. File Map

```
PythonScripts/
├── step_mass_budget.py                          # main analysis script
├── generate_mass_budget_report.py               # generates the Word report
├── OrbitGuard_Assembly_3.2.2_Thruster.STEP      # source CAD (152 MB)
├── OrbitGuard_Assembly_3.2.2_Thruster_mass_budget.csv   # output: full BOM with masses
├── OrbitGuard_3.2.2_Mass_Budget_Assessment.docx         # output: engineering report
└── STEP_mass_properties_context.md              # this file
```

**To re-run mass budget:**
```bash
python3 step_mass_budget.py OrbitGuard_Assembly_3.2.2_Thruster.STEP
```

**To apply material overrides:**
```bash
# Create overrides.json:
# { "BHT-350 T1 - BUSEK PROPRIETARY(2)": 4200, "Z-_Panel_-(2)_Z-_Panel_-(2)": 900 }
python3 step_mass_budget.py --materials overrides.json
```

**To regenerate the Word report:**
```bash
python3 generate_mass_budget_report.py
```
