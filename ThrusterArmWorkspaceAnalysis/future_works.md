# Future Works — Repository Streamlining & Technical Improvements

This document captures the proposed plan for restructuring the
ThrusterArmWorkspaceAnalysis repository and the technical improvements
identified during the audit.

---

## Current state of the repo

- **20+ Python files at the top level**, mixing core libraries, CLI runners,
  visualization tools, and test scripts with no separation.
- **No packaging metadata** — no `pyproject.toml`, `requirements.txt`,
  `setup.py`, or `.gitignore`.
- **Generated artifacts checked into git**: `workspace_erosion.html` (8 MB),
  `workspace_erosion.png` (1.2 MB), `geometry_verification.png` (544 KB),
  plus `__pycache__/` and `.DS_Store`.
- **Outputs scattered** across multiple folders at the repo root:
  `pareto_output/`, `pipeline_runner_output/`, `propellant_correlation_results/`,
  `urdf_output/`, `openplume_cases/`, `plumePipeline/`, `propellantCorrelation/`.
- **Two monolithic files**:
  - `plume_impingement_pipeline.py` — 2490 LOC, 7+ classes mixing geometry,
    collision, erosion, orchestration, and plotting.
  - `propellant_erosion_correlation.py` — 1226 LOC.
- **`sputter_erosion/` is the only well-structured subpackage** — but it is
  pulled in via a `sys.path.insert` hack at the top of
  `plume_impingement_pipeline.py`.
- **Tests use `if __name__ == "__main__"` + matplotlib visual inspection**, no
  `pytest`, no assertions.

---

## Proposed restructured layout

```
ThrusterArmWorkspaceAnalysis/
├── pyproject.toml              # NEW: deps, package metadata, ruff/pytest config
├── .gitignore                  # NEW: __pycache__, *.png, *.html, outputs/, .DS_Store
├── README.md
├── src/tawa/
│   ├── kinematics.py           # arm_kinematics.py
│   ├── dynamics.py             # arm_dynamics.py
│   ├── trajectory.py           # arm_trajectory.py
│   ├── mass_model.py           # composite_mass_model.py
│   ├── geometry/               # split from plume_impingement_pipeline.py
│   │   ├── arm.py              # RoboticArmGeometry, ArmGeometry
│   │   ├── stack.py            # StackConfig, ThrusterParams
│   │   ├── collision.py        # AABB/OBB/disc intersection helpers
│   │   └── engine.py           # GeometryEngine
│   ├── plume/
│   │   ├── case_matrix.py      # CaseMatrixGenerator
│   │   ├── screening.py        # ErosionEstimator (analytical)
│   │   ├── pipeline.py         # PlumePipeline orchestrator
│   │   └── correlation.py      # propellant_erosion_correlation.py
│   ├── feasibility/
│   │   ├── cells.py
│   │   └── map.py
│   ├── urdf/
│   │   ├── generator.py        # urdf_generator.py
│   │   └── exporter.py         # generate_arm_urdf.py
│   ├── sputter/                # move sputter_erosion/ here, drop sys.path hack
│   └── viz/
│       ├── geometry_viewer.py  # geometry_visualizer.py
│       ├── workspace_erosion.py
│       └── plots.py            # generate_heatmaps, generate_status_map
├── scripts/                    # thin __main__ entry points
│   ├── run_pipeline.py
│   ├── run_propellant_correlation.py
│   ├── run_pareto.py
│   ├── make_video.py
│   └── export_openplume.py
├── tests/                      # pytest, with assertions
│   ├── conftest.py
│   ├── test_kinematics.py
│   ├── test_geometry.py
│   ├── test_sputter_integration.py
│   └── test_workspace_hifi.py
├── configs/
│   ├── feasibility_inputs.json
│   └── stowed_config.json
├── data/                       # inputs (gitignored if large)
│   └── pdsk_6month_maneuverPlan.xlsx   # consider Git LFS or external storage
├── outputs/                    # gitignored, all runs land here
└── docs/
    ├── TUTORIAL.md
    ├── CHANGELOG.md
    ├── ThrusterArmAnalysis.md
    └── specs/
        ├── feasibility_map_spec.md
        └── thruster_arm_urdf_spec.md
```

---

## Phased execution plan

| Phase | Scope | Risk |
|---|---|---|
| **1. Hygiene** | Add `.gitignore`, remove `__pycache__`/`.DS_Store`, move generated `.html`/`.png` out, decide on `.xlsx` (LFS or out) | Low |
| **2. Packaging** | Add `pyproject.toml` with deps (numpy, pinocchio, plotly, matplotlib, openpyxl, …), make `tawa` an installable package, drop the `sys.path.insert` hack in favor of a proper import | Low |
| **3. Move + namespace** | Relocate files into `src/tawa/`; update imports. One commit per subpackage so diffs stay reviewable | Medium |
| **4. Split monoliths** | Break `plume_impingement_pipeline.py` along the 7-class boundary; same for `propellant_erosion_correlation.py` | Medium |
| **5. Tests** | Convert `test_*.py` to pytest with real assertions; keep the matplotlib viz scripts as separate `scripts/visual_check_*.py` | Medium |
| **6. Outputs** | Centralize on `outputs/<run_name>/…`; add a `--output-dir` flag everywhere; gitignore | Low |

---

## Technical improvements

1. **`sys.path.insert` for `sputter_erosion`** (`plume_impingement_pipeline.py:38`)
   — fragile; breaks when imported from another working directory. Fix by
   making `sputter_erosion` part of the installable package.
2. **Dataclass + JSON config drift** — `feasibility_inputs.json` is loaded by
   hand; no schema validation. A `pydantic`/`dataclasses-json` model would
   catch typos and document the contract.
3. **Test assertions are missing** — `test_sputter_integration.py` and
   `test_workspace_hifi.py` "pass" by producing plots. Add numerical
   regression checks (golden values with tolerance) so silent breakage is
   caught.
4. **No CI** — even a minimal GitHub Actions workflow running `pytest` +
   `ruff check` would prevent regressions.
5. **Vectorization opportunities** — `feasibility_cells.py` does per-cell
   Python loops over a 4D joint grid; the kinematics + collision checks are
   good candidates for numpy broadcasting (likely 10–100× speedup).
6. **Caching for expensive runs** — `PlumePipeline` regenerates case geometry
   from scratch on every invocation; a `@functools.cache` on geometry by
   `(joint_config, stack_config)` hash, or a parquet cache of case matrix
   results, would help iteration.
7. **Logging instead of `print`** — most modules use bare `print`; switching
   to `logging` lets you turn verbosity up/down without editing source.
8. **Type hints + `mypy --strict` on `src/tawa/`** — the math-heavy modules
   already have annotations on most public functions; tightening this catches
   unit/shape mistakes early.
9. **Plotly HTML outputs** — `workspace_erosion.html` at 8 MB suggests the
   figure embeds raw point clouds. Downsampling or
   `plotly.io.write_html(..., include_plotlyjs="cdn")` cuts file size
   dramatically.
10. **`make_video.py` + `urdf_check.py`** — the latter is an 8-line stub;
    either finish it or delete. The former duplicates animation logic that
    exists in `geometry_visualizer.py`.

---

## Open questions

1. Is **`pdsk_6month_maneuverPlan.xlsx`** (7.5 MB) actual mission input that
   needs to be versioned, or can it move to external storage / Git LFS?
2. Is **`sputter_erosion/`** intended to be reusable outside this repo
   (separate package) or always vendored here?
3. Package name — `tawa`, `thruster_arm`, `tawa_core`, or something else?
4. One big PR vs. phased PRs (hygiene → packaging → moves → splits → tests)?
   Phased is safer for review but more overhead.
