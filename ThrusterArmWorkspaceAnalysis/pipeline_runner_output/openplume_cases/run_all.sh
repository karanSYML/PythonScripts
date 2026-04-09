#!/usr/bin/env bash
# Mesh all OpenPlume cases with Gmsh
# Usage: bash run_all.sh

set -e
GMSH=$(command -v gmsh || echo gmsh)

echo "Meshing case_0000.geo..."
"${GMSH}" case_0000.geo -3 -o case_0000.msh -format msh2

echo "Meshing case_0001.geo..."
"${GMSH}" case_0001.geo -3 -o case_0001.msh -format msh2

echo "Meshing case_0002.geo..."
"${GMSH}" case_0002.geo -3 -o case_0002.msh -format msh2

echo "Meshing case_0003.geo..."
"${GMSH}" case_0003.geo -3 -o case_0003.msh -format msh2

echo "Meshing case_0004.geo..."
"${GMSH}" case_0004.geo -3 -o case_0004.msh -format msh2

echo "Done. Generated *.msh files:"
ls -lh *.msh 2>/dev/null || echo "(none found)"
