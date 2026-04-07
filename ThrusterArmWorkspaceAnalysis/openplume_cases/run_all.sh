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

echo "Meshing case_0005.geo..."
"${GMSH}" case_0005.geo -3 -o case_0005.msh -format msh2

echo "Meshing case_0006.geo..."
"${GMSH}" case_0006.geo -3 -o case_0006.msh -format msh2

echo "Meshing case_0007.geo..."
"${GMSH}" case_0007.geo -3 -o case_0007.msh -format msh2

echo "Meshing case_0008.geo..."
"${GMSH}" case_0008.geo -3 -o case_0008.msh -format msh2

echo "Meshing case_0009.geo..."
"${GMSH}" case_0009.geo -3 -o case_0009.msh -format msh2

echo "Meshing case_0010.geo..."
"${GMSH}" case_0010.geo -3 -o case_0010.msh -format msh2

echo "Meshing case_0011.geo..."
"${GMSH}" case_0011.geo -3 -o case_0011.msh -format msh2

echo "Meshing case_0012.geo..."
"${GMSH}" case_0012.geo -3 -o case_0012.msh -format msh2

echo "Meshing case_0013.geo..."
"${GMSH}" case_0013.geo -3 -o case_0013.msh -format msh2

echo "Meshing case_0014.geo..."
"${GMSH}" case_0014.geo -3 -o case_0014.msh -format msh2

echo "Meshing case_0015.geo..."
"${GMSH}" case_0015.geo -3 -o case_0015.msh -format msh2

echo "Meshing case_0016.geo..."
"${GMSH}" case_0016.geo -3 -o case_0016.msh -format msh2

echo "Meshing case_0017.geo..."
"${GMSH}" case_0017.geo -3 -o case_0017.msh -format msh2

echo "Meshing case_0018.geo..."
"${GMSH}" case_0018.geo -3 -o case_0018.msh -format msh2

echo "Meshing case_0019.geo..."
"${GMSH}" case_0019.geo -3 -o case_0019.msh -format msh2

echo "Meshing case_0020.geo..."
"${GMSH}" case_0020.geo -3 -o case_0020.msh -format msh2

echo "Meshing case_0021.geo..."
"${GMSH}" case_0021.geo -3 -o case_0021.msh -format msh2

echo "Meshing case_0022.geo..."
"${GMSH}" case_0022.geo -3 -o case_0022.msh -format msh2

echo "Meshing case_0023.geo..."
"${GMSH}" case_0023.geo -3 -o case_0023.msh -format msh2

echo "Meshing case_0024.geo..."
"${GMSH}" case_0024.geo -3 -o case_0024.msh -format msh2

echo "Meshing case_0025.geo..."
"${GMSH}" case_0025.geo -3 -o case_0025.msh -format msh2

echo "Meshing case_0026.geo..."
"${GMSH}" case_0026.geo -3 -o case_0026.msh -format msh2

echo "Meshing case_0027.geo..."
"${GMSH}" case_0027.geo -3 -o case_0027.msh -format msh2

echo "Meshing case_0028.geo..."
"${GMSH}" case_0028.geo -3 -o case_0028.msh -format msh2

echo "Meshing case_0029.geo..."
"${GMSH}" case_0029.geo -3 -o case_0029.msh -format msh2

echo "Meshing case_0030.geo..."
"${GMSH}" case_0030.geo -3 -o case_0030.msh -format msh2

echo "Meshing case_0031.geo..."
"${GMSH}" case_0031.geo -3 -o case_0031.msh -format msh2

echo "Meshing case_0032.geo..."
"${GMSH}" case_0032.geo -3 -o case_0032.msh -format msh2

echo "Meshing case_0033.geo..."
"${GMSH}" case_0033.geo -3 -o case_0033.msh -format msh2

echo "Meshing case_0034.geo..."
"${GMSH}" case_0034.geo -3 -o case_0034.msh -format msh2

echo "Meshing case_0035.geo..."
"${GMSH}" case_0035.geo -3 -o case_0035.msh -format msh2

echo "Meshing case_0036.geo..."
"${GMSH}" case_0036.geo -3 -o case_0036.msh -format msh2

echo "Meshing case_0037.geo..."
"${GMSH}" case_0037.geo -3 -o case_0037.msh -format msh2

echo "Meshing case_0038.geo..."
"${GMSH}" case_0038.geo -3 -o case_0038.msh -format msh2

echo "Meshing case_0039.geo..."
"${GMSH}" case_0039.geo -3 -o case_0039.msh -format msh2

echo "Meshing case_0040.geo..."
"${GMSH}" case_0040.geo -3 -o case_0040.msh -format msh2

echo "Meshing case_0041.geo..."
"${GMSH}" case_0041.geo -3 -o case_0041.msh -format msh2

echo "Meshing case_0042.geo..."
"${GMSH}" case_0042.geo -3 -o case_0042.msh -format msh2

echo "Meshing case_0043.geo..."
"${GMSH}" case_0043.geo -3 -o case_0043.msh -format msh2

echo "Meshing case_0044.geo..."
"${GMSH}" case_0044.geo -3 -o case_0044.msh -format msh2

echo "Meshing case_0045.geo..."
"${GMSH}" case_0045.geo -3 -o case_0045.msh -format msh2

echo "Meshing case_0046.geo..."
"${GMSH}" case_0046.geo -3 -o case_0046.msh -format msh2

echo "Meshing case_0047.geo..."
"${GMSH}" case_0047.geo -3 -o case_0047.msh -format msh2

echo "Meshing case_0048.geo..."
"${GMSH}" case_0048.geo -3 -o case_0048.msh -format msh2

echo "Meshing case_0049.geo..."
"${GMSH}" case_0049.geo -3 -o case_0049.msh -format msh2

echo "Meshing case_0050.geo..."
"${GMSH}" case_0050.geo -3 -o case_0050.msh -format msh2

echo "Meshing case_0051.geo..."
"${GMSH}" case_0051.geo -3 -o case_0051.msh -format msh2

echo "Meshing case_0052.geo..."
"${GMSH}" case_0052.geo -3 -o case_0052.msh -format msh2

echo "Meshing case_0053.geo..."
"${GMSH}" case_0053.geo -3 -o case_0053.msh -format msh2

echo "Meshing case_0054.geo..."
"${GMSH}" case_0054.geo -3 -o case_0054.msh -format msh2

echo "Meshing case_0055.geo..."
"${GMSH}" case_0055.geo -3 -o case_0055.msh -format msh2

echo "Meshing case_0056.geo..."
"${GMSH}" case_0056.geo -3 -o case_0056.msh -format msh2

echo "Meshing case_0057.geo..."
"${GMSH}" case_0057.geo -3 -o case_0057.msh -format msh2

echo "Meshing case_0058.geo..."
"${GMSH}" case_0058.geo -3 -o case_0058.msh -format msh2

echo "Meshing case_0059.geo..."
"${GMSH}" case_0059.geo -3 -o case_0059.msh -format msh2

echo "Meshing case_0060.geo..."
"${GMSH}" case_0060.geo -3 -o case_0060.msh -format msh2

echo "Meshing case_0061.geo..."
"${GMSH}" case_0061.geo -3 -o case_0061.msh -format msh2

echo "Meshing case_0062.geo..."
"${GMSH}" case_0062.geo -3 -o case_0062.msh -format msh2

echo "Meshing case_0063.geo..."
"${GMSH}" case_0063.geo -3 -o case_0063.msh -format msh2

echo "Done. Generated *.msh files:"
ls -lh *.msh 2>/dev/null || echo "(none found)"
