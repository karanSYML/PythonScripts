// ==================================================================
// OpenPlume Geometry — Case 0009
// Status   : CAUTION  |  Max erosion: 7.038 µm  (28.2% of Ag thickness)
//
// Thruster pos    : [3.00000, 0.00000, -1.55000] m
// Thrust direction: [-0.91657, 0.00000, 0.39988]
// Plume direction : [0.91657, -0.00000, -0.39988]
// Stack COG       : [0.00000, 0.00000, -0.24118] m
// Panel tracking  : 0.0°
// Servicer origin : [0.00000, 0.00000, -2.05000] m  (Z-, LAR docked)
//
// Coordinate system: X=velocity, Y=orbit-normal, Z=anti-earth
// Client bus centred at origin. Servicer below (Z−) via LAR.
//
// Mesh with: gmsh case_0009.geo -3 -o case_0009.msh
// ==================================================================

SetFactory("OpenCASCADE");

// ────────────────────────────────────────────────────────────────
// Characteristic mesh lengths
// ────────────────────────────────────────────────────────────────
lc_thruster  = 0.050;  // m — near thruster exit plane
lc_structure = 0.250;  // m — near spacecraft surfaces
lc_farfield  = 1.600;  // m — outer domain boundary

// ────────────────────────────────────────────────────────────────
// Client bus  2.500 × 2.200 × 3.000 m  (centred at origin)
// ────────────────────────────────────────────────────────────────
Box(1) = {-1.25000, -1.10000, -1.50000,  2.50000, 2.20000, 3.00000};

// ────────────────────────────────────────────────────────────────
// Servicer bus  1.000 × 1.000 × 1.000 m  (Z− docking via LAR)
// ────────────────────────────────────────────────────────────────
Box(2) = {-0.50000, -0.50000, -2.55000,  1.00000, 1.00000, 1.00000};

// ────────────────────────────────────────────────────────────────
// Solar panel +X  span=10.00 m, width=2.50 m, thickness=25 mm, track=0.0°
// ────────────────────────────────────────────────────────────────
// Panel before tracking rotation (hinge at X=1.2500, Z=1.5000)
Box(3) = {1.25000, -1.25000, 1.48750,  10.00000, 2.50000, 0.02500};

// ────────────────────────────────────────────────────────────────
// Solar panel −X  (symmetric, same tracking rotation)
// ────────────────────────────────────────────────────────────────
// Panel before tracking rotation (hinge at X=-1.2500, Z=1.5000)
Box(4) = {-11.25000, -1.25000, 1.48750,  10.00000, 2.50000, 0.02500};

// ────────────────────────────────────────────────────────────────
// Computational domain — sphere  R = 20.00 m
// ────────────────────────────────────────────────────────────────
// Encloses full panel span (11.2 m) plus 6 m margin
// To subtract spacecraft from domain (recommended before meshing):
// BooleanFragments{ Volume{1,2,3,4,100}; Delete; }{}  // conforming interfaces
Sphere(100) = {0, 0, 0, 20.00000};

// ────────────────────────────────────────────────────────────────
// Thruster source geometry
// ────────────────────────────────────────────────────────────────
// Thruster exit centre — OpenPlume plume source
Point(200) = {3.000000, 0.000000, -1.550000, lc_thruster};

// Plume-direction arrow tip (2 m along plume, visual reference)
// Direction: [0.91657, -0.00000, -0.39988]
Point(201) = {4.833139, 0.000000, -2.349752, lc_thruster};
Line(200) = {200, 201};  // plume-direction arrow

// Stack centre-of-gravity at mission start
Point(202) = {0.000000, 0.000000, -0.241176, lc_structure};

// Worst-case erosion panel point (off-axis 44.3°, d=8.64 m)
//   Max erosion = 7.038 µm at this panel point

// ────────────────────────────────────────────────────────────────
// Physical groups
// ────────────────────────────────────────────────────────────────
// Solid spacecraft structures — assign Wall BC in OpenPlume
Physical Volume("client_bus",     1) = {1};
Physical Volume("servicer_bus",   2) = {2};
Physical Volume("solar_panel_+X", 3) = {3};
Physical Volume("solar_panel_-X", 4) = {4};

// Fluid / simulation domain (modify after BooleanFragments if used)
Physical Volume("fluid_domain", 100) = {100};

// Source + reference
Physical Point("thruster_source",  200) = {200};
Physical Point("cog_marker",       202) = {202};
Physical Line("plume_direction",   200) = {200};

// ────────────────────────────────────────────────────────────────
// Mesh refinement — distance-based around thruster source
// ────────────────────────────────────────────────────────────────
// Field 1: distance from thruster exit point
Field[1] = Distance;
Field[1].PointsList = {200};

// Field 2: threshold — fine mesh near thruster, coarse at domain edge
Field[2] = Threshold;
Field[2].InField  = 1;
Field[2].DistMin  = 1.000;          // m  inner radius (fine)
Field[2].DistMax  = 12.000;    // m  outer radius (coarse)
Field[2].SizeMin  = lc_thruster;
Field[2].SizeMax  = lc_farfield;

// Field 3: distance from spacecraft volumes — surface refinement
Field[3] = Distance;
Field[3].VolumesList = {1, 2, 3, 4};

Field[4] = Threshold;
Field[4].InField  = 3;
Field[4].DistMin  = 0.000;
Field[4].DistMax  = 3.000;
Field[4].SizeMin  = lc_structure;
Field[4].SizeMax  = lc_farfield;

// Field 5: take minimum (finest mesh wins)
Field[5] = Min;
Field[5].FieldsList = {2, 4};

Background Field = 5;

// ────────────────────────────────────────────────────────────────
// Mesh settings
// ────────────────────────────────────────────────────────────────
Mesh.CharacteristicLengthMin = lc_thruster;
Mesh.CharacteristicLengthMax = lc_farfield;
Mesh.Algorithm3D             = 4;  // Frontal-Delaunay 3D
Mesh.Optimize                = 1;
