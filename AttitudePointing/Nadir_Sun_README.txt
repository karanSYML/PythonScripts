========================================================================
  AOCS → THERMAL DATA HANDOFF
  Attitude law: Nadir+Sun
  CCSDS OEM + AEM for Systema-Thermica Import
========================================================================

Generated:        2026-04-24 21:52:50 UTC
Object:           RDV_SAT
Originator:       AOCS_SIM
Source files:     dataPackage4Thermal_RDV.csv, dataPackage4Thermal_INS.csv

Time span:        2028-09-01T03:49:53.000000  to
                  2028-09-24T02:49:53.000243
Data points:      6621
Time step:        5 minutes (300 s)

------------------------------------------------------------------------
  FRAME DEFINITIONS
------------------------------------------------------------------------

  Reference frame:  EME2000 (Earth-centered inertial, J2000)
  Body frame:       SC_BODY_1 (= MRF, Mission Reference Frame)
  Attitude dir:     A2B (rotation from EME2000 to SC_BODY_1)

  Quaternion convention:
      Source CSV:  scalar-first  (q_a, q_i, q_j, q_k)
      AEM file:   scalar-last    (Q1, Q2, Q3, QC) per CCSDS 504.0-B
      Signs PRESERVED from AOCS source (195 hemisphere flips present).
      All quaternions are unit-normalized (||q|| = 1).

------------------------------------------------------------------------
  SYSTEMA-THERMICA IMPORT
------------------------------------------------------------------------

  1. Trajectory tab  → import .oem (center=EARTH, frame=EME2000)
  2. Kinematics tab  → import .aem (A2B, frame_A=EME2000, frame_B=SC_BODY_1)
  3. Mission tab     → associate trajectory + kinematics with geometry
  4. Validate        → animate 3D view, check Sun angles vs source CSV

