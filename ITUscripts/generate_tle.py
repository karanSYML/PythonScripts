#!/usr/bin/env python3
"""
Author : karan.anand@infiniteorbits.io

Generate a synthetic TLE for OG3 mission at 13°E (assumed) GEO slot.

Reference: HOTBIRD 13F (NORAD 54048) at 13°E
- Inclination: ~0.07°
- Eccentricity: ~0.0002 (near-circular GEO)
- Mean motion: ~1.00274 rev/day (GEO)
- GEO longitude: 13°E

For a synthetic/future mission we can use:
- Dummy NORAD catalog number: 99999
- International designator: 26001A (first launch of 2026)
- Epoch: 2026 day 180 (June 29, 2026) - plausible start of inspection phase
- Near-zero drag (B* = 0) for GEO
- Zero mean motion derivatives

"""

import math

def compute_checksum(line):
    """Compute TLE checksum (mod 10 of sum of digits, with '-' counting as 1)."""
    s = 0
    for c in line[:68]:
        if c.isdigit():
            s += int(c)
        elif c == '-':
            s += 1
    return s % 10

# === OG3 MISSION PARAMETERS ===

# Catalog / ID
norad_id = 99999 # Hypothetical
classification = 'U'
intl_desig_year = '26'
intl_desig_launch = '001'
intl_desig_piece = 'A  '

# Epoch: 2026, day 180.50000000 (June 29, 2026, 12:00:00 UTC)
epoch_year = 26
epoch_day = 180.50000000

# Derivatives and drag (all zero for synthetic GEO)
ndot = 0.0          # first derivative of mean motion / 2
nddot = 0.0         # second derivative of mean motion / 6
bstar = 0.0         # B* drag term

# Orbital elements (based on HOTBIRD 13F at 13°E)
inclination = 0.0700         # degrees (realistic small GEO inclination)
raan = 75.0000               # degrees (RAAN - realistic for 13°E slot)
eccentricity = 0.0002000     # near-circular
arg_perigee = 0.0000         # degrees
mean_anomaly = 280.0000      # degrees - adjusted to place satellite near 13°E

# Mean motion for GEO
# GEO semi-major axis: 42164.17 km
# Earth mu = 398600.4418 km^3/s^2
# Period = 2*pi*sqrt(a^3/mu)
a_geo = 42164.17  # km
mu = 398600.4418  # km^3/s^2
period_s = 2 * math.pi * math.sqrt(a_geo**3 / mu)
mean_motion = 86400.0 / period_s  # rev/day

ephemeris_type = 0
element_set_number = 999
rev_number = 100  # low rev number for a new mission

# === FORMAT LINE 1 ===
# Columns (1-indexed):
# 01     Line number
# 03-07  NORAD catalog number
# 08     Classification
# 10-11  Intl designator year
# 12-14  Intl designator launch number
# 15-17  Intl designator piece
# 19-20  Epoch year
# 21-32  Epoch day
# 34-43  ndot (1st deriv mean motion / 2)
# 45-52  nddot (2nd deriv mean motion / 6)
# 54-61  BSTAR
# 63     Ephemeris type
# 65-68  Element set number
# 69     Checksum

def format_exp(value):
    """Format a value in TLE exponential notation: ±DDDDD±E (decimal point assumed)."""
    if value == 0.0:
        return ' 00000-0'
    sign = '-' if value < 0 else ' '
    val = abs(value)
    exp = int(math.floor(math.log10(val)))
    mantissa = val / (10 ** exp)
    # mantissa is now [1, 10), we need it as 5-digit integer with implied leading decimal
    mantissa_int = int(round(mantissa * 10000))
    exp_val = exp + 1  # because we shifted the decimal
    exp_sign = '+' if exp_val >= 0 else '-'
    return f'{sign}{mantissa_int:05d}{exp_sign}{abs(exp_val)}'

# ndot formatting: ±.DDDDDDDD (10 chars, cols 34-43)
ndot_str = f'{ndot:+011.8f}'.replace('+', ' ')  # " .00000000" format
if ndot >= 0:
    ndot_str = f' .{abs(ndot):.8f}'[0:10]
else:
    ndot_str = f'-.{abs(ndot):.8f}'[0:10]
# Simpler: for zero
ndot_str = ' .00000000'

nddot_str = ' 00000-0'
bstar_str = ' 00000-0'

line1 = f'1 {norad_id:05d}{classification} {intl_desig_year}{intl_desig_launch}{intl_desig_piece} {epoch_year:02d}{epoch_day:012.8f} {ndot_str} {nddot_str} {bstar_str} {ephemeris_type} {element_set_number:4d}'

# Ensure exactly 68 chars before checksum
line1 = line1[:68]
# Pad if needed
line1 = line1.ljust(68)
cs1 = compute_checksum(line1)
line1 = line1 + str(cs1)

# === FORMAT LINE 2 ===
# 01     Line number
# 03-07  NORAD catalog number
# 09-16  Inclination (degrees, 8 chars)
# 18-25  RAAN (degrees, 8 chars)
# 27-33  Eccentricity (decimal point assumed, 7 chars)
# 35-42  Argument of perigee (degrees, 8 chars)
# 44-51  Mean anomaly (degrees, 8 chars)
# 53-63  Mean motion (rev/day, 11 chars)
# 64-68  Rev number at epoch (5 chars)
# 69     Checksum

line2 = f'2 {norad_id:05d} {inclination:8.4f} {raan:8.4f} {int(eccentricity * 10000000):07d} {arg_perigee:8.4f} {mean_anomaly:8.4f} {mean_motion:11.8f}{rev_number:5d}'

line2 = line2[:68]
line2 = line2.ljust(68)
cs2 = compute_checksum(line2)
line2 = line2 + str(cs2)

# Output
title = 'OG3'
print(f'{title}')
print(f'{line1}')
print(f'{line2}')
print()
print(f'Line 1 length: {len(line1)}')
print(f'Line 2 length: {len(line2)}')
print()

# Verification
print('=== VERIFICATION ===')
print(f'NORAD ID:      {norad_id}')
print(f'Epoch:         2026 day {epoch_day} (June 29, 2026 12:00 UTC)')
print(f'Inclination:   {inclination}°')
print(f'RAAN:          {raan}°')
print(f'Eccentricity:  {eccentricity}')
print(f'Arg Perigee:   {arg_perigee}°')
print(f'Mean Anomaly:  {mean_anomaly}°')
print(f'Mean Motion:   {mean_motion:.8f} rev/day')
print(f'GEO Period:    {period_s:.2f} s = {period_s/3600:.4f} h')
print(f'Semi-major a:  {a_geo} km')
print(f'Altitude:      {a_geo - 6378.137:.2f} km')

# =========== Optional =========== #
# # Also generate a TLE for the TARGET in graveyard orbit (GEO + 300km)
# print('\n\n=== TARGET SATELLITE (GRAVEYARD ORBIT: GEO + 300 km) ===')
# a_grave = a_geo + 300  # km
# period_grave = 2 * math.pi * math.sqrt(a_grave**3 / mu)
# mm_grave = 86400.0 / period_grave

# # Graveyard sats often have higher inclination (no more station-keeping)
# incl_target = 5.5000  # typical for aging GEO sat without N-S stationkeeping
# raan_target = 72.0000
# ecc_target = 0.0005000
# argp_target = 15.0000
# ma_target = 265.0000

# line1t = f'1 99998U 10025A   {epoch_year:02d}{epoch_day:012.8f}  .00000000  00000-0  00000-0 0  999'
# line1t = line1t[:68].ljust(68)
# cs1t = compute_checksum(line1t)
# line1t += str(cs1t)

# line2t = f'2 99998 {incl_target:8.4f} {raan_target:8.4f} {int(ecc_target * 10000000):07d} {argp_target:8.4f} {ma_target:8.4f} {mm_grave:11.8f}  100'
# line2t = line2t[:68].ljust(68)
# cs2t = compute_checksum(line2t)
# line2t += str(cs2t)

# print('TARGET-SAT')
# print(line1t)
# print(line2t)
# print()
# print(f'Target Semi-major axis: {a_grave} km')
# print(f'Target Altitude:        {a_grave - 6378.137:.2f} km')
# print(f'Target Mean Motion:     {mm_grave:.8f} rev/day')
# print(f'Target Period:          {period_grave/3600:.4f} h')
# print(f'Target Inclination:     {incl_target}° (drifted - no N-S stationkeeping)')

