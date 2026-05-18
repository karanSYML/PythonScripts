"""
rdv_phases.py
=============
Shared OG3 rendezvous sub-phase definitions and axis annotation helper.
Import this in any plotting script to get consistent phase shading.
"""

# (start_day, end_day, short_label, fill_color)
PHASES = [
    ( 0,  1,  "P1",  "#CBD5E1"),   # Station-keeping at −60 km
    ( 1, 11,  "P2",  "#BFDBFE"),   # Approach −60 → −30 km
    (11, 12,  "P3",  "#CBD5E1"),   # Station-keeping at −30 km
    (12, 16,  "P4",  "#FDE68A"),   # Accel. approach −30 → −1 km
    (16, 17,  "P5",  "#CBD5E1"),   # Station-keeping at −1 km
    (17, 19,  "P6",  "#FBCFE8"),   # Checkpoint resizing at −1 km
    (19, 20,  "P7",  "#CBD5E1"),   # Station-keeping at −1 km
    (20, 22,  "P8",  "#A7F3D0"),   # Fly-by −1 → +1 km
    (22, 25,  "P9",  "#CBD5E1"),   # Station-keeping at +1 km
]

PHASE_ALPHA = 0.18   # background band opacity
LABEL_COLOR = "#475569"


def shade_phases(ax, lo, hi, label_y=0.985, label_va="top", fontsize=7.0):
    """
    Shade mission sub-phase bands on a time-axis plot and add short labels.

    Parameters
    ----------
    ax       : matplotlib Axes with x-axis in mission elapsed days
    lo, hi   : x-range of the axes (days)
    label_y  : y position of labels in axes-fraction coordinates
    label_va : vertical alignment for labels ('top' or 'bottom')
    fontsize : font size of the phase labels
    """
    for start, end, label, color in PHASES:
        band_lo = max(start, lo)
        band_hi = min(end,   hi)
        if band_lo >= band_hi:
            continue
        ax.axvspan(band_lo, band_hi, alpha=PHASE_ALPHA, color=color,
                   zorder=0, linewidth=0)
        # boundary tick (skip the very first edge)
        if start > lo and start < hi:
            ax.axvline(start, color="#94A3B8", lw=0.6, ls=":", alpha=0.55, zorder=1)
        mid = (band_lo + band_hi) / 2
        ax.text(mid, label_y, label,
                transform=ax.get_xaxis_transform(),
                ha="center", va=label_va,
                fontsize=fontsize, color=LABEL_COLOR,
                fontweight="bold", alpha=0.85,
                clip_on=True)
