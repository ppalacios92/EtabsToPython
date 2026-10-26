"""Base exercise for `design.beams` — column-style section declaration
plus inspection via the P-M interaction diagram.

Run from the repo root:

    python design/example_beam_base.py

What this exercise shows:

  1. Declare the section the same way columns do — RebarLayout +
     RectangularSection + Beam(section=...).
  2. Inspect capacity via the closed-form scalars (capacity().phi_Mn_*).
  3. Inspect capacity via the P-M curve (beam.pm, beam.phi_Mn proxy).
  4. Run check(demands) and the three design modes.
  5. Verify after apply(envelope).

Beams have a P-M diagram now (restored). It's not used by `design()` —
that path remains closed-form for speed — but it's the right tool to
inspect Mn at non-zero Nu, plot the +M/-M envelopes, or audit a custom
rebar layout (skin reinforcement, bundled bars, multi-layer).
"""
from __future__ import annotations
import sys

# UTF-8 stdout on Windows so '·', 'φ', '§' don't crash the print
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from design import Concrete, Steel, Bar, RectangularSection
from design.sections.reinforcement import RebarLayout, two_layer_bars
from design.beams import Beam, BeamDemands


# ----------------------------------------------------------------- #
# 1) Materials and rebar layout — column-style declaration
# ----------------------------------------------------------------- #
concrete = Concrete(fc=28)               # MPa
steel    = Steel(fy=420, grade=60)

# Two-layer beam layout: 2 φ22 top, 4 φ22 bottom
group = two_layer_bars(
    b=400, h=500, cover=50,
    n_top=2, n_bottom=4,
    top_bar=Bar(diameter=22),
    bottom_bar=Bar(diameter=22),
    steel=steel,
)
section = RectangularSection(
    b=400, h=500,
    concrete=concrete,
    rebar=RebarLayout(groups=(group,)),
)

# ----------------------------------------------------------------- #
# 2) Beam — same pattern as Column(section=...)
# ----------------------------------------------------------------- #
beam = Beam(
    section=section,
    ln=5.0,                  # clear span (m)
    cover=40.0,              # clear cover to outside of stirrup
    Av=2 * 78.5,             # 2 legs φ10 = 2 × 78.5 mm²
    s=100.0,                 # stirrup spacing (mm)
    db_stirrup=10.0,
    seismic=True,
    Vg=80.0,                 # gravity shear at supports (kN)
    units=6,                 # kN_m_C
    label="B-base",
)

print("=" * 60)
print(" 1. SECTION SUMMARY  (column-style declaration)")
print("=" * 60)
beam.summary()

# ----------------------------------------------------------------- #
# 3) Inspect via the P-M interaction diagram
# ----------------------------------------------------------------- #
print()
print("=" * 60)
print(" 2. P-M INTERACTION DIAGRAM  (strain-compatibility)")
print("=" * 60)

# The bidirectional curve
pm = beam.pm
print(f"Po (pure compression)       = {pm.Po:.0f} kN")
print(f"To (pure tension)           = {pm.To:.0f} kN")
print(f"phi*Pn,max  (§22.4.2.1 cap) = {pm.Pn_max_phi:.0f} kN")

# Result-view proxies — column-style API
print()
print("Result-view proxies:")
print(f"  phi_Mn at Nu=0  (pos)   = {beam.phi_Mn.at(Nu=0, direction='pos'):.1f} kN·m")
print(f"  phi_Mn at Nu=0  (neg)   = {beam.phi_Mn.at(Nu=0, direction='neg'):.1f} kN·m")
print(f"  phi_Mn at Nu=200 (pos)  = {beam.phi_Mn.at(Nu=200, direction='pos'):.1f} kN·m"
      "   ← compression boosts +M")
print(f"  phi_Mn peak (both dir.) = {beam.phi_Mn.peak():.1f} kN·m")

# Also via Mn (nominal, unreduced)
print(f"  Mn peak (both dir.)     = {beam.Mn.peak():.1f} kN·m")

# Compare against closed-form scalars from capacity()
print()
cap = beam.capacity()
print("Sanity check vs closed-form mn_doubly / mpr:")
print(f"  phi_Mn_pos closed-form  = {cap.phi_Mn_pos:.1f} kN·m   (strain-compat: "
      f"{beam.phi_Mn.at(Nu=0, direction='pos'):.1f})")
print(f"  phi_Mn_neg closed-form  = {cap.phi_Mn_neg:.1f} kN·m   (strain-compat: "
      f"{beam.phi_Mn.at(Nu=0, direction='neg'):.1f})")
print(f"  Mpr_pos closed-form     = {cap.Mpr_pos:.1f} kN·m")
print(f"  Mpr_neg closed-form     = {cap.Mpr_neg:.1f} kN·m")

# ----------------------------------------------------------------- #
# 4) Demands + ACI checks (PROVIDED block)
# ----------------------------------------------------------------- #
print()
print("=" * 60)
print(" 3. ACI 318-25 CHECKS  on the as-input design")
print("=" * 60)

demands = BeamDemands(
    Mu_pos_i=80,    Mu_neg_i=300,  Vu_i=180,
    Mu_pos_mid=350, Mu_neg_mid=60, Vu_mid=80,
    Mu_pos_j=80,    Mu_neg_j=300,  Vu_j=180,
)
beam.print_checks(demands)

# ----------------------------------------------------------------- #
# 5) Three-mode design + apply + re-check
# ----------------------------------------------------------------- #
print()
print("=" * 60)
print(" 4. DESIGN  (minimum / demand / capacity / envelope)")
print("=" * 60)

results = beam.design(demands)
beam.print_design(results, demands=demands)

print()
print("=" * 60)
print(" 5. ADOPT THE ENVELOPE  and verify")
print("=" * 60)

snapshot = beam.current_proposal(demands)   # save the original for revert
beam.apply(results.envelope)
chk = beam.check(demands)
print(f"After apply(envelope):  ratio M = {chk.ratio_M_overall:.3f}   "
      f"ratio V max = {chk.ratio_V_overall:.3f}   passed = {chk.passed}")

# Revert if you want to compare back
beam.apply(snapshot)
chk2 = beam.check(demands)
print(f"After revert to PROVIDED:  ratio M = {chk2.ratio_M_overall:.3f}   "
      f"ratio V max = {chk2.ratio_V_overall:.3f}   passed = {chk2.passed}")

# ----------------------------------------------------------------- #
# 6) Optional: plot the P-M diagram (commented out — uncomment to view)
# ----------------------------------------------------------------- #
# import matplotlib.pyplot as plt
# beam.plot.pm(direction='both')
# plt.show()
