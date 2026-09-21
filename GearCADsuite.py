"""
Parametric Multi-Standard Gear CAD Suite - Gear App V1.5
=========================================================
File: gear_app.py
Author: Senior Computational CAD, Mechanical Geometry & Swiss Horological Software Engineer
Standards Reference:
  - Standard Involute: DIN 867, ISO 53, AGMA 2001-D04
  - Asymmetric Involute: Dual Base Circle Direct Involute Gearing
  - Swiss Horology: NIHS 20-25, NIHS 56702, NIHS 56704, NHS 56704, DIN 58405

Disciplines Supported:
1. Standard Involute Spur Gear (DIN 867 / ISO 53)
2. Asymmetric Involute Spur Gear (Independent Drive αd and Coast αc Pressure Angles)
3. Swiss Horological Epicycloidal Gear (Official NIHS 20-25 Standards)
4. Internal Involute Ring Gear
5. Internal Epicycloidal Ring Gear
"""

from __future__ import annotations
import math
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Dict, Any

import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.patches as patches

# CAD export modules with graceful fallback
try:
    import ezdxf
    from ezdxf import units as dxf_units
    EZDXF_AVAILABLE = True
except ImportError:
    EZDXF_AVAILABLE = False

try:
    import cadquery as cq
    CADQUERY_AVAILABLE = True
except ImportError:
    CADQUERY_AVAILABLE = False


# =============================================================================
# Standard NIHS Module Series (NIHS 56702)
# =============================================================================

NIHS_MODULE_SERIES = [
    0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15, 0.18, 0.20,
    0.22, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80,
    0.90, 1.00, 1.20, 1.50, 2.00
]


# =============================================================================
# Topological Helpers: Shortest-Path Arc Interpolation & Seam Deduplication
# =============================================================================

def shortest_path_arc(
    cx: float, cy: float, radius: float,
    theta_start: float, theta_end: float,
    num_pts: int = 16
) -> List[Tuple[float, float]]:
    """
    Sweeps a circular arc from theta_start to theta_end taking the direct shortest path,
    normalizing the angular delta across the +-pi boundary to prevent spirograph loops.
    """
    d_theta = (theta_end - theta_start + math.pi) % (2.0 * math.pi) - math.pi
    pts: List[Tuple[float, float]] = []
    for i in range(num_pts):
        t = i / max(1, num_pts - 1)
        theta_t = theta_start + t * d_theta
        pts.append((cx + radius * math.cos(theta_t), cy + radius * math.sin(theta_t)))
    return pts


def deduplicate_polygon(pts: List[Tuple[float, float]], tolerance: float = 1e-4) -> List[Tuple[float, float]]:
    """
    Filters consecutive duplicate vertices and ensures the endpoint does not repeat
    the starting point so CadQuery .close() builds a valid manifold B-Rep wire.
    """
    if not pts:
        return []

    clean: List[Tuple[float, float]] = [(float(pts[0][0]), float(pts[0][1]))]
    for pt in pts[1:]:
        x, y = float(pt[0]), float(pt[1])
        dx = x - clean[-1][0]
        dy = y - clean[-1][1]
        if math.hypot(dx, dy) >= tolerance:
            clean.append((x, y))

    if len(clean) > 2:
        dx0 = clean[-1][0] - clean[0][0]
        dy0 = clean[-1][1] - clean[0][1]
        if math.hypot(dx0, dy0) < tolerance:
            clean.pop()

    return clean


# =============================================================================
# Parametric Gear Configuration Dataclass
# =============================================================================

@dataclass
class GearConfig:
    gear_type: str              # Standard Involute, Asymmetric Involute, or Swiss NIHS
    module: float               # Metric module m (mm)
    teeth: int                  # Number of teeth z
    face_width: float           # Face width / extrusion depth b (mm)
    bore_diameter: float        # Center shaft hole diameter (mm)
    outer_blank_diameter: float = 0.0 # Outer diameter for internal ring gears (mm)
    backlash: float = 0.0       # Normal backlash allowance jt (mm)
    enable_keyway: bool = False # Keyway toggle
    keyway_width: float = 0.0   # Keyway width W (mm)
    keyway_depth: float = 0.0   # Keyway depth H (mm)

    # Standard Involute Specific
    pressure_angle_deg: float = 20.0
    involute_tip_form: str = "Standard Flat Tip"

    # Asymmetric Involute Specific
    drive_pressure_angle_deg: float = 28.0
    coast_pressure_angle_deg: float = 20.0

    # Tip Rounding Parameters (Involute)
    tip_fillet_factor: float = 0.20               # r_tip / m
    auto_full_radius_tip: bool = False            # Auto-calculate full semicircular crest

    # Swiss Horological Epicycloid Specific (NIHS 20-25)
    component_role: str = "Driving Wheel (Wheel)"          # "Driving Wheel (Wheel)" or "Driven Pinion (Leaf/Pinion)"
    enforce_nihs_proportions: bool = True                  # Strict NIHS calculation lock
    mating_teeth: int = 12                                 # Mating gear teeth count z_mate
    tooth_thickness_factor: float = 0.53                   # ks = s / p
    addendum_crest_form: str = "NIHS 20-25 Standard (Auto Gothic / Truncated)"
    addendum_factor: float = 1.40                          # ha / m
    root_profile_form: str = "U-Shaped Semicircular Root"  # "U-Shaped Semicircular Root" or "Radial Flank with Corner Fillet"
    dedendum_factor: float = 1.75                          # hf / m (standard NIHS 20-25 is 1.75)
    fillet_radius_factor: float = 0.40                     # r_fillet / m

    # Sampling resolution
    samples_per_flank: int = 40
    samples_per_arc: int = 20


# =============================================================================
# Engine A: Standard & Asymmetric Involute Geometry Engine (DIN 867 / Direct)
# =============================================================================

class InvoluteGeometryEngine:
    @staticmethod
    def inv(alpha_rad: float) -> float:
        return math.tan(alpha_rad) - alpha_rad

    @classmethod
    def generate_internal_tooth(cls, p: GearConfig) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
        m = float(p.module)
        z = int(p.teeth)
        jt = float(p.backlash)
        alpha = math.radians(p.pressure_angle_deg)
        h_f_factor = float(p.dedendum_factor) if float(p.dedendum_factor) > 0.1 else 1.25

        r = m * z / 2.0
        r_b = r * math.cos(alpha)
        r_a = r - 1.0 * m
        r_f = r + h_f_factor * m

        d = 2.0 * r
        d_a = 2.0 * r_a
        d_f = 2.0 * r_f
        d_b = 2.0 * r_b
        circular_pitch = math.pi * m

        s = (math.pi * m / 2.0) - (jt / 2.0)
        psi = s / (2.0 * r)
        theta_sector = math.pi / z

        inv_alpha = cls.inv(alpha)

        def theta_tooth(R: float) -> float:
            R_eff = r_b if R <= r_b else R
            alpha_R = math.acos(r_b / R_eff)
            delta_theta = cls.inv(alpha_R) - inv_alpha
            return psi - delta_theta

        theta_f = theta_tooth(r_f)
        theta_a = theta_tooth(max(r_b, r_a))
        theta_b = theta_tooth(r_b)

        # 1. Left Root Land (Outer Arc)
        th_root_L = np.linspace(-theta_sector, -theta_f, p.samples_per_arc)
        seg_A = [(r_f * math.cos(th), r_f * math.sin(th)) for th in th_root_L]

        # 2. Left Flank (Descending inward)
        seg_B = []
        for R_val in np.linspace(r_f, max(r_b, r_a), p.samples_per_flank):
            th_L = -theta_tooth(R_val)
            seg_B.append((R_val * math.cos(th_L), R_val * math.sin(th_L)))

        if r_a < r_b:
            seg_B.append((r_a * math.cos(-theta_b), r_a * math.sin(-theta_b)))

        # 3. Tip Land Arc (Inner Crest)
        if r_a < r_b:
            th_tip = np.linspace(-theta_b, theta_b, p.samples_per_arc)
        else:
            th_tip = np.linspace(-theta_a, theta_a, p.samples_per_arc)
        seg_C = [(r_a * math.cos(th), r_a * math.sin(th)) for th in th_tip]

        # 4. Right Flank (Ascending outward)
        seg_D = []
        if r_a < r_b:
            seg_D.append((r_a * math.cos(theta_b), r_a * math.sin(theta_b)))
        for R_val in np.linspace(max(r_b, r_a), r_f, p.samples_per_flank):
            th_R = theta_tooth(R_val)
            seg_D.append((R_val * math.cos(th_R), R_val * math.sin(th_R)))

        # 5. Right Root Land (Outer Arc)
        th_root_R = np.linspace(theta_f, theta_sector, p.samples_per_arc)
        seg_E = [(r_f * math.cos(th), r_f * math.sin(th)) for th in th_root_R]

        tooth_pts = seg_A + seg_B[1:] + seg_C[1:-1] + seg_D + seg_E[1:]

        dims = {
            "d": d, "d_a": d_a, "d_f": d_f, "d_b": d_b,
            "r": r, "r_a": r_a, "r_f": r_f, "r_b": r_b,
            "circular_pitch": circular_pitch,
            "tooth_thickness": s,
            "space_width": circular_pitch - s,
            "top_land_thickness": 2.0 * (theta_b if r_a < r_b else theta_a) * r_a,
            "bottom_clearance": (h_f_factor - 1.0) * m,
            "is_nihs_compliant": False,
            "nihs_notes": "Internal Involute Ring Gear",
        }
        return tooth_pts, dims

    @classmethod
    def generate_single_tooth(cls, p: GearConfig) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
        m = float(p.module)
        z = int(p.teeth)
        jt = float(p.backlash)
        k_tip = max(0.001, float(p.tip_fillet_factor))
        h_f_factor = float(p.dedendum_factor) if float(p.dedendum_factor) > 0.1 else 1.25

        is_asymmetric = "Asymmetric" in p.gear_type
        if is_asymmetric:
            alpha_d = math.radians(p.drive_pressure_angle_deg)
            alpha_c = math.radians(p.coast_pressure_angle_deg)
        else:
            alpha_d = math.radians(p.pressure_angle_deg)
            alpha_c = alpha_d

        d = m * z
        r = d / 2.0
        r_bd = r * math.cos(alpha_d)
        r_bc = r * math.cos(alpha_c)
        d_bd = 2.0 * r_bd
        d_bc = 2.0 * r_bc

        h_a = 1.0 * m
        h_f = h_f_factor * m
        d_a = d + 2.0 * h_a
        r_a = d_a / 2.0
        d_f = max(0.05 * m, d - 2.0 * h_f)
        r_f = d_f / 2.0

        circular_pitch = math.pi * m
        tooth_thickness = (math.pi * m / 2.0) - (jt / 2.0)

        if tooth_thickness <= 0:
            raise ValueError("Backlash value is too large: tooth thickness <= 0.")

        sector_half = math.pi / z
        psi = tooth_thickness / (2.0 * r)
        inv_ad = cls.inv(alpha_d)
        inv_ac = cls.inv(alpha_c)

        def flank_half_angle_d(radius: float) -> float:
            if radius <= r_bd:
                return psi + inv_ad
            alpha_R = math.acos(min(1.0, r_bd / radius))
            return psi + inv_ad - cls.inv(alpha_R)

        def flank_half_angle_c(radius: float) -> float:
            if radius <= r_bc:
                return psi + inv_ac
            alpha_R = math.acos(min(1.0, r_bc / radius))
            return psi + inv_ac - cls.inv(alpha_R)

        r_inv_start_d = max(r_bd, r_f)
        r_inv_start_c = max(r_bc, r_f)

        theta_root_d = flank_half_angle_d(r_inv_start_d)
        theta_root_c = flank_half_angle_c(r_inv_start_c)
        theta_tip_d = flank_half_angle_d(r_a)
        theta_tip_c = flank_half_angle_c(r_a)

        if theta_root_d >= sector_half:
            theta_root_d = sector_half * 0.999
        if theta_root_c >= sector_half:
            theta_root_c = sector_half * 0.999

        if (theta_tip_d + theta_tip_c) <= 0:
            raise ValueError("Tooth points at tip: addendum too large or tooth count too small.")

        tip_mode = p.involute_tip_form
        is_full_dome = ("Full-Radius" in tip_mode) or ("Tangent Dome" in tip_mode) or p.auto_full_radius_tip
        is_corner_filleted = ("Corner-Filleted" in tip_mode) or ("Corner Filleted" in tip_mode)

        if not is_asymmetric and is_full_dome:
            def calc_involute_apex(R_eval: float) -> Tuple[float, float, float, float]:
                al_R = 0.001 if R_eval <= r_bd else math.acos(min(1.0, r_bd / R_eval))
                th = flank_half_angle_d(R_eval)
                denom = max(1e-6, math.sin(th + al_R))
                xc_val = (R_eval * math.sin(al_R)) / denom
                rtip_val = (R_eval * math.sin(th)) / denom
                return xc_val + rtip_val, xc_val, rtip_val, th

            r_low = max(r_inv_start_d, r_a * 0.70)
            r_high = r_a
            r_contact = r_a * 0.95
            for _ in range(40):
                r_mid = 0.5 * (r_low + r_high)
                ap_mid, _, _, _ = calc_involute_apex(r_mid)
                if ap_mid < r_a:
                    r_low = r_mid
                else:
                    r_high = r_mid
                r_contact = r_mid

            _, xc_tip, r_tip_calc, th_contact = calc_involute_apex(r_contact)
            r_flank_max_d = r_contact
            r_flank_max_c = r_contact
            al_Rc = math.acos(min(1.0, r_bd / r_contact)) if r_contact > r_bd else 0.0
            phi_L = (th_contact + al_Rc) - math.pi
            phi_R = math.pi - (th_contact + al_Rc)
            top_land_w = 0.0

        elif not is_asymmetric and is_corner_filleted:
            r_tip_max = (r_a * math.sin(theta_tip_d)) / (1.0 + math.sin(theta_tip_d))
            r_tip_f = min(k_tip * m, r_tip_max * 0.95)
            r_cf = r_a - r_tip_f
            delta = math.asin(min(0.999, r_tip_f / r_cf))
            th_cL = -theta_tip_d + delta
            th_cR = +theta_tip_d - delta

            r_tan_top = math.sqrt(max(0.01, r_a**2 - 2.0 * r_a * r_tip_f))
            r_flank_max_d = max(r_inv_start_d, r_tan_top)
            r_flank_max_c = max(r_inv_start_c, r_tan_top)

            c_tip_lx = r_cf * math.cos(th_cL)
            c_tip_ly = r_cf * math.sin(th_cL)
            c_tip_rx = r_cf * math.cos(th_cR)
            c_tip_ry = r_cf * math.sin(th_cR)
            top_land_w = max(0.0, 2.0 * r_a * th_cR)

        else:
            r_flank_max_d = r_a
            r_flank_max_c = r_a
            top_land_w = (theta_tip_d + theta_tip_c) * r_a

        # Segment A: Left Root Land
        thetas_root_left = np.linspace(-sector_half, -theta_root_d, p.samples_per_arc)
        seg_A = [(r_f * math.cos(th), r_f * math.sin(th)) for th in thetas_root_left]

        # Segment B: Left Drive Involute Flank
        seg_B: List[Tuple[float, float]] = []
        if r_f < r_bd:
            for R_val in np.linspace(r_f, r_bd, max(3, p.samples_per_flank // 4), endpoint=False):
                seg_B.append((R_val * math.cos(-theta_root_d), R_val * math.sin(-theta_root_d)))

        for R_val in np.linspace(r_inv_start_d, r_flank_max_d, p.samples_per_flank):
            th_L = -flank_half_angle_d(R_val)
            seg_B.append((R_val * math.cos(th_L), R_val * math.sin(th_L)))

        # Segment C: Addendum Tip Land / Crest
        seg_C: List[Tuple[float, float]] = []
        if not is_asymmetric and is_full_dome:
            seg_C = shortest_path_arc(xc_tip, 0.0, r_tip_calc, phi_L, phi_R, p.samples_per_arc)
        elif not is_asymmetric and is_corner_filleted:
            al_start_L = -theta_tip_d - math.pi / 2.0
            al_end_L = th_cL
            seg_fillet_L = shortest_path_arc(c_tip_lx, c_tip_ly, r_tip_f, al_start_L, al_end_L, max(4, p.samples_per_arc // 2))

            seg_top = []
            if th_cR > th_cL:
                thetas_top = np.linspace(th_cL, th_cR, max(4, p.samples_per_arc // 2))
                seg_top = [(r_a * math.cos(tf), r_a * math.sin(tf)) for tf in thetas_top[1:-1]]

            al_start_R = th_cR
            al_end_R = +theta_tip_c + math.pi / 2.0
            seg_fillet_R = shortest_path_arc(c_tip_rx, c_tip_ry, r_tip_f, al_start_R, al_end_R, max(4, p.samples_per_arc // 2))
            seg_C = seg_fillet_L + seg_top + seg_fillet_R
        else:
            thetas_tip = np.linspace(-theta_tip_d, theta_tip_c, p.samples_per_arc)
            seg_C = [(r_a * math.cos(th), r_a * math.sin(th)) for th in thetas_tip[1:-1]]

        # Segment D: Right Coast Involute Flank
        seg_D: List[Tuple[float, float]] = []
        for R_val in np.linspace(r_flank_max_c, r_inv_start_c, p.samples_per_flank):
            th_R = +flank_half_angle_c(R_val)
            seg_D.append((R_val * math.cos(th_R), R_val * math.sin(th_R)))

        if r_f < r_bc:
            for R_val in np.linspace(r_bc, r_f, max(3, p.samples_per_flank // 4), endpoint=False):
                seg_D.append((R_val * math.cos(+theta_root_c), R_val * math.sin(+theta_root_c)))

        # Segment E: Right Root Land
        thetas_root_right = np.linspace(theta_root_c, sector_half, p.samples_per_arc)
        seg_E = [(r_f * math.cos(th), r_f * math.sin(th)) for th in thetas_root_right]

        tooth_pts = seg_A + seg_B + seg_C + seg_D + seg_E

        dims = {
            "d": d, "d_b": d_bd, "d_bd": d_bd, "d_bc": d_bc, "d_a": d_a, "d_f": d_f,
            "r": r, "r_b": r_bd, "r_bd": r_bd, "r_bc": r_bc, "r_a": r_a, "r_f": r_f,
            "circular_pitch": circular_pitch,
            "tooth_thickness": tooth_thickness,
            "space_width": circular_pitch - tooth_thickness,
            "top_land_thickness": top_land_w,
            "bottom_clearance": h_f - h_a,
            "is_nihs_compliant": False,
            "nihs_notes": "Industrial Involute (DIN 867 / ISO 53)" if not is_asymmetric else "Asymmetric Involute",
        }
        return tooth_pts, dims


# =============================================================================
# Engine B: Swiss Horology NIHS 20-25 Epicycloidal Engine
# =============================================================================

class HorologicalEpicycloidEngine:
    @classmethod
    def get_nihs_2025_factor(cls, z: int) -> float:
        if z <= 8: return 2.32
        elif z == 9: return 2.34
        elif z <= 11: return 2.38
        elif z <= 13: return 2.40
        elif z <= 16: return 2.44
        elif z <= 20: return 2.48
        elif z <= 25: return 2.52
        elif z <= 34: return 2.54
        elif z <= 54: return 2.58
        elif z <= 74: return 2.62
        else: return 2.66

    @classmethod
    def generate_single_tooth(cls, p: GearConfig) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
        m = float(p.module)
        z = int(p.teeth)
        z_mate = int(p.mating_teeth)
        jt = float(p.backlash)
        is_wheel = "Wheel" in p.component_role

        f_corr = cls.get_nihs_2025_factor(z)
        R_p = (m * z) / 2.0
        d = 2.0 * R_p
        R_a_std = (m * (z + f_corr)) / 2.0

        # Responsive dedendum height factor
        if p.enforce_nihs_proportions:
            kd = 1.75
        else:
            kd = float(p.dedendum_factor) if float(p.dedendum_factor) > 0.1 else 1.75

        R_f = max(0.05 * m, R_p - kd * m)
        d_f = 2.0 * R_f
        circular_pitch = math.pi * m

        if p.enforce_nihs_proportions:
            s = (1.41 * m if is_wheel else 0.42 * math.pi * m) - (jt / 2.0)
        else:
            ks = float(p.tooth_thickness_factor)
            s = (ks * math.pi * m) - (jt / 2.0)

        space_width = circular_pitch - s
        if s <= 0:
            raise ValueError("Backlash too large: tooth thickness <= 0.")

        phi = s / (2.0 * R_p)
        psi = math.pi / z
        if phi >= psi:
            phi = psi * 0.999

        alpha = psi - phi
        r_ogive = 0.8 * f_corr * m

        X_c = R_p * math.sin(phi) - r_ogive * math.cos(phi)
        Y_c = R_p * math.cos(phi) + r_ogive * math.sin(phi)

        disc = max(0.0001, r_ogive**2 - X_c**2)
        Y_apex = Y_c + math.sqrt(disc)

        # Decide between pointed Gothic crest and flat top land
        # If Y_apex < R_a_std or "Pointed Gothic" is enforced in default mode:
        enforce_pointed = (p.addendum_crest_form == "Enforce Pointed Gothic Apex")
        if enforce_pointed:
            is_pointed = True
        else:
            is_pointed = (Y_apex <= R_a_std)
        
        if is_pointed:
            R_a = Y_apex
            d_a = 2.0 * R_a
            top_land_w = 0.0
            tip_rx, tip_ry = 0.0, Y_apex
        else:
            R_a = R_a_std
            d_a = 2.0 * R_a
            d_c = math.hypot(X_c, Y_c)
            a_proj = (R_a**2 - r_ogive**2 + d_c**2) / (2.0 * d_c)
            h_proj = math.sqrt(max(0.0, R_a**2 - a_proj**2))
            tip_rx = a_proj * (X_c / d_c) + h_proj * (Y_c / d_c)
            tip_ry = a_proj * (Y_c / d_c) - h_proj * (X_c / d_c)
            th_tip = abs(math.atan2(tip_rx, tip_ry))
            top_land_w = 2.0 * th_tip * R_a

        h_a = R_a - R_p
        h_f = R_p - R_f

        sin_alpha = math.sin(alpha)
        cos_alpha = math.cos(alpha)

        D_r = R_f / (1.0 - sin_alpha)
        R_r = (R_f * sin_alpha) / (1.0 - sin_alpha)
        L = D_r * cos_alpha

        if L >= R_p * 0.95:
            L = R_p * 0.95
            D_r = L / cos_alpha
            R_r = D_r * sin_alpha
            R_f = max(0.05 * m, D_r - R_r)
            d_f = 2.0 * R_f

        c_rx = D_r * math.sin(psi)
        c_ry = D_r * math.cos(psi)

        P_space_center_R = (R_f * math.sin(psi), R_f * math.cos(psi))
        T_p_R = (L * math.sin(phi), L * math.cos(phi))

        ang_start_root_R = math.atan2(P_space_center_R[1] - c_ry, P_space_center_R[0] - c_rx)
        ang_end_root_R = math.atan2(T_p_R[1] - c_ry, T_p_R[0] - c_rx)

        seg_root_R = shortest_path_arc(c_rx, c_ry, R_r, ang_start_root_R, ang_end_root_R, p.samples_per_arc)

        P_p_R = (R_p * math.sin(phi), R_p * math.cos(phi))
        rad_samples = np.linspace(L, R_p, max(3, p.samples_per_flank // 3))
        seg_radial_R = [(rad * math.sin(phi), rad * math.cos(phi)) for rad in rad_samples[1:]]

        ang_start_ogive_R = math.atan2(P_p_R[1] - Y_c, P_p_R[0] - X_c)
        ang_end_ogive_R = math.atan2(tip_ry - Y_c, tip_rx - X_c)

        seg_ogive_R = shortest_path_arc(X_c, Y_c, r_ogive, ang_start_ogive_R, ang_end_ogive_R, p.samples_per_flank)
        seg_ogive_R[-1] = (tip_rx, tip_ry)

        right_half = seg_root_R + seg_radial_R[1:] + seg_ogive_R[1:]
        left_pts_down = [(-x, y) for (x, y) in reversed(right_half)]

        if is_pointed:
            tooth_pts = right_half[:-1] + left_pts_down
        else:
            thetas_top = np.linspace(-th_tip, th_tip, p.samples_per_arc)
            seg_top_land = [(R_a * math.sin(t), R_a * math.cos(t)) for t in reversed(thetas_top)]
            tooth_pts = right_half + seg_top_land[1:-1] + left_pts_down

        is_compliant = True
        compliance_msg = "✓ NIHS 20-25 Compliant"
        if is_wheel and z < 20:
            is_compliant = False
            compliance_msg = "Note: NIHS specifies z >= 20 for driving wheels"
        elif not is_wheel and (z < 6 or z > 18):
            is_compliant = False
            compliance_msg = f"Note: NIHS specifies z in [6, 18] for pinions (current: {z})"

        dims = {
            "d": d, "d_g": 2.0 * ((m * z_mate) / 4.0), "d_a": d_a, "d_f": d_f,
            "r": R_p, "r_g": (m * z_mate) / 4.0, "r_a": R_a, "r_f": R_f,
            "circular_pitch": circular_pitch,
            "tooth_thickness": s,
            "space_width": space_width,
            "top_land_thickness": top_land_w,
            "bottom_clearance": h_f - h_a,
            "z_mate": float(z_mate),
            "ratio": float(z) / float(z_mate),
            "is_ogive": is_pointed,
            "thickness_ratio": s / circular_pitch,
            "is_nihs_compliant": is_compliant,
            "nihs_notes": compliance_msg,
            "f_correction": f_corr,
            "ogive_radius": r_ogive,
            "root_fillet_radius": R_r,
        }
        return tooth_pts, dims

    @classmethod
    def generate_internal_tooth(cls, p: GearConfig) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
        m = float(p.module)
        z = int(p.teeth)
        jt = float(p.backlash)

        f_corr = cls.get_nihs_2025_factor(z)
        R_p = (m * z) / 2.0
        d = 2.0 * R_p

        kd = 1.40 if p.enforce_nihs_proportions else (float(p.dedendum_factor) if float(p.dedendum_factor) > 0.1 else 1.40)
        R_f = R_p + kd * m

        h_a = 1.0 * m
        R_a_std = R_p - h_a

        if p.enforce_nihs_proportions:
            s = 1.41 * m - jt
        else:
            s = float(p.tooth_thickness_factor) * math.pi * m - jt

        phi = s / (2.0 * R_p)
        psi = math.pi / z
        if phi >= psi:
            phi = psi * 0.999
        alpha = psi - phi

        sin_alpha = math.sin(alpha)
        cos_alpha = math.cos(alpha)
        D_r = R_f / (1.0 + sin_alpha)
        R_r = D_r * sin_alpha
        L = D_r * cos_alpha

        if L <= R_p * 1.02:
            L = R_p * 1.02
            D_r = L / cos_alpha
            R_r = D_r * sin_alpha
            R_f = D_r + R_r

        c_rx = D_r * math.sin(psi)
        c_ry = D_r * math.cos(psi)

        r_ogive = 0.8 * f_corr * m
        X_c = R_p * math.sin(phi) - r_ogive * math.cos(phi)
        Y_c = R_p * math.cos(phi) + r_ogive * math.sin(phi)

        disc = max(0.0001, r_ogive**2 - X_c**2)
        Y_apex = Y_c - math.sqrt(disc)

        enforce_pointed = (p.addendum_crest_form == "Enforce Pointed Gothic Apex")
        if enforce_pointed:
            is_pointed = True
        else:
            is_pointed = (Y_apex >= R_a_std)
            
        if is_pointed:
            R_a = Y_apex
            tip_rx, tip_ry = 0.0, Y_apex
        else:
            R_a = R_a_std
            d_c = math.hypot(X_c, Y_c)
            a_proj = (R_a**2 - r_ogive**2 + d_c**2) / (2.0 * d_c)
            if R_a**2 < a_proj**2:
                is_pointed = True
                R_a = Y_apex
                tip_rx, tip_ry = 0.0, Y_apex
            else:
                h_proj = math.sqrt(max(0.0, R_a**2 - a_proj**2))
                tip_rx = a_proj * (X_c / d_c) + h_proj * (Y_c / d_c)
                tip_ry = a_proj * (Y_c / d_c) - h_proj * (X_c / d_c)

        ang_start_root_R = math.atan2(R_f * math.cos(psi) - c_ry, R_f * math.sin(psi) - c_rx)
        ang_end_root_R = math.atan2(L * math.cos(phi) - c_ry, L * math.sin(phi) - c_rx)

        seg_root_R = shortest_path_arc(c_rx, c_ry, R_r, ang_start_root_R, ang_end_root_R, p.samples_per_arc)

        rad_samples = np.linspace(L, R_p, max(3, p.samples_per_flank // 3))
        seg_radial_R = [(rad * math.sin(phi), rad * math.cos(phi)) for rad in rad_samples[1:]]

        ang_start_ogive_R = math.atan2(R_p * math.cos(phi) - Y_c, R_p * math.sin(phi) - X_c)
        ang_end_ogive_R = math.atan2(tip_ry - Y_c, tip_rx - X_c)

        seg_ogive_R = shortest_path_arc(X_c, Y_c, r_ogive, ang_start_ogive_R, ang_end_ogive_R, p.samples_per_flank)
        seg_ogive_R[-1] = (tip_rx, tip_ry)

        right_half = seg_root_R + seg_radial_R[1:] + seg_ogive_R[1:]
        left_pts = [(-x, y) for (x, y) in right_half]
        left_pts_down = list(reversed(left_pts))

        if is_pointed:
            tooth_pts = right_half[:-1] + left_pts_down
        else:
            ang_tip_R = math.atan2(tip_ry, tip_rx)
            ang_tip_L = math.atan2(tip_ry, -tip_rx)
            seg_top = shortest_path_arc(0, 0, R_a, ang_tip_R, ang_tip_L, max(3, p.samples_per_arc // 2))
            tooth_pts = right_half[:-1] + seg_top + left_pts_down[1:]

        dims = {
            "d": d, "d_a": 2.0 * R_a, "d_f": 2.0 * R_f, "d_b": d,
            "r": R_p, "r_a": R_a, "r_f": R_f, "r_b": R_p,
            "r_g": r_ogive, "d_g": 2.0 * r_ogive,
            "circular_pitch": math.pi * m,
            "tooth_thickness": s,
            "space_width": (math.pi * m) - s,
            "top_land_thickness": 0.0,
            "bottom_clearance": R_f - R_p,
            "is_nihs_compliant": False,
            "nihs_notes": "Internal Epicycloidal Ring Gear",
            "f_correction": f_corr,
            "ogive_radius": r_ogive,
            "root_fillet_radius": R_r,
            "z_mate": float(p.mating_teeth),
            "ratio": float(z) / max(1.0, float(p.mating_teeth)),
            "is_ogive": True,
            "thickness_ratio": s / (math.pi * m),
        }
        return tooth_pts, dims


# =============================================================================
# Unified Multi-Standard Geometry Dispatcher
# =============================================================================

class UnifiedGearEngine:
    @classmethod
    def calculate_gear_geometry(cls, p: GearConfig) -> Dict[str, Any]:
        if "Internal" in p.gear_type and "Epicycloidal" in p.gear_type:
            tooth_pts, dims = HorologicalEpicycloidEngine.generate_internal_tooth(p)
        elif "Epicycloidal" in p.gear_type:
            tooth_pts, dims = HorologicalEpicycloidEngine.generate_single_tooth(p)
        elif "Internal" in p.gear_type:
            inv_pts, dims = InvoluteGeometryEngine.generate_internal_tooth(p)
            tooth_pts = [(-y, x) for x, y in inv_pts]
        else:
            inv_pts, dims = InvoluteGeometryEngine.generate_single_tooth(p)
            tooth_pts = [(-y, x) for x, y in inv_pts]

        z = int(p.teeth)
        d_bore = float(p.bore_diameter)
        r_bore = d_bore / 2.0
        r_f = dims["r_f"]

        if "Internal" in p.gear_type:
            r_outer = p.outer_blank_diameter / 2.0
            margin = 6.0 if "Epicycloidal" in p.gear_type else 20.0
            min_r_outer = (dims["d"] + margin * float(p.module)) / 2.0
            if r_outer < min_r_outer:
                r_outer = min_r_outer

            th_circle = np.linspace(0, 2.0 * math.pi, max(360, z * 2), endpoint=False)
            raw_outer_boundary = [(r_outer * math.cos(t), r_outer * math.sin(t)) for t in th_circle]
            clean_outer_boundary = deduplicate_polygon(raw_outer_boundary, tolerance=1e-4)

            raw_bore_boundary: List[Tuple[float, float]] = []
            tau = 2.0 * math.pi / z
            for i in range(z):
                phi = i * tau
                c_phi, s_phi = math.cos(phi), math.sin(phi)
                for x, y in tooth_pts:
                    rx = x * c_phi - y * s_phi
                    ry = x * s_phi + y * c_phi
                    raw_bore_boundary.append((rx, ry))
            clean_bore_boundary = deduplicate_polygon(raw_bore_boundary, tolerance=1e-4)

        else:
            if r_bore > 0 and r_bore >= r_f:
                raise ValueError(
                    f"Bore radius ({r_bore:.3f} mm) must be strictly smaller than root radius ({r_f:.3f} mm)."
                )

            if p.enable_keyway and p.keyway_width > 0 and p.keyway_depth > 0:
                if (r_bore + p.keyway_depth) >= r_f:
                    raise ValueError(
                        f"Bore + Keyway depth ({r_bore + p.keyway_depth:.3f} mm) exceeds root radius ({r_f:.3f} mm)."
                    )
                if (p.keyway_width / 2.0) >= r_bore and r_bore > 0:
                    raise ValueError("Keyway width must be strictly less than bore diameter.")

            raw_outer_boundary: List[Tuple[float, float]] = []
            tau = 2.0 * math.pi / z

            for i in range(z):
                phi = i * tau
                c_phi, s_phi = math.cos(phi), math.sin(phi)
                for x, y in tooth_pts:
                    rx = x * c_phi - y * s_phi
                    ry = x * s_phi + y * c_phi
                    raw_outer_boundary.append((rx, ry))

            clean_outer_boundary = deduplicate_polygon(raw_outer_boundary, tolerance=1e-4)

            raw_bore_boundary = []
            if r_bore > 0:
                if not p.enable_keyway or p.keyway_width <= 0 or p.keyway_depth <= 0:
                    th_circle = np.linspace(0, 2.0 * math.pi, 72, endpoint=False)
                    raw_bore_boundary = [(r_bore * math.cos(t), r_bore * math.sin(t)) for t in th_circle]
                else:
                    hw = p.keyway_width / 2.0
                    y_top = r_bore + p.keyway_depth
                    th_intersect = math.asin(hw / r_bore)
                    th_right = math.pi / 2.0 - th_intersect
                    th_left = math.pi / 2.0 + th_intersect

                    n_arc = int(72 * (2.0 * math.pi - (th_left - th_right)) / (2.0 * math.pi))
                    arc_thetas = np.linspace(th_left, th_right + 2.0 * math.pi, max(16, n_arc), endpoint=False)
                    bore_arc = [
                        (r_bore * math.cos(t % (2.0 * math.pi)), r_bore * math.sin(t % (2.0 * math.pi)))
                        for t in arc_thetas
                    ]
                    keyway_box = [(hw, y_top), (-hw, y_top)]
                    raw_bore_boundary = bore_arc + keyway_box

            clean_bore_boundary = deduplicate_polygon(raw_bore_boundary, tolerance=1e-4)

        result = {
            "outer_perimeter": clean_outer_boundary,
            "bore_perimeter": clean_bore_boundary,
            "gear_type": p.gear_type,
            "d": dims["d"],
            "d_a": dims["d_a"],
            "d_f": dims["d_f"],
            "r": dims["r"],
            "r_a": dims["r_a"],
            "r_f": dims["r_f"],
            "r_bore": r_bore,
            "circular_pitch": dims["circular_pitch"],
            "tooth_thickness": dims["tooth_thickness"],
            "space_width": dims["space_width"],
            "top_land_thickness": dims["top_land_thickness"],
            "bottom_clearance": dims["bottom_clearance"],
            "face_width": float(p.face_width),
            "backlash": float(p.backlash),
            "enable_keyway": bool(p.enable_keyway),
            "is_nihs_compliant": dims["is_nihs_compliant"],
            "nihs_notes": dims["nihs_notes"],
        }

        if "Epicycloidal" in p.gear_type:
            result["d_g"] = dims["d_g"]
            result["r_g"] = dims["r_g"]
            result["z_mate"] = dims["z_mate"]
            result["ratio"] = dims["ratio"]
            result["is_ogive"] = dims["is_ogive"]
            result["thickness_ratio"] = dims["thickness_ratio"]
            result["component_role"] = p.component_role
            result["f_correction"] = dims.get("f_correction", 2.50)
            result["ogive_radius"] = dims.get("ogive_radius", 1.0)
            result["root_fillet_radius"] = dims.get("root_fillet_radius", 0.2)
        elif "Asymmetric" in p.gear_type:
            result["d_bd"] = dims["d_bd"]
            result["d_bc"] = dims["d_bc"]
            result["r_bd"] = dims["r_bd"]
            result["r_bc"] = dims["r_bc"]
            result["drive_pressure_angle"] = p.drive_pressure_angle_deg
            result["coast_pressure_angle"] = p.coast_pressure_angle_deg
        else:
            result["d_b"] = dims["d_b"]
            result["r_b"] = dims["r_b"]
            result["pressure_angle"] = p.pressure_angle_deg

        return result


# =============================================================================
# Custom Dual Synchronized Input Control (Slider + Numeric Box)
# =============================================================================

class SynchronizedSliderEntry(ttk.Frame):
    def __init__(
        self,
        parent,
        label_text: str,
        from_: float,
        to: float,
        step: float,
        default_val: float,
        is_integer: bool = False,
        unit: str = "",
        on_change: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        super().__init__(parent, **kwargs)
        self.from_ = from_
        self.to = to
        self.step = step
        self.is_integer = is_integer
        self.unit = unit
        self.on_change = on_change
        self._updating = False

        header = ttk.Frame(self)
        header.pack(fill=tk.X, expand=True)

        lbl = ttk.Label(header, text=label_text, font=("Segoe UI", 9, "bold"))
        lbl.pack(side=tk.LEFT)

        if unit:
            unit_lbl = ttk.Label(header, text=f"[{unit}]", font=("Segoe UI", 8), foreground="#64748B")
            unit_lbl.pack(side=tk.LEFT, padx=4)

        control_row = ttk.Frame(self)
        control_row.pack(fill=tk.X, expand=True, pady=(2, 4))

        self.scale_var = tk.DoubleVar(value=default_val)
        self.entry_var = tk.StringVar(value=str(int(default_val) if is_integer else f"{default_val:.3f}"))

        self.slider = ttk.Scale(
            control_row,
            from_=from_,
            to=to,
            variable=self.scale_var,
            orient=tk.HORIZONTAL,
            command=self._on_slider_move,
        )
        self.slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8))

        self.entry = ttk.Entry(
            control_row,
            textvariable=self.entry_var,
            width=8,
            justify="right",
            font=("Segoe UI", 9),
        )
        self.entry.pack(side=tk.RIGHT)
        self.entry.bind("<KeyRelease>", self._on_entry_type)
        self.entry.bind("<FocusOut>", self._on_entry_focus_out)

    def get_value(self) -> float:
        try:
            val = float(self.entry_var.get())
            return int(round(val)) if self.is_integer else val
        except ValueError:
            return float(self.scale_var.get())

    def set_value(self, val: float) -> None:
        self._updating = True
        val = max(self.from_, min(self.to, val))
        if self.is_integer:
            val = int(round(val))
            self.scale_var.set(val)
            self.entry_var.set(str(val))
        else:
            self.scale_var.set(val)
            self.entry_var.set(f"{val:.3f}")
        self._updating = False

    def set_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        self.slider.configure(state=state)
        self.entry.configure(state=state)

    def _on_slider_move(self, val_str: str) -> None:
        if self._updating:
            return
        self._updating = True
        try:
            val = float(val_str)
            if self.is_integer:
                val = int(round(val))
                self.scale_var.set(val)
                self.entry_var.set(str(val))
            else:
                if self.step > 0:
                    val = round(val / self.step) * self.step
                self.entry_var.set(f"{val:.3f}")
            if self.on_change:
                self.on_change()
        finally:
            self._updating = False

    def _on_entry_type(self, event=None) -> None:
        if self._updating:
            return
        text = self.entry_var.get().strip()
        try:
            val = float(text)
            if self.from_ <= val <= self.to:
                self._updating = True
                self.scale_var.set(val)
                self._updating = False
                if self.on_change:
                    self.on_change()
        except ValueError:
            pass

    def _on_entry_focus_out(self, event=None) -> None:
        try:
            val = float(self.entry_var.get().strip())
            self.set_value(val)
            if self.on_change:
                self.on_change()
        except ValueError:
            self.set_value(self.scale_var.get())


# =============================================================================
# Main Application GUI (Gear App V1.2)
# =============================================================================

class GearCADApplication(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Parametric Multi-Standard Gear CAD Suite (Gear App V1.2)")
        self.geometry("1400x940")
        self.minsize(1150, 780)

        self._setup_theme()
        self.calc_data: Optional[Dict[str, Any]] = None
        self._debounce_timer: Optional[str] = None

        self._build_ui()
        self._on_gear_type_changed()

    def _setup_theme(self) -> None:
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        self.style.configure(".", font=("Segoe UI", 9))
        self.style.configure("TFrame", background="#F8FAFC")
        self.style.configure("Sidebar.TFrame", background="#FFFFFF")
        self.style.configure("Header.TLabel", font=("Segoe UI", 11, "bold"), foreground="#0F172A", background="#FFFFFF")
        self.style.configure("SubHeader.TLabel", font=("Segoe UI", 9, "bold"), foreground="#334155", background="#FFFFFF")
        self.style.configure("MetricName.TLabel", font=("Segoe UI", 9), foreground="#64748B", background="#F8FAFC")
        self.style.configure("MetricVal.TLabel", font=("Segoe UI", 9, "bold"), foreground="#0F172A", background="#F8FAFC")
        self.style.configure("Primary.TButton", font=("Segoe UI", 9, "bold"), foreground="#FFFFFF", background="#2563EB")
        self.style.map("Primary.TButton", background=[("active", "#1D4ED8"), ("disabled", "#94A3B8")])

    def _build_ui(self) -> None:
        container = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        container.pack(fill=tk.BOTH, expand=True)

        left_wrapper = ttk.Frame(container, style="Sidebar.TFrame", width=480)
        container.add(left_wrapper, weight=0)

        canvas_scroll = tk.Canvas(left_wrapper, bg="#FFFFFF", highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_wrapper, orient=tk.VERTICAL, command=canvas_scroll.yview)
        scroll_content = ttk.Frame(canvas_scroll, style="Sidebar.TFrame", padding=14)

        scroll_content.bind(
            "<Configure>",
            lambda e: canvas_scroll.configure(scrollregion=canvas_scroll.bbox("all")),
        )
        canvas_scroll.create_window((0, 0), window=scroll_content, anchor="nw", width=440)
        canvas_scroll.configure(yscrollcommand=scrollbar.set)

        canvas_scroll.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._build_sidebar_controls(scroll_content)

        right_panel = ttk.Frame(container, style="TFrame")
        container.add(right_panel, weight=1)
        self._build_viewport(right_panel)

    def _build_sidebar_controls(self, parent: ttk.Frame) -> None:
        type_box = ttk.LabelFrame(parent, text=" Gearing Standard & Discipline ", padding=8)
        type_box.pack(fill=tk.X, pady=(0, 10))
        self.gear_type_var = tk.StringVar(value="Swiss Horological Epicycloidal Gear (Official NIHS 20-25)")
        self.combo_gear_type = ttk.Combobox(
            type_box,
            textvariable=self.gear_type_var,
            values=[
                "Standard Involute Spur Gear (DIN 867 / ISO 53)",
                "Asymmetric Involute Spur Gear",
                "Swiss Horological Epicycloidal Gear (Official NIHS 20-25)",
                "Internal Involute Ring Gear",
                "Internal Epicycloidal Ring Gear"
            ],
            state="readonly",
            font=("Segoe UI", 9, "bold"),
        )
        self.combo_gear_type.pack(fill=tk.X, pady=2)
        self.combo_gear_type.bind("<<ComboboxSelected>>", lambda e: self._on_gear_type_changed())

        ttk.Label(parent, text="Primary Module & Dimensions", style="Header.TLabel").pack(anchor="w", pady=(0, 4))

        self.mod_frame = ttk.Frame(parent, style="Sidebar.TFrame")
        self.mod_frame.pack(fill=tk.X, pady=(0, 2))
        ttk.Label(self.mod_frame, text="NIHS Module Series:", font=("Segoe UI", 8, "bold"), foreground="#475569").pack(side=tk.LEFT)
        self.nihs_mod_var = tk.StringVar(value="0.50")
        self.combo_nihs_mod = ttk.Combobox(
            self.mod_frame,
            textvariable=self.nihs_mod_var,
            values=[f"{m_val:.2f}" if m_val >= 0.1 else f"{m_val:.3f}" for m_val in NIHS_MODULE_SERIES],
            state="readonly",
            width=8,
        )
        self.combo_nihs_mod.pack(side=tk.RIGHT)
        self.combo_nihs_mod.bind("<<ComboboxSelected>>", self._on_nihs_module_picked)

        self.ctrl_m = SynchronizedSliderEntry(
            parent, "Module (m)", from_=0.05, to=5.0, step=0.01, default_val=0.50, unit="mm",
            on_change=self._on_control_changed, style="Sidebar.TFrame"
        )
        self.ctrl_m.pack(fill=tk.X)

        self.ctrl_z = SynchronizedSliderEntry(
            parent, "Teeth Count (z)", from_=6, to=180, step=1.0, default_val=60, is_integer=True,
            on_change=self._on_control_changed, style="Sidebar.TFrame"
        )
        self.ctrl_z.pack(fill=tk.X)

        self.ctrl_backlash = SynchronizedSliderEntry(
            parent, "Backlash (jt)", from_=0.0, to=0.5, step=0.005, default_val=0.01, unit="mm",
            on_change=self._on_control_changed, style="Sidebar.TFrame"
        )
        self.ctrl_backlash.pack(fill=tk.X)

        self.dynamic_params_frame = ttk.Frame(parent, style="Sidebar.TFrame")
        self.dynamic_params_frame.pack(fill=tk.X, pady=(4, 0))

        # Standard Involute Panel
        self.std_involute_frame = ttk.Frame(self.dynamic_params_frame, style="Sidebar.TFrame")
        ttk.Label(self.std_involute_frame, text="Standard Involute (DIN 867)", style="SubHeader.TLabel").pack(anchor="w", pady=(4, 2))
        self.ctrl_alpha = SynchronizedSliderEntry(
            self.std_involute_frame, "Pressure Angle (α)", from_=14.5, to=35.0, step=0.5, default_val=20.0, unit="deg",
            on_change=self._on_control_changed, style="Sidebar.TFrame"
        )
        self.ctrl_alpha.pack(fill=tk.X)

        inv_tip_row = ttk.Frame(self.std_involute_frame, style="Sidebar.TFrame")
        inv_tip_row.pack(fill=tk.X, pady=(2, 4))
        ttk.Label(inv_tip_row, text="Addendum Form:", font=("Segoe UI", 9, "bold"), background="#FFFFFF").pack(side=tk.LEFT)
        self.involute_tip_var = tk.StringVar(value="Standard Flat Tip")
        self.combo_inv_tip = ttk.Combobox(
            inv_tip_row,
            textvariable=self.involute_tip_var,
            values=["Standard Flat Tip", "Full-Radius Tangent Dome", "Corner-Filleted Crest"],
            state="readonly",
            width=24,
        )
        self.combo_inv_tip.pack(side=tk.RIGHT)
        self.combo_inv_tip.bind("<<ComboboxSelected>>", lambda e: self._on_tip_mode_changed())

        # Asymmetric Involute Panel
        self.asym_involute_frame = ttk.Frame(self.dynamic_params_frame, style="Sidebar.TFrame")
        ttk.Label(self.asym_involute_frame, text="Asymmetric Involute (Direct Gear Design)", style="SubHeader.TLabel").pack(anchor="w", pady=(4, 2))
        self.ctrl_alpha_d = SynchronizedSliderEntry(
            self.asym_involute_frame, "Drive Pressure Angle (αd)", from_=14.5, to=45.0, step=0.5, default_val=28.0, unit="deg",
            on_change=self._on_control_changed, style="Sidebar.TFrame"
        )
        self.ctrl_alpha_d.pack(fill=tk.X)

        self.ctrl_alpha_c = SynchronizedSliderEntry(
            self.asym_involute_frame, "Coast Pressure Angle (αc)", from_=14.5, to=35.0, step=0.5, default_val=20.0, unit="deg",
            on_change=self._on_control_changed, style="Sidebar.TFrame"
        )
        self.ctrl_alpha_c.pack(fill=tk.X)

        # Swiss Horology Panel (NIHS 20-25)
        self.epicycloid_frame = ttk.Frame(self.dynamic_params_frame, style="Sidebar.TFrame")
        ttk.Label(self.epicycloid_frame, text="Swiss Watchmaking Standards (NIHS 20-25)", style="SubHeader.TLabel").pack(anchor="w", pady=(4, 2))

        self.enforce_nihs_var = tk.BooleanVar(value=True)
        self.chk_enforce_nihs = ttk.Checkbutton(
            self.epicycloid_frame,
            text="Enforce Strict NIHS Proportions (Auto-Lock)",
            variable=self.enforce_nihs_var,
            command=self._on_enforce_nihs_toggled,
            style="Sidebar.TFrame",
        )
        self.chk_enforce_nihs.pack(anchor="w", pady=(2, 4))

        role_row = ttk.Frame(self.epicycloid_frame, style="Sidebar.TFrame")
        role_row.pack(fill=tk.X, pady=(2, 4))
        ttk.Label(role_row, text="Component Role:", font=("Segoe UI", 9, "bold"), background="#FFFFFF").pack(side=tk.LEFT)
        self.component_role_var = tk.StringVar(value="Driving Wheel (Wheel)")
        self.combo_role = ttk.Combobox(
            role_row,
            textvariable=self.component_role_var,
            values=["Driving Wheel (Wheel)", "Driven Pinion (Leaf/Pinion)"],
            state="readonly",
            width=24,
        )
        self.combo_role.pack(side=tk.RIGHT)
        self.combo_role.bind("<<ComboboxSelected>>", lambda e: self._on_role_changed())

        self.ctrl_z_mate = SynchronizedSliderEntry(
            self.epicycloid_frame, "Mating Count (z_mate)", from_=6, to=180, step=1.0, default_val=12, is_integer=True,
            on_change=self._on_control_changed, style="Sidebar.TFrame"
        )
        self.ctrl_z_mate.pack(fill=tk.X)

        self.ctrl_ks = SynchronizedSliderEntry(
            self.epicycloid_frame, "Tooth Thickness Ratio (s/p)", from_=0.35, to=0.65, step=0.005, default_val=0.53,
            on_change=self._on_custom_proportion_changed, style="Sidebar.TFrame"
        )
        self.ctrl_ks.pack(fill=tk.X)

        crest_row = ttk.Frame(self.epicycloid_frame, style="Sidebar.TFrame")
        crest_row.pack(fill=tk.X, pady=(2, 4))
        ttk.Label(crest_row, text="Addendum Form:", font=("Segoe UI", 9, "bold"), background="#FFFFFF").pack(side=tk.LEFT)
        self.crest_form_var = tk.StringVar(value="NIHS 20-25 Standard (Auto Gothic / Truncated)")
        self.combo_crest = ttk.Combobox(
            crest_row,
            textvariable=self.crest_form_var,
            values=[
                "NIHS 20-25 Standard (Auto Gothic / Truncated)",
                "Enforce Pointed Gothic Apex",
            ],
            state="readonly",
            width=24,
        )
        self.combo_crest.pack(side=tk.RIGHT)
        self.combo_crest.bind("<<ComboboxSelected>>", lambda e: self._on_tip_mode_changed())

        root_row = ttk.Frame(self.epicycloid_frame, style="Sidebar.TFrame")
        root_row.pack(fill=tk.X, pady=(2, 4))
        ttk.Label(root_row, text="Root Trough Form:", font=("Segoe UI", 9, "bold"), background="#FFFFFF").pack(side=tk.LEFT)
        self.root_form_var = tk.StringVar(value="U-Shaped Semicircular Root")
        self.combo_root = ttk.Combobox(
            root_row,
            textvariable=self.root_form_var,
            values=["U-Shaped Semicircular Root", "Radial Flank with Corner Fillet"],
            state="readonly",
            width=24,
        )
        self.combo_root.pack(side=tk.RIGHT)
        self.combo_root.bind("<<ComboboxSelected>>", lambda e: self._on_control_changed())

        # Responsive Dedendum Factor Slider (Enabled across all gear types)
        self.ctrl_kd = SynchronizedSliderEntry(
            parent, "Dedendum Factor (hf/m)", from_=0.80, to=2.40, step=0.02, default_val=1.75,
            on_change=self._on_custom_proportion_changed, style="Sidebar.TFrame"
        )
        self.ctrl_kd.pack(fill=tk.X)

        self.tip_panel = ttk.LabelFrame(parent, text=" Tip Fillet & Crest Controls ", padding=8)
        self.tip_panel.pack(fill=tk.X, pady=(8, 0))

        self.ctrl_tip_fillet = SynchronizedSliderEntry(
            self.tip_panel, "Tip Fillet Radius (rtip/m)", from_=0.02, to=0.65, step=0.01, default_val=0.20,
            on_change=self._on_control_changed, style="Sidebar.TFrame"
        )
        self.ctrl_tip_fillet.pack(fill=tk.X)

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        ttk.Label(parent, text="Solid Blank & Hub", style="Header.TLabel").pack(anchor="w", pady=(0, 4))

        self.ctrl_b = SynchronizedSliderEntry(
            parent, "Face Width (b)", from_=0.1, to=50.0, step=0.1, default_val=1.5, unit="mm",
            on_change=self._on_control_changed, style="Sidebar.TFrame"
        )
        self.ctrl_b.pack(fill=tk.X)

        self.ctrl_bore = SynchronizedSliderEntry(
            parent, "Center Pivot/Bore (d_bore)", from_=0.0, to=30.0, step=0.1, default_val=1.0, unit="mm",
            on_change=self._on_control_changed, style="Sidebar.TFrame"
        )
        self.ctrl_bore.pack(fill=tk.X)

        self.enable_keyway_var = tk.BooleanVar(value=False)
        self.chk_keyway = ttk.Checkbutton(
            parent,
            text="Enable Hub Keyway Slot (DIN 6885)",
            variable=self.enable_keyway_var,
            command=self._on_control_changed,
            style="Sidebar.TFrame",
        )
        self.chk_keyway.pack(anchor="w", pady=(2, 2))

        self.ctrl_kw_w = SynchronizedSliderEntry(
            parent, "Keyway Width (W)", from_=0.0, to=10.0, step=0.1, default_val=1.0, unit="mm",
            on_change=self._on_control_changed, style="Sidebar.TFrame"
        )
        self.ctrl_kw_w.pack(fill=tk.X)

        self.ctrl_kw_h = SynchronizedSliderEntry(
            parent, "Keyway Depth (H)", from_=0.0, to=5.0, step=0.05, default_val=0.5, unit="mm",
            on_change=self._on_control_changed, style="Sidebar.TFrame"
        )
        self.ctrl_kw_h.pack(fill=tk.X)

        self.ctrl_outer_diam = SynchronizedSliderEntry(
            parent, "Outer Blank Diameter (D_outer)", from_=10.0, to=500.0, step=1.0, default_val=50.0, unit="mm",
            on_change=self._on_control_changed, style="Sidebar.TFrame"
        )
        self.ctrl_outer_diam.pack(fill=tk.X)
        self.ctrl_outer_diam.pack_forget()

        ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        self.readout_box = ttk.LabelFrame(parent, text=" Live Engineering Readouts ", padding=8)
        self.readout_box.pack(fill=tk.X, pady=(0, 10))

        self.readout_widgets: Dict[str, ttk.Label] = {}
        metrics = [
            ("metric_1", "Pitch Diameter (d):"),
            ("metric_2", "Base / Gen Dia:"),
            ("metric_3", "Tip / Apex Dia (da):"),
            ("metric_4", "Root Diameter (df):"),
            ("metric_5", "Circular Pitch (p):"),
            ("metric_6", "Tooth Thickness (s):"),
            ("metric_7", "Thickness / Space:"),
            ("metric_8", "Bottom Clearance (c):"),
            ("metric_9", "NIHS Factor / Radius:"),
        ]

        self.readout_labels: Dict[str, ttk.Label] = {}
        for key, text_label in metrics:
            row = ttk.Frame(self.readout_box)
            row.pack(fill=tk.X, pady=1)
            name_lbl = ttk.Label(row, text=text_label, style="MetricName.TLabel")
            name_lbl.pack(side=tk.LEFT)
            self.readout_labels[key] = name_lbl
            val_lbl = ttk.Label(row, text="-- mm", style="MetricVal.TLabel")
            val_lbl.pack(side=tk.RIGHT)
            self.readout_widgets[key] = val_lbl

        self.compliance_badge_var = tk.StringVar(value="✓ NIHS 20-25 Compliant")
        self.lbl_compliance = ttk.Label(
            self.readout_box,
            textvariable=self.compliance_badge_var,
            font=("Segoe UI", 9, "bold"),
            foreground="#16A34A",
            padding=(0, 4, 0, 0),
        )
        self.lbl_compliance.pack(anchor="w")

        # CAD File Export Box with TXT profile options
        export_box = ttk.LabelFrame(parent, text=" CAD File Export ", padding=8)
        export_box.pack(fill=tk.X, pady=(0, 8))

        format_row = ttk.Frame(export_box)
        format_row.pack(fill=tk.X, pady=(0, 6))
        ttk.Label(format_row, text="Format:").pack(side=tk.LEFT)

        self.export_type_var = tk.StringVar(value="DXF (2D Profile - Laser/Wire-EDM)")
        self.combo_format = ttk.Combobox(
            format_row,
            textvariable=self.export_type_var,
            values=[
                "DXF (2D Profile - Laser/Wire-EDM)",
                "STEP (3D Solid B-Rep)",
                "TXT (2D Coordinates - X Y mm)",
                "TXT (3D Curve - SolidWorks/Fusion XYZ mm)"
            ],
            state="readonly",
            width=24,
        )
        self.combo_format.pack(side=tk.RIGHT)

        self.btn_export = ttk.Button(
            export_box,
            text="Generate & Export CAD File",
            style="Primary.TButton",
            command=self._execute_export,
        )
        self.btn_export.pack(fill=tk.X, ipady=3)

        self.status_msg = tk.StringVar(value="System ready.")
        self.lbl_status = ttk.Label(parent, textvariable=self.status_msg, foreground="#16A34A", wraplength=390)
        self.lbl_status.pack(fill=tk.X, pady=(4, 0))

    def _build_viewport(self, parent: ttk.Frame) -> None:
        top_bar = ttk.Frame(parent, padding=(10, 8, 10, 0))
        top_bar.pack(fill=tk.X)
        self.lbl_viewport_title = ttk.Label(top_bar, text="Interactive 2D Parametric CAD Viewport", font=("Segoe UI", 11, "bold"))
        self.lbl_viewport_title.pack(side=tk.LEFT)

        self.fig = Figure(figsize=(7, 7), dpi=100, facecolor="#F8FAFC")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.set_facecolor("#FFFFFF")

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()

        self.toolbar = NavigationToolbar2Tk(self.canvas, parent, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

    def _on_nihs_module_picked(self, event=None) -> None:
        try:
            m_val = float(self.nihs_mod_var.get())
            self.ctrl_m.set_value(m_val)
        except ValueError:
            pass

    def _on_custom_proportion_changed(self) -> None:
        # Automatically unlock strict enforcement when user changes kd or ks manually
        if self.enforce_nihs_var.get() and "Epicycloidal" in self.gear_type_var.get():
            self.enforce_nihs_var.set(False)
        self._on_control_changed()

    def _on_enforce_nihs_toggled(self) -> None:
        if self.enforce_nihs_var.get():
            is_wheel = "Wheel" in self.component_role_var.get()
            is_internal = "Internal" in self.gear_type_var.get()
            if is_wheel:
                self.ctrl_ks.set_value(1.41 / math.pi)
            else:
                self.ctrl_ks.set_value(0.42)
            self.ctrl_kd.set_value(1.40 if is_internal else 1.75)
        self._on_control_changed()

    def _on_tip_mode_changed(self) -> None:
        gtype = self.gear_type_var.get()
        if "Standard" in gtype:
            tip_mode = self.involute_tip_var.get()
            self.ctrl_tip_fillet.set_enabled("Corner-Filleted" in tip_mode)
        else:
            self.ctrl_tip_fillet.set_enabled(False)

        self._on_control_changed()

    def _on_gear_type_changed(self) -> None:
        gtype = self.gear_type_var.get()

        if "Epicycloidal" in gtype:
            self.std_involute_frame.pack_forget()
            self.asym_involute_frame.pack_forget()
            self.epicycloid_frame.pack(fill=tk.X)
            self.mod_frame.pack(fill=tk.X, pady=(0, 2))
            self.tip_panel.pack_forget()
            self.ctrl_kd.set_value(1.40 if "Internal" in gtype else 1.75)

            self.readout_labels["metric_2"].configure(text="Generating Circle (2rg):")
            self.readout_labels["metric_7"].configure(text="Tooth : Space (s : e):")
            self.readout_labels["metric_8"].configure(text="Bottom Clearance (c):")
            self.readout_labels["metric_9"].configure(text="NIHS Factor f / r_ogive:")
            self.lbl_compliance.pack(anchor="w")

        elif "Asymmetric" in gtype:
            self.std_involute_frame.pack_forget()
            self.epicycloid_frame.pack_forget()
            self.asym_involute_frame.pack(fill=tk.X)
            self.mod_frame.pack_forget()
            self.tip_panel.pack_forget()
            self.ctrl_kd.set_value(1.25)

            self.readout_labels["metric_2"].configure(text="Base Diameters (dbd / dbc):")
            self.readout_labels["metric_7"].configure(text="Top Land Thickness:")
            self.readout_labels["metric_8"].configure(text="Clearance (c):")
            self.readout_labels["metric_9"].configure(text="Drive / Coast Angle:")
            self.lbl_compliance.pack_forget()

        elif "Internal" in gtype:
            self.asym_involute_frame.pack_forget()
            self.epicycloid_frame.pack_forget()
            self.std_involute_frame.pack(fill=tk.X)
            self.mod_frame.pack_forget()
            self.tip_panel.pack_forget()
            self.ctrl_kd.set_value(1.25)

            self.readout_labels["metric_2"].configure(text="Base Diameter (db):")
            self.readout_labels["metric_7"].configure(text="Space Width / Void:")
            self.readout_labels["metric_8"].configure(text="Bottom Clearance (c):")
            self.readout_labels["metric_9"].configure(text="Pressure Angle (α):")
            self.lbl_compliance.pack_forget()

        else:
            self.asym_involute_frame.pack_forget()
            self.epicycloid_frame.pack_forget()
            self.std_involute_frame.pack(fill=tk.X)
            self.mod_frame.pack_forget()
            self.tip_panel.pack(fill=tk.X, pady=(8, 0))
            self.ctrl_kd.set_value(1.25)

            self.readout_labels["metric_2"].configure(text="Base Diameter (db):")
            self.readout_labels["metric_7"].configure(text="Top Land Thickness:")
            self.readout_labels["metric_8"].configure(text="Clearance (c):")
            self.readout_labels["metric_9"].configure(text="Pressure Angle (α):")
            self.lbl_compliance.pack_forget()

        if "Internal" in gtype:
            self.ctrl_outer_diam.pack(fill=tk.X, after=self.ctrl_b)
            self.ctrl_bore.pack_forget()
            self.chk_keyway.pack_forget()
            self.ctrl_kw_w.pack_forget()
            self.ctrl_kw_h.pack_forget()
        else:
            self.ctrl_outer_diam.pack_forget()
            self.ctrl_bore.pack(fill=tk.X, after=self.ctrl_b)
            self.chk_keyway.pack(anchor="w", pady=(2, 2), after=self.ctrl_bore)
            self.ctrl_kw_w.pack(fill=tk.X, after=self.chk_keyway)
            self.ctrl_kw_h.pack(fill=tk.X, after=self.ctrl_kw_w)

        self._on_control_changed()
        self._on_tip_mode_changed()

    def _on_role_changed(self) -> None:
        role = self.component_role_var.get()
        if "Pinion" in role:
            self.ctrl_z.set_value(12)
            self.ctrl_z_mate.set_value(60)
            self.root_form_var.set("U-Shaped Semicircular Root")
            if self.enforce_nihs_var.get():
                self.ctrl_ks.set_value(0.42)
                self.ctrl_kd.set_value(1.75)
        else:
            self.ctrl_z.set_value(60)
            self.ctrl_z_mate.set_value(12)
            self.root_form_var.set("U-Shaped Semicircular Root")
            if self.enforce_nihs_var.get():
                self.ctrl_ks.set_value(1.41 / math.pi)
                self.ctrl_kd.set_value(1.75)

        self._on_tip_mode_changed()

    def _get_specs_from_ui(self) -> GearConfig:
        return GearConfig(
            gear_type=self.gear_type_var.get(),
            module=self.ctrl_m.get_value(),
            teeth=int(self.ctrl_z.get_value()),
            face_width=self.ctrl_b.get_value(),
            bore_diameter=self.ctrl_bore.get_value(),
            outer_blank_diameter=self.ctrl_outer_diam.get_value(),
            backlash=self.ctrl_backlash.get_value(),
            enable_keyway=self.enable_keyway_var.get(),
            keyway_width=self.ctrl_kw_w.get_value(),
            keyway_depth=self.ctrl_kw_h.get_value(),
            pressure_angle_deg=self.ctrl_alpha.get_value(),
            involute_tip_form=self.involute_tip_var.get(),
            drive_pressure_angle_deg=self.ctrl_alpha_d.get_value(),
            coast_pressure_angle_deg=self.ctrl_alpha_c.get_value(),
            tip_fillet_factor=self.ctrl_tip_fillet.get_value(),
            auto_full_radius_tip=False,
            component_role=self.component_role_var.get(),
            enforce_nihs_proportions=self.enforce_nihs_var.get(),
            mating_teeth=int(self.ctrl_z_mate.get_value()),
            tooth_thickness_factor=self.ctrl_ks.get_value(),
            addendum_crest_form=self.crest_form_var.get(),
            root_profile_form=self.root_form_var.get(),
            dedendum_factor=self.ctrl_kd.get_value(),
        )

    def _on_control_changed(self) -> None:
        if self._debounce_timer:
            self.after_cancel(self._debounce_timer)
        self._debounce_timer = self.after(15, self._trigger_update)

    def _trigger_update(self) -> None:
        self._debounce_timer = None
        params = self._get_specs_from_ui()

        try:
            calc = UnifiedGearEngine.calculate_gear_geometry(params)
            self.calc_data = calc
            self.status_msg.set(f"{params.gear_type} geometry valid & CCW ordered (manifold safe).")
            self.lbl_status.configure(foreground="#16A34A")

            self.readout_widgets["metric_1"].configure(text=f"{calc['d']:.3f} mm")

            if "Epicycloidal" in params.gear_type:
                self.readout_widgets["metric_2"].configure(text=f"{calc['d_g']:.3f} mm")
                s_val = calc['tooth_thickness']
                e_val = calc['space_width']
                self.readout_widgets["metric_7"].configure(text=f"{s_val:.3f} : {e_val:.3f} ({calc['thickness_ratio']*100:.1f}%)")
                f_val = calc.get('f_correction', 2.50)
                rog_val = calc.get('ogive_radius', 1.0)
                self.readout_widgets["metric_9"].configure(text=f"f={f_val:.2f} | r={rog_val:.3f} mm")

                if calc["is_nihs_compliant"]:
                    self.compliance_badge_var.set("✓ NIHS 20-25 Compliant")
                    self.lbl_compliance.configure(foreground="#16A34A")
                else:
                    self.compliance_badge_var.set(f"⚠ {calc['nihs_notes']}")
                    self.lbl_compliance.configure(foreground="#D97706")

            elif "Asymmetric" in params.gear_type:
                self.readout_widgets["metric_2"].configure(text=f"{calc['d_bd']:.3f} / {calc['d_bc']:.3f} mm")
                self.readout_widgets["metric_7"].configure(text=f"{calc['top_land_thickness']:.3f} mm")
                self.readout_widgets["metric_9"].configure(text=f"{calc['drive_pressure_angle']:.1f}° / {calc['coast_pressure_angle']:.1f}°")

            else:
                self.readout_widgets["metric_2"].configure(text=f"{calc['d_b']:.3f} mm")
                self.readout_widgets["metric_7"].configure(text=f"{calc['top_land_thickness']:.3f} mm")
                self.readout_widgets["metric_9"].configure(text=f"α = {calc['pressure_angle']:.1f}°")

            self.readout_widgets["metric_3"].configure(text=f"{calc['d_a']:.3f} mm")
            self.readout_widgets["metric_4"].configure(text=f"{calc['d_f']:.3f} mm")
            self.readout_widgets["metric_5"].configure(text=f"{calc['circular_pitch']:.3f} mm")
            self.readout_widgets["metric_6"].configure(text=f"{calc['tooth_thickness']:.3f} mm")
            self.readout_widgets["metric_8"].configure(text=f"{calc['bottom_clearance']:.3f} mm")

            self._render_cad_preview(calc)

        except Exception as err:
            self.status_msg.set(f"Constraint Error: {err}")
            self.lbl_status.configure(foreground="#DC2626")

    def _render_cad_preview(self, data: Dict[str, Any]) -> None:
        self.ax.clear()
        self.ax.set_facecolor("#FFFFFF")
        self.ax.grid(True, linestyle=":", color="#CBD5E1", alpha=0.7)

        outer_pts = np.array(data["outer_perimeter"])
        bore_pts = np.array(data["bore_perimeter"]) if len(data["bore_perimeter"]) > 0 else None

        self.ax.add_patch(patches.Circle((0, 0), data["r"], fill=False, edgecolor="#0284C7", linestyle="--", linewidth=1.2, label="Pitch Circle (d)"))

        if "Epicycloidal" in data["gear_type"]:
            r_g = data["r_g"]
            rg_center_y = data["r"] + r_g
            self.ax.add_patch(patches.Circle((0, rg_center_y), r_g, fill=False, edgecolor="#10B981", linestyle=":", linewidth=1.2, label=f"Generating Circle (rg={r_g:.3f})"))
        elif "Asymmetric" in data["gear_type"]:
            self.ax.add_patch(patches.Circle((0, 0), data["r_bd"], fill=False, edgecolor="#F59E0B", linestyle=":", linewidth=1.0, label=f"Drive Base (rbd={data['r_bd']:.2f})"))
            self.ax.add_patch(patches.Circle((0, 0), data["r_bc"], fill=False, edgecolor="#8B5CF6", linestyle=":", linewidth=1.0, label=f"Coast Base (rbc={data['r_bc']:.2f})"))
        else:
            self.ax.add_patch(patches.Circle((0, 0), data["r_b"], fill=False, edgecolor="#94A3B8", linestyle=":", linewidth=1.0, label="Base Circle (db)"))

        self.ax.add_patch(patches.Circle((0, 0), data["r_f"], fill=False, edgecolor="#E11D48", linestyle=":", linewidth=0.9, label="Root Circle (df)"))

        if "Internal" in data["gear_type"] and bore_pts is not None and len(bore_pts) > 0:
            from matplotlib.path import Path
            verts = list(outer_pts) + [outer_pts[0]] + list(bore_pts)[::-1] + [bore_pts[-1]]
            codes = [Path.MOVETO] + [Path.LINETO]*(len(outer_pts)-1) + [Path.CLOSEPOLY] + \
                    [Path.MOVETO] + [Path.LINETO]*(len(bore_pts)-1) + [Path.CLOSEPOLY]

            path = Path(verts, codes)
            ring_patch = patches.PathPatch(path, facecolor="#E2E8F0", edgecolor="#0F172A", lw=1.4, zorder=3, label="Solid Ring")
            self.ax.add_patch(ring_patch)
            self.ax.plot(*bore_pts.T, color="#2563EB", linewidth=1.5, zorder=4, label="Internal Gear Profile")
        else:
            outer_poly = patches.Polygon(outer_pts, closed=True, facecolor="#E2E8F0", edgecolor="#0F172A", linewidth=1.4, zorder=3, label="Tooth Boundary")
            self.ax.add_patch(outer_poly)

            if bore_pts is not None and len(bore_pts) > 0:
                bore_poly = patches.Polygon(bore_pts, closed=True, facecolor="#FFFFFF", edgecolor="#2563EB", linewidth=1.5, zorder=4, label="Bore / Pivot")
                self.ax.add_patch(bore_poly)

        cross_len = data["r_a"] * 0.12
        self.ax.plot([-cross_len, cross_len], [0, 0], color="#64748B", linewidth=0.8, zorder=5)
        self.ax.plot([0, 0], [-cross_len, cross_len], color="#64748B", linewidth=0.8, zorder=5)

        margin = data["r_a"] * 1.2
        if "Epicycloidal" in data["gear_type"]:
            margin = max(margin, (data["r"] + 2.1 * data["r_g"]) * 1.05)
        elif "Internal" in data["gear_type"]:
            margin = (self.ctrl_outer_diam.get_value() / 2.0) * 1.15

        self.ax.set_xlim(-margin, margin)
        self.ax.set_ylim(-margin, margin)
        self.ax.set_xlabel("X [mm]", fontsize=9, color="#475569")
        self.ax.set_ylabel("Y [mm]", fontsize=9, color="#475569")

        if "Epicycloidal" in data["gear_type"]:
            title_text = f"NIHS 20-25 Epicycloidal {data['component_role']} (z={int(self.ctrl_z.get_value())}, m={self.ctrl_m.get_value():.2f}mm, f={data.get('f_correction', 2.5):.2f})"
        elif "Asymmetric" in data["gear_type"]:
            title_text = f"Asymmetric Involute Spur Gear (z={int(self.ctrl_z.get_value())}, m={self.ctrl_m.get_value():.2f}mm, αd={self.ctrl_alpha_d.get_value():.1f}°, αc={self.ctrl_alpha_c.get_value():.1f}°)"
        else:
            title_text = f"DIN 867 Involute Spur Gear (z={int(self.ctrl_z.get_value())}, m={self.ctrl_m.get_value():.2f}mm, α={self.ctrl_alpha.get_value():.1f}°)"

        self.ax.set_title(title_text, fontsize=10, fontweight="bold", color="#0F172A")
        self.ax.legend(loc="upper right", fontsize=7.5, framealpha=0.9)
        self.canvas.draw_idle()

    def _execute_export(self) -> None:
        if self.calc_data is None:
            messagebox.showerror("Export Error", "No valid gear geometry is available for export.")
            return

        choice = self.export_type_var.get()
        if "DXF" in choice:
            self._export_dxf()
        elif "STEP" in choice:
            self._export_step()
        else:
            self._export_txt()

    def _export_txt(self) -> None:
        data = self.calc_data
        z_val = int(self.ctrl_z.get_value())
        m_val = self.ctrl_m.get_value()
        choice = self.export_type_var.get()
        is_xyz = "XYZ" in choice

        gtype_raw = self.gear_type_var.get()
        gprefix = "nihs_epi" if "Epicycloidal" in gtype_raw else ("asym_inv" if "Asymmetric" in gtype_raw else "iso_inv")

        filepath = filedialog.asksaveasfilename(
            title="Export Profile Coordinates (TXT)",
            defaultextension=".txt",
            filetypes=[("Text File (*.txt)", "*.txt"), ("CSV File (*.csv)", "*.csv"), ("All Files (*.*)", "*.*")],
            initialfile=f"gear_{gprefix}_points_z{z_val}_m{m_val:.2f}.txt",
        )
        if not filepath:
            return

        try:
            outer_pts = data["outer_perimeter"]
            bore_pts = data.get("bore_perimeter", [])

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# Parametric Gear CAD Suite (Gear App V1.2) - Coordinate Export\n")
                f.write(f"# Discipline: {data['gear_type']}\n")
                f.write(f"# Module: {m_val:.3f} mm | Tooth Count (z): {z_val}\n")
                f.write(f"# Pitch Dia (d): {data['d']:.3f} mm | Tip Dia (da): {data['d_a']:.3f} mm | Root Dia (df): {data['d_f']:.3f} mm\n")
                f.write(f"# Face Width (b): {data['face_width']:.3f} mm\n")
                header_format = 'X [mm]\tY [mm]\tZ [mm]' if is_xyz else 'X [mm]\tY [mm]'
                f.write(f"# Format: {header_format}\n\n")

                f.write("# --- OUTER CONTOUR ---\n")
                for x, y in outer_pts:
                    if is_xyz:
                        f.write(f"{x:14.6f}\t{y:14.6f}\t{0.0:14.6f}\n")
                    else:
                        f.write(f"{x:14.6f}\t{y:14.6f}\n")

                if bore_pts:
                    f.write("\n# --- INNER BORE / INTERNAL PROFILE CONTOUR ---\n")
                    for x, y in bore_pts:
                        if is_xyz:
                            f.write(f"{x:14.6f}\t{y:14.6f}\t{0.0:14.6f}\n")
                        else:
                            f.write(f"{x:14.6f}\t{y:14.6f}\n")

            messagebox.showinfo("Export Successful", f"Coordinate profile successfully exported to:\n{filepath}")
        except Exception as err:
            messagebox.showerror("TXT Export Error", f"Failed to export TXT file: {err}")

    def _export_dxf(self) -> None:
        if not EZDXF_AVAILABLE:
            messagebox.showerror("Library Missing", "ezdxf is required for DXF export.\nRun: pip install ezdxf")
            return

        z_val = int(self.ctrl_z.get_value())
        m_val = self.ctrl_m.get_value()
        gtype_raw = self.gear_type_var.get()
        if "Epicycloidal" in gtype_raw:
            gtype = "nihs_epi"
        elif "Asymmetric" in gtype_raw:
            gtype = "asym_inv"
        else:
            gtype = "iso_inv"

        filepath = filedialog.asksaveasfilename(
            title=f"Export 2D {self.gear_type_var.get()} Profile (DXF)",
            defaultextension=".dxf",
            filetypes=[("AutoCAD DXF File", "*.dxf"), ("All Files", "*.*")],
            initialfile=f"gear_{gtype}_z{z_val}_m{m_val:.2f}.dxf",
        )
        if not filepath:
            return

        try:
            data = self.calc_data
            doc = ezdxf.new("R2010", setup=True)
            doc.units = dxf_units.MM
            msp = doc.modelspace()

            doc.layers.add(name="CONTOUR", color=1)
            doc.layers.add(name="BORE", color=3)
            doc.layers.add(name="PITCH_CIRCLE", color=4)
            doc.layers.add(name="ROOT_CIRCLE", color=6)

            outer_closed = data["outer_perimeter"] + [data["outer_perimeter"][0]]

            if "Internal" in data["gear_type"]:
                doc.layers.add(name="OUTER_RING", color=7)
                msp.add_lwpolyline(outer_closed, dxfattribs={"layer": "OUTER_RING", "closed": True})
                if data["bore_perimeter"]:
                    bore_closed = data["bore_perimeter"] + [data["bore_perimeter"][0]]
                    msp.add_lwpolyline(bore_closed, dxfattribs={"layer": "CONTOUR", "closed": True})
            else:
                msp.add_lwpolyline(outer_closed, dxfattribs={"layer": "CONTOUR", "closed": True})
                if data["bore_perimeter"]:
                    bore_closed = data["bore_perimeter"] + [data["bore_perimeter"][0]]
                    msp.add_lwpolyline(bore_closed, dxfattribs={"layer": "BORE", "closed": True})

            msp.add_circle((0.0, 0.0), data["r"], dxfattribs={"layer": "PITCH_CIRCLE"})
            msp.add_circle((0.0, 0.0), data["r_f"], dxfattribs={"layer": "ROOT_CIRCLE"})

            if "Epicycloidal" in data["gear_type"]:
                doc.layers.add(name="GEN_CIRCLE", color=2)
                msp.add_circle((0.0, data["r"] + data["r_g"]), data["r_g"], dxfattribs={"layer": "GEN_CIRCLE"})
            elif "Asymmetric" in data["gear_type"]:
                doc.layers.add(name="DRIVE_BASE", color=2)
                doc.layers.add(name="COAST_BASE", color=5)
                msp.add_circle((0.0, 0.0), data["r_bd"], dxfattribs={"layer": "DRIVE_BASE"})
                msp.add_circle((0.0, 0.0), data["r_bc"], dxfattribs={"layer": "COAST_BASE"})
            else:
                doc.layers.add(name="BASE_CIRCLE", color=8)
                msp.add_circle((0.0, 0.0), data["r_b"], dxfattribs={"layer": "BASE_CIRCLE"})

            doc.saveas(filepath)
            messagebox.showinfo("Export Successful", f"2D DXF CAD profile successfully exported to:\n{filepath}")
        except Exception as err:
            messagebox.showerror("DXF Export Error", f"Failed to export DXF: {err}")

    def _export_step(self) -> None:
        if not CADQUERY_AVAILABLE:
            messagebox.showerror("Library Missing", "cadquery is required for STEP solid export.\nRun: pip install cadquery")
            return

        z_val = int(self.ctrl_z.get_value())
        m_val = self.ctrl_m.get_value()
        b_val = self.ctrl_b.get_value()
        gtype_raw = self.gear_type_var.get()
        if "Epicycloidal" in gtype_raw:
            gtype = "nihs_epi"
        elif "Asymmetric" in gtype_raw:
            gtype = "asym_inv"
        else:
            gtype = "iso_inv"

        filepath = filedialog.asksaveasfilename(
            title=f"Export 3D {self.gear_type_var.get()} Solid (STEP)",
            defaultextension=".step",
            filetypes=[("STEP CAD Solid", "*.step;*.stp"), ("All Files", "*.*")],
            initialfile=f"gear_{gtype}_z{z_val}_m{m_val:.2f}_b{b_val:.1f}.step",
        )
        if not filepath:
            return

        try:
            data = self.calc_data
            b = float(data["face_width"])
            r_bore = float(data["r_bore"])
            outer_pts = data["outer_perimeter"]

            if "Internal" in data["gear_type"]:
                m_calc = data["circular_pitch"] / math.pi
                margin = 6.0 if "Epicycloidal" in data["gear_type"] else 20.0
                r_outer = (data["d"] + margin * m_calc) / 2.0
                if self.ctrl_outer_diam.get_value() / 2.0 > r_outer:
                    r_outer = self.ctrl_outer_diam.get_value() / 2.0
                gear_solid = cq.Workplane("XY").circle(r_outer).extrude(b)
            else:
                gear_solid = cq.Workplane("XY").polyline(outer_pts).close().extrude(b)

            if r_bore > 0 or data.get("bore_perimeter"):
                if not data.get("enable_keyway", False) and not data.get("bore_perimeter") and r_bore > 0:
                    gear_solid = gear_solid.faces(">Z").workplane().circle(r_bore).cutThruAll()
                elif data.get("bore_perimeter"):
                    bore_pts = data["bore_perimeter"]
                    bore_cutter = (
                        cq.Workplane("XY")
                        .polyline(bore_pts)
                        .close()
                        .extrude(b * 2.0)
                        .translate((0.0, 0.0, -b * 0.5))
                    )
                    gear_solid = gear_solid.cut(bore_cutter)

            cq.exporters.export(gear_solid, filepath)
            messagebox.showinfo("Export Successful", f"3D STEP CAD Solid successfully exported to:\n{filepath}")
        except Exception as err:
            messagebox.showerror("STEP Export Error", f"Failed to export STEP: {err}")


if __name__ == "__main__":
    app = GearCADApplication()
    app.mainloop()
