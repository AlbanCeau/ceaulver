"""Overlap-volume gradient diagnostic (Feng-2021 step 1).

Validates the single geometric identity the whole energy-conserving contact model
rests on (Feng 2021, CMAME 113454, framework Eqs. 26/28; contact-volume specialisation):

    a_c := sum of OUTWARD vector areas of the overlap-polyhedron faces that come
           from body A's planes   ==   d V_c / d (position of A)

where V_c is the volume of the intersection A ∩ B. If this holds, then for any
energy w(V_c) the contact force F = -w'(V_c) a_c is *exactly* -∇w -> conservative
by construction, with the normal n = a_c/|a_c| continuous across feature changes.

Pure numpy, correctness over speed (220-triple vertex enumeration).

Run:  uv run python -m ceaulver.reference
"""

import numpy as np

from . import config   # constants only: no Taichi import, no GPU

# --- tolerances (positions ~O(1), half-extents ~1) -------------------------
DET_EPS  = 1e-9    # reject near-parallel plane triples
FEAS_EPS = 1e-7    # "inside all half-spaces" slack
ONPL_EPS = 1e-6    # "lies on this plane" slack
DEDUP_EPS = 1e-6   # merge coincident vertices (>3 planes meet)


def rot_xyz(ax, ay, az):
    """Rotation matrix R (columns = body axes), intrinsic X then Y then Z."""
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def obb_planes(p, R, he):
    """6 half-spaces of an OBB. Each: (n_outward, d) with interior {x: n.x <= d}."""
    planes = []
    for i in range(3):
        a = R[:, i]
        planes.append((a.copy(),  a @ p + he[i]))   # +a face
        planes.append((-a.copy(), -a @ p + he[i]))  # -a face
    return planes


def intersection_polyhedron(planes):
    """Vertex-enumerate the convex polyhedron {x : n_k.x <= d_k}.

    Returns (faces, verts) where faces[k] is the list of vertex indices lying on
    plane k (unordered), and verts is an (M,3) array. Empty if no intersection.
    """
    N = np.array([pl[0] for pl in planes])          # (P,3)
    D = np.array([pl[1] for pl in planes])          # (P,)
    P = len(planes)
    scale = max(1.0, float(np.max(np.abs(D))))

    raw = []
    for a in range(P):
        for b in range(a + 1, P):
            for c in range(b + 1, P):
                M = np.stack([N[a], N[b], N[c]])
                if abs(np.linalg.det(M)) < DET_EPS:
                    continue
                x = np.linalg.solve(M, np.array([D[a], D[b], D[c]]))
                if np.all(N @ x <= D + FEAS_EPS * scale):
                    raw.append(x)
    if not raw:
        return None, None

    # dedupe coincident vertices (a corner where >3 planes meet recurs)
    raw = np.array(raw)
    verts = []
    for x in raw:
        if not verts or np.min(np.linalg.norm(np.array(verts) - x, axis=1)) > DEDUP_EPS:
            verts.append(x)
    verts = np.array(verts)

    # assign each vertex to every plane it lies on
    faces = [[] for _ in range(P)]
    for vi, x in enumerate(verts):
        on = np.where(np.abs(N @ x - D) < ONPL_EPS * scale)[0]
        for k in on:
            faces[k].append(vi)
    return faces, verts


def order_ccw(pts, n):
    """Order coplanar points CCW about outward normal n. Returns ordered (m,3)."""
    c = pts.mean(0)
    # tangent basis from the world axis least aligned with n
    u0 = np.eye(3)[np.argmin(np.abs(n))]
    u = u0 - (u0 @ n) * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    rel = pts - c
    ang = np.arctan2(rel @ v, rel @ u)
    return pts[np.argsort(ang)]


def vector_area(ordered, n):
    """Outward vector area A*n of an ordered planar polygon (origin-independent)."""
    a = 0.5 * np.sum(np.cross(ordered, np.roll(ordered, -1, axis=0)), axis=0)
    return a if a @ n >= 0 else -a   # align to outward normal


def coincident_mask(planes):
    """Flag planes shared by A (0..5) and B (6..11).

    When A and B share half-extents/center on an axis their supporting planes
    coincide exactly (e.g. axis-aligned equal cubes). Such a face is a *flush*
    boundary, not a penetration interface: it must be counted ONCE in the volume
    and excluded from both contact caps.
    """
    shared = [False] * len(planes)
    for i in range(6):
        ni, di = planes[i]
        for j in range(6, 12):
            nj, dj = planes[j]
            if np.allclose(ni, nj, atol=1e-7) and abs(di - dj) < 1e-7:
                shared[i] = shared[j] = True
    return shared


def overlap_volume_and_caps(pA, RA, heA, pB, RB, heB):
    """Return (V_c, a_c_A, a_c_B): overlap volume + outward cap vector areas.

    a_c_A = sum of outward vector areas of A-plane faces (the claimed dV/dpA).
    a_c_B = same for B-plane faces (should equal -a_c_A: closed surface => sum 0).
    """
    planes = obb_planes(pA, RA, heA) + obb_planes(pB, RB, heB)  # 0..5 = A, 6..11 = B
    faces, verts = intersection_polyhedron(planes)
    if faces is None:
        return 0.0, np.zeros(3), np.zeros(3)

    shared = coincident_mask(planes)

    V = 0.0
    aA = np.zeros(3)
    aB = np.zeros(3)
    a_all = np.zeros(3)   # full closed-surface vector area (closure check, tag-free)
    for k, vidx in enumerate(faces):
        if len(vidx) < 3:
            continue
        n = planes[k][0]
        ordered = order_ccw(verts[vidx], n)
        af = vector_area(ordered, n)
        a_all += af
        # contact caps: only non-shared planes are genuine penetration interfaces
        if not shared[k]:
            if k < 6:
                aA += af
            else:
                aB += af
        # volume: count each geometric face once -> a shared B-plane is skipped
        # (its coincident A-plane partner is the representative).
        if shared[k] and k >= 6:
            continue
        # divergence theorem: (1/6) sum over outward face-triangles
        w0 = ordered[0]
        for t in range(1, len(ordered) - 1):
            V += w0 @ np.cross(ordered[t], ordered[t + 1]) / 6.0
    return V, aA, aB, a_all


def grad_fd(pA, RA, heA, pB, RB, heB, eps=1e-5):
    """Central finite-difference of V_c w.r.t. translating A, per world axis."""
    g = np.zeros(3)
    for d in range(3):
        e = np.zeros(3); e[d] = eps
        Vp = overlap_volume_and_caps(pA + e, RA, heA, pB, RB, heB)[0]
        Vm = overlap_volume_and_caps(pA - e, RA, heA, pB, RB, heB)[0]
        g[d] = (Vp - Vm) / (2 * eps)
    return g


def run_case(name, pA, RA, heA, pB, RB, heB):
    V, aA, aB, a_all = overlap_volume_and_caps(pA, RA, heA, pB, RB, heB)
    g = grad_fd(pA, RA, heA, pB, RB, heB)
    err = np.linalg.norm(aA - g)
    closure = np.linalg.norm(a_all)   # full closed surface => sum of areas ~0
    print(f"\n=== {name} ===")
    print(f"  V_c              = {V:.6f}")
    print(f"  a_c (A caps)     = [{aA[0]:+.5f} {aA[1]:+.5f} {aA[2]:+.5f}]")
    print(f"  dV/dpA (central) = [{g[0]:+.5f} {g[1]:+.5f} {g[2]:+.5f}]")
    print(f"  |a_c - dV/dpA|   = {err:.2e}   {'PASS' if err < 1e-4 else 'FAIL'}")
    print(f"  |a_c_A + a_c_B|  = {closure:.2e}   (surface closure, want ~0)")
    if V > 0:
        n = aA / np.linalg.norm(aA)
        print(f"  effective normal = [{n[0]:+.3f} {n[1]:+.3f} {n[2]:+.3f}]"
              f"   eff. area = {np.linalg.norm(aA):.4f}")


def main_gradient():
    he = np.array([1.0, 1.0, 1.0])
    I = np.eye(3)

    # 1. axis-aligned face-face overlap (the tilt=0 case the 2D clip mishandled)
    run_case("face-face, aligned (overlap 0.5 in x)",
             np.array([1.5, 0.0, 0.0]), I, he,
             np.array([0.0, 0.0, 0.0]), I, he)

    # 2. tilted contact (obliquity => 2D clip injected; here should be exact)
    run_case("tilted 30deg about z, offset (1.4, 0.3, 0)",
             np.array([1.4, 0.3, 0.0]), rot_xyz(0, 0, np.radians(30)), he,
             np.array([0.0, 0.0, 0.0]), I, he)

    # 3. doubly-tilted, off in all axes (multi cap-face, generic)
    run_case("tilted 30z+20y, offset (1.3, 0.4, 0.5)",
             np.array([1.3, 0.4, 0.5]), rot_xyz(0, np.radians(20), np.radians(30)), he,
             np.array([0.0, 0.0, 0.0]), I, he)

    # 4. corner-first: A rotated 45/45 driving a vertex into B's face
    run_case("corner-first 45z+35y, offset (1.2, 0.0, 0.0)",
             np.array([1.2, 0.0, 0.0]), rot_xyz(0, np.radians(35), np.radians(45)), he,
             np.array([0.0, 0.0, 0.0]), I, he)


# ==================================================================
# Feng overlap-volume contact: 2-cube elastic conservation test (step 2).
#
# Builds the energy-conserving contact FORCE on top of the validated geometric
# core above (all four gradient checks PASS). The model:
#
#     contact energy  w(V_c) = 2/5 kH V_c^2.5         (any differentiable w(V) conserves)
#     stiffness       kH = FENG_KHERTZ / (AcA * AcB),  Ac = mean face area of the box
#     force on A      F_A  = -w'(V_c) dV/dpA   = -kH V_c^1.5 a_c
#     torque on A     tau_A = -w'(V_c) dV/dthetaA = -kH V_c^1.5 * sum_{A cap} (c_f - pA) x a_f
#     on B            F_B = -F_A ;  tau_B = -kH V_c^1.5 * sum_{B cap} (c_f - pB) x a_f
#
# Because (F, tau) = -grad w of a SINGLE scalar of both bodies' configs, linear and
# angular momentum are conserved exactly, and energy is conserved up to time-
# integration error only (made negligible here with a soft, well-resolved spring).
#
# This is the decisive test vs the old 2D Sutherland-Hodgman clip, which injected
# +7.9% (flat) to -12% (oblique) KE in the same elastic 2-cube collision. If the 3D model holds KE to ~round-off across the tilt/offset
# sweep, the structural injection is gone.
#
# No gravity, no walls, no damping, no friction -- pure elastic, so KE before
# contact must equal KE after separation.
# ==================================================================

def contact_features(pA, RA, heA, pB, RB, heB):
    """Geometric gradients of the overlap volume V_c.

    Returns (V, a_c, gA, gB, cc):
      V    overlap volume
      a_c  = dV/dpA   = sum of outward vector areas of A's non-shared cap faces
      gA   = dV/dthetaA = sum_{A cap} (r - pA) x n dA  (exact, per triangle)
      gB   = dV/dthetaB = sum_{B cap} (r - pB) x n dA
      cc   centroid of the overlap volume (the point the dashpot acts at)
    """
    planes = obb_planes(pA, RA, heA) + obb_planes(pB, RB, heB)  # 0..5 A, 6..11 B
    faces, verts = intersection_polyhedron(planes)
    if faces is None:
        return 0.0, np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
    shared = coincident_mask(planes)
    verts = verts - pA   # local frame centred on A, as _feng_faces does

    V = 0.0
    cvol = np.zeros(3)
    a_c = np.zeros(3)
    gA = np.zeros(3)
    gB = np.zeros(3)
    for k, vidx in enumerate(faces):
        if len(vidx) < 3:
            continue
        n = planes[k][0]
        ordered = order_ccw(verts[vidx], n)
        w0 = ordered[0]
        # volume: count each geometric face once (skip the shared B-plane partner)
        if not (shared[k] and k >= 6):
            for t in range(1, len(ordered) - 1):
                tv = w0 @ np.cross(ordered[t], ordered[t + 1]) / 6.0
                V += tv
                cvol += tv * (w0 + ordered[t] + ordered[t + 1]) * 0.25
        if shared[k]:
            continue   # flush boundary -> not a contact cap
        ref = np.zeros(3) if k < 6 else pB - pA
        for t in range(1, len(ordered) - 1):
            a_tri = 0.5 * np.cross(ordered[t] - w0, ordered[t + 1] - w0)   # outward
            c_tri = (w0 + ordered[t] + ordered[t + 1]) / 3.0
            g = np.cross(c_tri - ref, a_tri)
            if k < 6:
                a_c += a_tri
                gA += g
            else:
                gB += g
    cc = pA + cvol / V if V > 0.0 else np.zeros(3)
    return V, a_c, gA, gB, cc


def _mean_face_area(he):
    return (4.0 / 3.0) * (he[0] * he[1] + he[1] * he[2] + he[2] * he[0])


def _max_face_area(he):
    return 4.0 * max(he[0] * he[1], he[1] * he[2], he[2] * he[0])


def box_inertia_inv_world(m, he, R):
    """World-frame inverse inertia tensor of a box (I_k = m/3 * sum of the other two he^2)."""
    I = m / 3.0 * np.array([he[1]**2 + he[2]**2, he[0]**2 + he[2]**2, he[0]**2 + he[1]**2])
    return R @ np.diag(1.0 / I) @ R.T


def pair_force(pA, RA, heA, vA, wA, mA, IiA, pB, RB, heB, vB, wB, mB, IiB, dt, *,
               khertz=None, d_hertz=None, zeta_n=None, vmax=None, cap_conservative=None):
    """Block-block NORMAL law of contact._feng_forces, line for line. Friction is not included.

    IiA, IiB are world-frame inverse inertia tensors. Tunables default to config.
    Returns (F_A, tau_A, tau_B, pe, V); F_B = -F_A. pe is the elastic potential w(V).
    """
    khertz  = config.FENG_KHERTZ  if khertz  is None else khertz
    d_hertz = config.D_HERTZ      if d_hertz is None else d_hertz
    zeta_n  = config.ZETA_N       if zeta_n  is None else zeta_n
    vmax    = config.VMAX_CONTACT if vmax    is None else vmax
    if cap_conservative is None:
        cap_conservative = getattr(config, "CAP_CONSERVATIVE", 0.0)

    Vc, a_c, gA, gB, cc = contact_features(pA, RA, heA, pB, RB, heB)
    z = np.zeros(3)
    if Vc <= 0.0:
        return z, z, z, 0.0, 0.0

    # size-normalized Hertz-volume stiffness: w(V) = 2/5 kH V^2.5, w' = kH V^1.5
    kH = khertz / (_mean_face_area(heA) * _mean_face_area(heB))
    wprime = kH * Vc * np.sqrt(Vc)
    pe = 0.4 * kH * Vc**2.5
    cons = cap_conservative > 0.5
    if cons:   # T-11: saturate w'(V) itself -> still an exact gradient
        amax = _max_face_area(heA) if mA <= mB else _max_face_area(heB)
        wmax = vmax * min(mA, mB) / dt / amax
        if wprime > wmax:
            wprime = wmax
            Vs = (wmax / kH) ** (2.0 / 3.0)            # w is linear in V past V*
            pe = 0.4 * kH * Vs**2.5 + wmax * (Vc - Vs)
    F = -wprime * a_c
    tA = -wprime * gA
    tB = -wprime * gB
    # deep-overlap safeguard (a force rescale: NOT a gradient, pe above ignores it)
    fm = np.linalg.norm(F)
    fcap = vmax * min(mA, mB) / dt
    if fm > fcap and not cons:
        sc = fcap / fm
        F = F * sc; tA = tA * sc; tB = tB * sc

    an = np.linalg.norm(a_c)
    if an > 1e-9:
        nhat = a_c / an
        ra = cc - pA; rb = cc - pB
        va = vA + np.cross(wA, ra)
        vb = vB + np.cross(wB, rb)
        vn = (va - vb) @ nhat
        rxna = np.cross(ra, nhat); rxnb = np.cross(rb, nhat)
        me = 1.0 / (1.0 / mA + rxna @ IiA @ rxna + 1.0 / mB + rxnb @ IiB @ rxnb)
        # P&S (2.12) point dashpot, repulsive-only
        De = d_hertz * np.sqrt(Vc) * np.sqrt(me)
        Fd = min(De * abs(vn), me * abs(vn) / dt)
        if vn > 0:
            Fd = -Fd
        else:
            Fd = min(Fd, wprime * an)
        # Rayleigh dashpot on dV_c/dt, replaces the point dashpot when zeta_n > 0
        useR = 1.0 if zeta_n > 0.0 else 0.0
        Vdot = a_c @ (vA - vB) + gA @ wA + gB @ wB
        invV = an * an * (1.0 / mA + 1.0 / mB) + gA @ IiA @ gA + gB @ IiB @ gB
        mV = 1.0 / max(invV, 1e-12)
        kV = 1.5 * kH * np.sqrt(Vc)
        cR = min(2.0 * zeta_n * np.sqrt(kV * mV), mV / dt)
        fV = max(cR * Vdot, -wprime)
        Fdv = Fd * nhat * (1.0 - useR)
        F = F + Fdv - (useR * fV) * a_c
        tA = tA + np.cross(ra, Fdv) - (useR * fV) * gA
        tB = tB + np.cross(rb, -Fdv) - (useR * fV) * gB
    return F, tA, tB, pe, Vc


def contact_force(pA, RA, he, pB, RB, he2, m, Is, dt):
    """Elastic part only (dampers off): (F_A, tau_A, F_B, tau_B, pe, V)."""
    z = np.zeros(3)
    Ii = np.eye(3) / Is
    F_A, tau_A, tau_B, pe, V = pair_force(pA, RA, he, z, z, m, Ii, pB, RB, he2, z, z, m, Ii, dt,
                                          d_hertz=0.0, zeta_n=0.0)
    return F_A, tau_A, -F_A, tau_B, pe, V


def rodrigues(w, dt):
    """Exact rotation matrix for angular velocity w over dt."""
    n = np.linalg.norm(w)
    if n * dt < 1e-14:
        return np.eye(3)
    ax = w / n
    th = n * dt
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def momentum(m, Is, pA, vA, wA, pB, vB, wB):
    P = m * (vA + vB)
    L = (m * np.cross(pA, vA) + Is * wA) + (m * np.cross(pB, vB) + Is * wB)
    return P, L


def run(tilt_deg, offy, v0=3.0, he=1.0, m=1.0, dt=1e-4, max_steps=60000):
    he3 = np.array([he, he, he])
    Is = m / 6.0 * (2 * he) ** 2            # isotropic inertia of a cube

    pA = np.array([-1.5, offy, 0.0]); vA = np.array([v0, 0.0, 0.0]); wA = np.zeros(3)
    pB = np.array([1.5, 0.0, 0.0]);  vB = np.array([-v0, 0.0, 0.0]); wB = np.zeros(3)
    RA = rot_xyz(0, 0, np.radians(tilt_deg)); RB = np.eye(3)

    F_A, tA, F_B, tB, pe, V = contact_force(pA, RA, he3, pB, RB, he3, m, Is, dt)
    ke0 = 0.5 * m * (vA @ vA + vB @ vB) + 0.5 * Is * (wA @ wA + wB @ wB)
    P0, L0 = momentum(m, Is, pA, vA, wA, pB, vB, wB)
    E0 = ke0 + pe
    Emin = Emax = E0
    dP = dL = 0.0
    contacted = False

    for _ in range(max_steps):
        vA += F_A / m * dt / 2; wA += tA / Is * dt / 2
        vB += F_B / m * dt / 2; wB += tB / Is * dt / 2
        pA += vA * dt; pB += vB * dt
        RA = rodrigues(wA, dt) @ RA; RB = rodrigues(wB, dt) @ RB
        F_A, tA, F_B, tB, pe, V = contact_force(pA, RA, he3, pB, RB, he3, m, Is, dt)
        vA += F_A / m * dt / 2; wA += tA / Is * dt / 2
        vB += F_B / m * dt / 2; wB += tB / Is * dt / 2

        ke = 0.5 * m * (vA @ vA + vB @ vB) + 0.5 * Is * (wA @ wA + wB @ wB)
        E = ke + pe
        Emin = min(Emin, E); Emax = max(Emax, E)
        P, L = momentum(m, Is, pA, vA, wA, pB, vB, wB)
        dP = max(dP, np.linalg.norm(P - P0)); dL = max(dL, np.linalg.norm(L - L0))

        if V > 0:
            contacted = True
        elif contacted:
            break   # collision complete, bodies separated

    ke_after = 0.5 * m * (vA @ vA + vB @ vB) + 0.5 * Is * (wA @ wA + wB @ wB)
    drop = (ke0 - ke_after) / ke0 * 100.0   # +loss / -GAIN(injection)
    return dict(drop=drop, ke0=ke0, ke1=ke_after, dP=dP, dL=dL,
                dE=Emax - Emin, contacted=contacted)


def main_two_cube():
    print(f"Feng overlap-volume 2-cube elastic test (FENG_KHERTZ={config.FENG_KHERTZ:.3g}, dt=1e-4)")
    print("drop% = (KE_before - KE_after)/KE_before * 100   "
          "[+ = loss, - = INJECTION]\n")

    print("--- tilt sweep (offy=0): static obliquity, the 2D-clip injector ---")
    print(f"{'tilt':>5} {'drop%':>9} {'max|dP|':>10} {'max|dL|':>10} {'dE band':>10}")
    for tilt in (0, 5, 15, 30, 44):
        r = run(tilt, offy=0.0)
        print(f"{tilt:>5} {r['drop']:>9.4f} {r['dP']:>10.2e} "
              f"{r['dL']:>10.2e} {r['dE']:>10.2e}"
              f"{'' if r['contacted'] else '   (NO CONTACT)'}")

    print("\n--- offy sweep (tilt=0): induced spin, translation<->rotation ---")
    print(f"{'offy':>5} {'drop%':>9} {'max|dP|':>10} {'max|dL|':>10} {'dE band':>10}")
    for offy in (0.0, 0.1, 0.3, 0.5, 0.8):
        r = run(0, offy=offy)
        print(f"{offy:>5} {r['drop']:>9.4f} {r['dP']:>10.2e} "
              f"{r['dL']:>10.2e} {r['dE']:>10.2e}"
              f"{'' if r['contacted'] else '   (NO CONTACT)'}")


if __name__ == "__main__":
    main_gradient()
    print()
    main_two_cube()
