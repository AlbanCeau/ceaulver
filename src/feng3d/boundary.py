"""Boundary contact: the ground (Feng overlap volume), slope / wall / cylinder / box (per-corner
penalty), and force assembly."""

import taichi as ti

from .params import (
    PRM, _P_CAP_CONSERVATIVE, _P_D_HERTZ, _P_FENG_KHERTZ, _P_GRAVITY, _P_GROUND_Y, _P_PEN_MAX,
    _P_SLOPE_D, _P_SLOPE_K, _P_SLOPE_MU, _P_SLOPE_MU_HI, _P_SLOPE_MU_LO, _P_SOIL_MU, _P_STIFFNESS_ALPHA,
    _P_VMAX_CONTACT, _P_ZETA_N,
)
from .fields import (
    F_f, Iil_f, Iiw_f, MAX_N, R_f, _CS, active, box_on, box_w, cnt_n_f, cntct_f, cyl_R_f,
    cyl_cx_f, cyl_cz_f, cyl_on, dbg_on, dbg_pclm, dbg_pcor, fsq_f, gnd_f, he_f, ledge_on,
    ledge_x0, ledge_x1, ledge_z0, ledge_z1, mass_f, n_above, n_f, noslope_f, omega_f, pos_f,
    side_on, side_w, slope_d_f, slope_mu_y_hi, slope_mu_y_lo, slope_n_f, slope_o_f,
    slope_feng_on, slope_on, slope_t_f, slope_umin_f, slp_f, tau_f, vel_f, wall_d_f, wall_n_f, wall_on,
)
from .friction import (
    kinetic_friction, plane_friction, plane_spring_release,
)

@ti.func
def corner_w(ci: int, k: int) -> ti.math.vec3:
    s = _CS[k]
    return R_f[ci] @ ti.Vector([s[0]*he_f[ci][0], s[1]*he_f[ci][1], s[2]*he_f[ci][2]]) + pos_f[ci]

# ==================================================================
# Plane contact
# ==================================================================

@ti.func
def plane_force(ci: int, pn: ti.math.vec3, pd: ti.f32, dt: ti.f32,
                K: ti.f32, dc: ti.f32, mu: ti.f32, pid: ti.template()):
    # Multi-point (per-corner) penalty contact. Each penetrating corner carries
    # its own spring K*pen along pn, applied AT the corner -- so the total force
    # is the gradient of U = sum_k 1/2 K pen_k^2, which a single *averaged*
    # contact point is not. That keeps the normal contact conservative under
    # rocking (no spurious rotational-energy injection). Cundall & Strack 1979
    # get away with one point only because discs put the normal through the
    # centre (zero normal-torque); a box corner has a lever arm, so it needs the
    # per-corner manifold. Damping/friction are evaluated per corner too.
    Fo = ti.Vector([0.0, 0.0, 0.0]); To = ti.Vector([0.0, 0.0, 0.0])
    pen_max = 0.0; ok = 0
    for k in ti.static(range(8)):
        c = corner_w(ci, k)
        pen = pd - c.dot(pn)
        if pen > 0:
            pen = ti.min(pen, PRM[_P_PEN_MAX])
            ok = 1
            if dbg_on[None] == 1: ti.atomic_add(dbg_pcor[None], 1)
            if pen > pen_max: pen_max = pen
            rc = c - pos_f[ci]
            vc = vel_f[ci] + omega_f[ci].cross(rc)
            vn = vc.dot(pn)
            rxn = rc.cross(pn)
            me = 1.0 / (1.0/mass_f[ci] + rxn.dot(Iiw_f[ci] @ rxn))
            Ke = ti.min(K, PRM[_P_STIFFNESS_ALPHA] * me / (dt*dt))
            if dbg_on[None] == 1:
                if PRM[_P_STIFFNESS_ALPHA] * me / (dt*dt) < K:
                    ti.atomic_add(dbg_pclm[None], 1)
            Fs = Ke * pen
            De = dc * ti.sqrt(Ke * me)
            Fd = ti.min(De * ti.abs(vn), me * ti.abs(vn) / dt)
            if vn > 0: Fd = -Fd
            Fn = ti.max(0.0, Fs + Fd)
            Ffr = plane_friction(ci, pid, k, dt, pn, rc, vc, vn, me, Ke, Fn, mu)
            f_k = Fn * pn + Ffr
            Fo += f_k
            To += rc.cross(f_k)
        else:
            plane_spring_release(ci, pid, k)
    return Fo, To, pen_max, ok

# ==================================================================
# Plane contact, Feng overlap volume (the ground)
# ==================================================================
# The block-block law against a half-space: V = volume of the box below the plane,
# w(V) = (2/5) kH V^2.5, F = -w'(V) dV/dx, tau = -w'(V) dV/dtheta. With S = dV/dx and
# G = dV/dtheta as integrals of n and r x n over the submerged part of the box surface,
# each of the box's 12 surface triangles is clipped at the plane and contributes its
# submerged area A and first moment M (about the block centre): S = sum A n_f,
# G = sum M x n_f. V comes from the same areas by the divergence theorem about a point
# on the plane, where the cut face contributes nothing: V = 1/3 sum A (n_f . (x_f - o)).
# A face lying flush on the plane is no special case: its triangles are either in or out.

@ti.func
def _tri_below(a: ti.math.vec3, b: ti.math.vec3, c: ti.math.vec3,
               da: ti.f32, db: ti.f32, dc: ti.f32, area: ti.f32):
    # Part of triangle abc with depth d > 0: (area, area * centroid). Branch-free: the
    # vertex on its own side ("odd": the one inside if 1 is, the one outside if 2 are)
    # cuts off a sub-triangle; the result is it (1 inside) or the rest (2 inside).
    ia = ti.select(da > 0.0, 1.0, 0.0); ib = ti.select(db > 0.0, 1.0, 0.0)
    ic = ti.select(dc > 0.0, 1.0, 0.0)
    k = ia + ib + ic
    two = ti.select(k > 1.5, 1.0, 0.0)
    wa = ia + two * (1.0 - 2.0 * ia); wb = ib + two * (1.0 - 2.0 * ib)
    wc = ic + two * (1.0 - 2.0 * ic)
    o = wa * a + wb * b + wc * c
    p = wa * b + wb * c + wc * a
    q = wa * c + wb * a + wc * b
    do = wa * da + wb * db + wc * dc
    dp = wa * db + wb * dc + wc * da
    dq = wa * dc + wb * da + wc * db
    den1 = do - dp; den2 = do - dq
    t1 = do / ti.select(ti.abs(den1) > 1e-20, den1, 1.0)
    t2 = do / ti.select(ti.abs(den2) > 1e-20, den2, 1.0)
    a_sub = area * t1 * t2
    m_sub = a_sub * (3.0 * o + (p - o) * t1 + (q - o) * t2) / 3.0
    m_full = area * (a + b + c) / 3.0
    s1 = ti.select(ti.abs(k - 1.0) < 0.5, 1.0, 0.0)
    s2 = ti.select(ti.abs(k - 2.0) < 0.5, 1.0, 0.0)
    s3 = ti.select(k > 2.5, 1.0, 0.0)
    A = s1 * a_sub + s2 * (area - a_sub) + s3 * area
    M = s1 * m_sub + s2 * (m_full - m_sub) + s3 * m_full
    return A, M

@ti.func
def plane_volume(ci: int, pn: ti.math.vec3, pd: ti.f32):
    # (V, S, G): submerged volume of block ci below the plane n . x = d, dV/dx, dV/dtheta.
    R = R_f[ci]; h = he_f[ci]
    s_c = pos_f[ci].dot(pn) - pd                  # centre's height above the plane
    S = ti.Vector([0.0, 0.0, 0.0]); G = ti.Vector([0.0, 0.0, 0.0]); V = 0.0
    for f in ti.static(range(6)):
        ax = f // 2; a1 = (ax + 1) % 3; a2 = (ax + 2) % 3
        sg = 1.0 - 2.0 * (f % 2)
        u = ti.Vector([R[0, ax], R[1, ax], R[2, ax]])
        e1 = ti.Vector([R[0, a1], R[1, a1], R[2, a1]]) * h[a1]
        e2 = ti.Vector([R[0, a2], R[1, a2], R[2, a2]]) * h[a2]
        nf = sg * u
        fc = nf * h[ax]
        r00 = fc - e1 - e2; r10 = fc + e1 - e2; r11 = fc + e1 + e2; r01 = fc - e1 + e2
        d00 = -s_c - pn.dot(r00); d10 = -s_c - pn.dot(r10)
        d11 = -s_c - pn.dot(r11); d01 = -s_c - pn.dot(r01)
        tri = 2.0 * h[a1] * h[a2]
        A1, M1 = _tri_below(r00, r10, r11, d00, d10, d11, tri)
        A2, M2 = _tri_below(r00, r11, r01, d00, d11, d01, tri)
        Af = A1 + A2
        S += Af * nf
        G += (M1 + M2).cross(nf)
        V += Af * (h[ax] + s_c * nf.dot(pn)) / 3.0
    return V, S, G

@ti.func
def plane_force_feng(ci: int, pn: ti.math.vec3, pd: ti.f32, dt: ti.f32, mu: ti.f32,
                     pid: ti.template()):
    # The plane as a block of ci's own size: kH, the caps and the Rayleigh dashpot are
    # _feng_forces' with j's terms dropped. Friction: the plane's Cundall spring (slot 0)
    # at the contact point -- the cut face's centroid line, met with the plane.
    Fo = ti.Vector([0.0, 0.0, 0.0]); To = ti.Vector([0.0, 0.0, 0.0])
    ok = 0
    V, S, G = plane_volume(ci, pn, pd)
    an = S.norm()
    if V > 1e-9 and an > 1e-9:
        ok = 1
        h = he_f[ci]; m = mass_f[ci]
        Ac = (4.0/3.0)*(h[0]*h[1] + h[1]*h[2] + h[2]*h[0])
        kH = PRM[_P_FENG_KHERTZ] / (Ac * Ac)
        wprime = kH * V * ti.sqrt(V)
        cons = PRM[_P_CAP_CONSERVATIVE] > 0.5
        if cons:
            amax = 4.0 * ti.max(h[0]*h[1], ti.max(h[1]*h[2], h[2]*h[0]))
            wprime = ti.min(wprime, PRM[_P_VMAX_CONTACT] * m / dt / amax)
        Fo = -wprime * S
        To = -wprime * G
        fm = Fo.norm()
        fcap = PRM[_P_VMAX_CONTACT] * m / dt
        if fm > fcap and not cons:
            Fo = Fo * (fcap / fm); To = To * (fcap / fm)
        nhat = S / an                             # into the plane: vn > 0 approaching
        s_c = pos_f[ci].dot(pn) - pd
        rc = S.cross(G) / (an * an) - s_c * pn
        vc = vel_f[ci] + omega_f[ci].cross(rc)
        vn = vc.dot(nhat)
        rxn = rc.cross(nhat)
        me = 1.0 / (1.0/m + rxn.dot(Iiw_f[ci] @ rxn))
        De = PRM[_P_D_HERTZ] * ti.sqrt(V) * ti.sqrt(me)
        Fd = ti.min(De * ti.abs(vn), me * ti.abs(vn) / dt)
        if vn > 0:
            Fd = -Fd
        else:
            Fd = ti.min(Fd, wprime * an)
        zn = PRM[_P_ZETA_N]
        useR = 0.0
        if zn > 0.0:
            useR = 1.0
        Vdot = S.dot(vel_f[ci]) + G.dot(omega_f[ci])
        mV = 1.0 / ti.max(an * an / m + G.dot(Iiw_f[ci] @ G), 1e-12)
        kV = 1.5 * kH * ti.sqrt(V)
        cR = ti.min(2.0 * zn * ti.sqrt(kV * mV), mV / dt)
        fV = ti.max(cR * Vdot, -wprime)
        Fdv = Fd * nhat * (1.0 - useR)
        Fo += Fdv - (useR * fV) * S
        To += rc.cross(Fdv) - (useR * fV) * G
        Fn = ti.max(0.0, wprime * an - Fd)
        if zn > 0.0:
            Fn = ti.max(0.0, (wprime + fV) * an)
        Ke = ti.min(1.5 * kH * an * an * ti.sqrt(V), PRM[_P_STIFFNESS_ALPHA] * me / (dt*dt))
        Ffr = plane_friction(ci, pid, 0, dt, pn, rc, vc, vc.dot(pn), me, Ke, Fn, mu)
        Fo += Ffr
        To += rc.cross(Ffr)
    else:
        plane_spring_release(ci, pid, 0)
    return Fo, To, V, ok

# ==================================================================
# Cylinder contact -- true per-corner radial. (A per-block tangent plane
# rotates as the block slides along the wall: a moving constraint that
# does net work -> energy injection. Each corner vs the actual cylinder
# surface is a fixed potential U = sum 1/2 K (r_corner - R)^2 -> conservative.)
# ==================================================================

@ti.func
def cyl_force(ci: int, ccx: ti.f32, ccz: ti.f32, R: ti.f32, dt: ti.f32,
              K: ti.f32, dc: ti.f32, mu: ti.f32):
    Fo = ti.Vector([0.0, 0.0, 0.0]); To = ti.Vector([0.0, 0.0, 0.0])
    ok = 0
    for k in ti.static(range(8)):
        c = corner_w(ci, k)
        dx = c[0] - ccx; dz = c[2] - ccz
        rh = ti.sqrt(dx*dx + dz*dz)
        if rh > 1e-6:
            pen = ti.min(rh - R, PRM[_P_PEN_MAX])
            if pen > 0:
                ok = 1
                n = ti.Vector([-dx/rh, 0.0, -dz/rh])   # inward radial normal AT this corner
                rc = c - pos_f[ci]
                vc = vel_f[ci] + omega_f[ci].cross(rc)
                vn = vc.dot(n)
                rxn = rc.cross(n)
                me = 1.0 / (1.0/mass_f[ci] + rxn.dot(Iiw_f[ci] @ rxn))
                Ke = ti.min(K, PRM[_P_STIFFNESS_ALPHA] * me / (dt*dt))
                Fs = Ke * pen
                De = dc * ti.sqrt(Ke * me)
                Fd = ti.min(De * ti.abs(vn), me * ti.abs(vn) / dt)
                if vn > 0: Fd = -Fd
                Fn = ti.max(0.0, Fs + Fd)
                Fc = Fs * n
                vt = vc - vn * n; vtm = vt.norm()
                Fc += kinetic_friction(vt, vtm, mu * Fn, me, dt)
                f_k = (Fn - Fs) * n + Fc
                Fo += f_k
                To += rc.cross(f_k)
    return Fo, To, ok

# Force assembly is one kernel for the reset and one per boundary type, run in this
# order (BOUNDARY_KERNELS). As a single kernel, plane_force was inlined 9x (each
# unrolled over 8 corners) into one loop body, and Taichi's optimiser took ~115 s
# on it at the first, uncached launch. The accumulation order into F_f / tau_f /
# fsq_f is unchanged. Frozen interior blocks (active == 0) get Iiw + gravity from
# _prep (neighbours may read Iiw) and are skipped by every boundary kernel: they
# don't move and aren't wake-checked.

@ti.kernel
def _prep(dt: ti.f32):
    for i in range(MAX_N):
        if i >= n_f[None]: continue
        F_f[i] = ti.Vector([0.0,0.0,0.0])
        tau_f[i] = ti.Vector([0.0,0.0,0.0])
        gnd_f[i] = 0; slp_f[i] = 0; cntct_f[i] = 0; cnt_n_f[i] = 0; n_above[i] = 0
        R = R_f[i]; d = Iil_f[i]
        for row in ti.static(range(3)):
            for col in ti.static(range(3)):
                Iiw_f[i][row, col] = (d[0]*R[row,0]*R[col,0]
                                     + d[1]*R[row,1]*R[col,1]
                                     + d[2]*R[row,2]*R[col,2])
        F_f[i][1] = PRM[_P_GRAVITY] * mass_f[i]
        fsq_f[i] = F_f[i][1] * F_f[i][1]

@ti.kernel
def _bnd_ground(dt: ti.f32):
    for i in range(MAX_N):
        if i >= n_f[None] or active[i] == 0: continue
        gn = ti.Vector([0.0,1.0,0.0])
        on_ledge = 0
        if ledge_on[None] == 1:
            px = pos_f[i][0]; pz = pos_f[i][2]
            if px >= ledge_x0[None] and px <= ledge_x1[None] and pz >= ledge_z0[None] and pz <= ledge_z1[None]:
                on_ledge = 1
        if on_ledge == 1:
            f,t,p,ok = plane_force_feng(i, gn, PRM[_P_GROUND_Y], dt, PRM[_P_SOIL_MU], 0)
            if ok == 1: F_f[i] += f; fsq_f[i] += f.norm_sqr(); tau_f[i] += t; gnd_f[i] = 1

@ti.kernel
def _bnd_slope(dt: ti.f32):
    for i in range(MAX_N):
        if i >= n_f[None] or active[i] == 0: continue
        if slope_on[None] == 1 and noslope_f[i] == 0 and (pos_f[i] - slope_o_f[None]).dot(slope_t_f[None]) >= slope_umin_f[None]:
            y_hi = slope_mu_y_hi[None]; y_lo = slope_mu_y_lo[None]
            blend = ti.max(0.0, ti.min(1.0, (pos_f[i][1] - y_lo) / (y_hi - y_lo + 1e-6)))
            mu_slope = PRM[_P_SLOPE_MU_HI] + blend * (PRM[_P_SLOPE_MU_LO] - PRM[_P_SLOPE_MU_HI])
            f2 = ti.Vector([0.0, 0.0, 0.0]); t2 = ti.Vector([0.0, 0.0, 0.0]); ok2 = 0
            if slope_feng_on[None] == 1:
                f2, t2, _v2, ok2 = plane_force_feng(i, slope_n_f[None], slope_d_f[None], dt, mu_slope, 1)
            else:
                f2, t2, _p2, ok2 = plane_force(i, slope_n_f[None], slope_d_f[None], dt,
                                               PRM[_P_SLOPE_K], PRM[_P_SLOPE_D], mu_slope, 1)
            if ok2 == 1: F_f[i] += f2; fsq_f[i] += f2.norm_sqr(); tau_f[i] += t2; slp_f[i] = 1

@ti.kernel
def _bnd_wall(dt: ti.f32):
    for i in range(MAX_N):
        if i >= n_f[None] or active[i] == 0: continue
        if wall_on[None] == 1:
            f3,t3,p3,ok3 = plane_force(i, wall_n_f[None], wall_d_f[None], dt,
                                         PRM[_P_SLOPE_K], PRM[_P_SLOPE_D], PRM[_P_SLOPE_MU], -1)
            if ok3 == 1: F_f[i] += f3; fsq_f[i] += f3.norm_sqr(); tau_f[i] += t3

@ti.kernel
def _bnd_cyl(dt: ti.f32):
    for i in range(MAX_N):
        if i >= n_f[None] or active[i] == 0: continue
        if cyl_on[None] == 1:
            f4, t4, ok4 = cyl_force(i, cyl_cx_f[None], cyl_cz_f[None], cyl_R_f[None], dt,
                                    PRM[_P_SLOPE_K], PRM[_P_SLOPE_D], PRM[_P_SLOPE_MU])
            if ok4 == 1: F_f[i] += f4; fsq_f[i] += f4.norm_sqr(); tau_f[i] += t4

@ti.kernel
def _bnd_walls(dt: ti.f32):
    # The six axis-aligned walls -- box: -x, +x, -z, +z at box_w; side: -z, +z at
    # side_w, frictionless -- as a RUNTIME loop around one plane_force, so its
    # corner code is compiled once, not six times.
    for i in range(MAX_N):
        if i >= n_f[None] or active[i] == 0: continue
        for q in range(6):
            on = box_on[None]; wd = box_w[None]; wmu = PRM[_P_SLOPE_MU]
            if q >= 4:
                on = side_on[None]; wd = side_w[None]; wmu = 0.0
            if on == 1:
                sg = 1.0
                if q == 0 or q == 2 or q == 4: sg = -1.0
                nx = 0.0; nz = sg
                if q < 2:
                    nx = sg; nz = 0.0
                fb,tb,_pb,ob = plane_force(i, ti.Vector([nx, 0.0, nz]), -wd, dt,
                                           PRM[_P_SLOPE_K], PRM[_P_SLOPE_D], wmu, -1)
                if ob == 1: F_f[i] += fb; fsq_f[i] += fb.norm_sqr(); tau_f[i] += tb

BOUNDARY_KERNELS = (_prep, _bnd_ground, _bnd_slope, _bnd_wall, _bnd_cyl, _bnd_walls)
