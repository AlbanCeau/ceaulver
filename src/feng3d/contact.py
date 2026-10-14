"""Feng (2021) overlap-volume block-block contact: face cull, clip, Hertzian-volume force."""

import taichi as ti

from .params import (
    DEBUG_OMDT, FENG_FACECULL, FENG_FACELIST, PRM, _P_CAP_CONSERVATIVE, _P_D_HERTZ,
    _P_FENG_KHERTZ, _P_VMAX_CONTACT, _P_WAKE_VEL, _P_ZETA_N,
)
from .fields import (
    F_f, Iiw_f, MAX_PAIRS, R_f, active, cnt_n_f, cntct_f, dbg_cap, dbg_capblk, dbg_ncon,
    dbg_omdt_blk, dbg_on, facc_V, facc_a, facc_c, facc_gA, facc_gB, face_list, face_n,
    fsq_f, he_f, mass_f, max_omdt, n_above, nowake_f, omega_f, pair_i, pair_j, pair_n,
    pair_s, pos_f, sleep_f, tau_f, vel_f,
)
from .friction import (
    pair_friction,
)
from .broadphase import (
    _mi,
)


@ti.kernel
def _feng_facelist():
    # PASS 2a' (FENG_FACELIST): one thread per pair -- the slab cull of _feng_faces for all 12
    # faces, then ONE atomic to reserve the survivors' slots in face_list. About half of the
    # face threads die at the cull (4K-block pile: 49 % survive); as threads of _feng_faces
    # they still held a SIMD lane while their neighbours clipped. Same test, same frame as
    # there: block i's centre is the origin.
    face_n[None] = 0
    npairs = ti.min(pair_n[None], MAX_PAIRS)
    for p in range(npairs):
        i = pair_i[p]; j = pair_j[p]
        facc_V[p] = 0.0                  # zeroed here too: on a reused list _feng_claim does not run
        facc_c[p] = ti.Vector([0.0, 0.0, 0.0]); facc_a[p] = ti.Vector([0.0, 0.0, 0.0])
        facc_gA[p] = ti.Vector([0.0, 0.0, 0.0]); facc_gB[p] = ti.Vector([0.0, 0.0, 0.0])
        pA = ti.Vector([0.0, 0.0, 0.0]); RA = R_f[i]; hA = he_f[i]
        pB = _mi(pos_f[j] - pos_f[i]); RB = R_f[j]; hB = he_f[j]
        keep = 0
        if active[i] == 0 and active[j] == 0:
            continue                     # FENG_REUSE: listed while one was active; both frozen now
        for fi in ti.static(range(12)):
            loc = fi % 6; fa = loc // 2
            fs = 1.0 - 2.0 * (loc % 2)
            po = pA; Ro = RA; ho = hA; pq = pB; Rq = RB; hq = hB
            if ti.static(fi >= 6):
                po = pB; Ro = RB; ho = hB; pq = pA; Rq = RA; hq = hA
            a1 = (fa + 1) % 3; a2 = (fa + 2) % 3
            ua = ti.Vector([Ro[0,fa], Ro[1,fa], Ro[2,fa]])
            e1 = ti.Vector([Ro[0,a1], Ro[1,a1], Ro[2,a1]]) * ho[a1]
            e2 = ti.Vector([Ro[0,a2], Ro[1,a2], Ro[2,a2]]) * ho[a2]
            fctr = po + fs * ho[fa] * ua
            culled = 0
            for qa in ti.static(range(3)):
                uq = ti.Vector([Rq[0,qa], Rq[1,qa], Rq[2,qa]])
                hw = ti.abs(e1.dot(uq)) + ti.abs(e2.dot(uq))
                if ti.abs((fctr - pq).dot(uq)) - hw > hq[qa]: culled = 1
            if culled == 0: keep |= (1 << fi)
        k = 0
        for fi in ti.static(range(12)):
            k += (keep >> fi) & 1
        if k > 0:
            base = ti.atomic_add(face_n[None], k)
            for fi in ti.static(range(12)):
                if (keep >> fi) & 1:
                    face_list[base] = p * 12 + fi
                    base += 1


@ti.kernel
def _feng_faces():
    # PASS 2a: one thread per (pair, FACE). Each clips one of the pair's 12 box faces
    # against the other box and atomically adds its share of V_c, cvol, a_c, gA, gB to
    # the pair's accumulators (zeroed by _feng_claim / _feng_facelist).
    # FENG_FACELIST: one thread per face that survived _feng_facelist's cull; else one per
    # (pair, face). The clip polygon is thread-local; the `_once` loop below runs once.
    npairs = ti.min(pair_n[None], MAX_PAIRS)
    nthr = npairs * 12
    if ti.static(FENG_FACELIST == 1):
        nthr = face_n[None]
    for th in range(nthr):
        code = th
        if ti.static(FENG_FACELIST == 1):
            code = face_list[th]
        fi = code % 12
        p = code // 12
        for _once in range(1):
            i = pair_i[p]; j = pair_j[p]
            # --- overlap-volume narrowphase ---
            # LOCAL FRAME centred on block i. V_c and cvol are summed as
            # tetrahedra fanned from the coordinate origin, so in world
            # coordinates each term is ~|p|^3 and they cancel down to
            # V_c ~ 1e-3: float32 cancellation. Measured on a resting
            # tile (T-05, slope_hold rig): contact-point scatter 0.0007
            # at |p| = 5, 0.05 at 20, 7.0 at 200, with V_c noise 9% /
            # 50% / 36%. Every downstream use is a difference (gA, gB,
            # ra, rb), so shifting the origin changes only the rounding.
            pA = ti.Vector([0.0, 0.0, 0.0]); RA = R_f[i]; hA = he_f[i]
            pB = _mi(pos_f[j] - pos_f[i]); RB = R_f[j]; hB = he_f[j]
            Vc = 0.0
            a_c = ti.Vector([0.0,0.0,0.0])
            gA = ti.Vector([0.0,0.0,0.0])
            gB = ti.Vector([0.0,0.0,0.0])
            cvol = ti.Vector([0.0,0.0,0.0])
            ownA = 1
            if fi >= 6: ownA = 0
            loc = fi % 6; fa = loc // 2
            fs = 1.0 - 2.0 * (loc % 2)
            po = pA; Ro = RA; ho = hA; pq = pB; Rq = RB; hq = hB
            if ownA == 0:
                po = pB; Ro = RB; ho = hB; pq = pA; Rq = RA; hq = hA
            a1 = (fa + 1) % 3; a2 = (fa + 2) % 3
            ua = ti.Vector([Ro[0,fa], Ro[1,fa], Ro[2,fa]])
            u1 = ti.Vector([Ro[0,a1], Ro[1,a1], Ro[2,a1]])
            u2 = ti.Vector([Ro[0,a2], Ro[1,a2], Ro[2,a2]])
            nf = fs * ua
            fctr = po + fs * ho[fa] * ua
            e1 = u1 * ho[a1]; e2 = u2 * ho[a2]
            # per-face cull: if the face's projection is entirely outside
            # one of the other box's 3 slabs it can't touch the interior
            # (clip would be empty) -> skip. Exact: never culls a contributor.
            if ti.static(FENG_FACECULL == 1 and FENG_FACELIST == 0):
                culled = 0
                for qa in ti.static(range(3)):
                    uq = ti.Vector([Rq[0,qa], Rq[1,qa], Rq[2,qa]])
                    hw = ti.abs(e1.dot(uq)) + ti.abs(e2.dot(uq))
                    if ti.abs((fctr - pq).dot(uq)) - hw > hq[qa]: culled = 1
                if culled == 1: continue
            # coincident (flush) plane test against the other box
            shared = 0
            for qa in ti.static(range(3)):
                uqx = ti.Vector([Rq[0,qa], Rq[1,qa], Rq[2,qa]])
                if ti.abs(nf.dot(uqx)) > 1.0 - 1e-5:
                    dface = (fctr - pq).dot(uqx)
                    if ti.abs(ti.abs(dface) - hq[qa]) < 1e-5: shared = 1
            # 4 corners CCW about nf (flip winding when fs<0). The clip polygon lives in a
            # thread-local array (rows 0-11 / 12-23 = the two ping-pong buffers).
            pb = ti.Matrix.zero(ti.f32, 24, 3)
            c0 = fctr - e1 - e2; c2 = fctr + e1 + e2
            c1 = fctr + e1 - e2; c3 = fctr - e1 + e2
            if fs <= 0:
                c1 = fctr - e1 + e2; c3 = fctr + e1 - e2
            for d3 in ti.static(range(3)):
                pb[0, d3] = c0[d3]; pb[1, d3] = c1[d3]; pb[2, d3] = c2[d3]; pb[3, d3] = c3[d3]
            pcount = 4; src = 0
            for cpl in range(6):
                ca = cpl // 2; cs = 1.0 - 2.0 * (cpl % 2)
                uqx = ti.Vector([Rq[0,ca], Rq[1,ca], Rq[2,ca]])
                dst = 1 - src; oc = 0
                sb = src * 12; db = dst * 12
                for vi in range(pcount):
                    pvi = vi - 1
                    if pvi < 0: pvi = pcount - 1
                    cur = ti.Vector([pb[sb + vi, 0], pb[sb + vi, 1], pb[sb + vi, 2]])
                    prv = ti.Vector([pb[sb + pvi, 0], pb[sb + pvi, 1], pb[sb + pvi, 2]])
                    fcd = cs * ((cur - pq).dot(uqx)) - hq[ca]
                    fpd = cs * ((prv - pq).dot(uqx)) - hq[ca]
                    if (fcd <= 0.0) != (fpd <= 0.0):
                        tt = fpd / (fpd - fcd)
                        xv = prv + tt*(cur - prv)
                        if oc < 12:
                            for d3 in ti.static(range(3)): pb[db + oc, d3] = xv[d3]
                            oc += 1
                    if fcd <= 0.0:
                        if oc < 12:
                            for d3 in ti.static(range(3)): pb[db + oc, d3] = cur[d3]
                            oc += 1
                pcount = oc; src = dst
                if pcount == 0: break
            if pcount >= 3:
                sb = src * 12
                w0 = ti.Vector([pb[sb, 0], pb[sb, 1], pb[sb, 2]])
                skipvol = 0
                if shared == 1 and ownA == 0: skipvol = 1
                for t in range(1, pcount - 1):
                    wt = ti.Vector([pb[sb + t, 0], pb[sb + t, 1], pb[sb + t, 2]])
                    wt1 = ti.Vector([pb[sb + t + 1, 0], pb[sb + t + 1, 1], pb[sb + t + 1, 2]])
                    if skipvol == 0:
                        tv = w0.dot(wt.cross(wt1)) / 6.0
                        Vc += tv
                        cvol += tv * (w0 + wt + wt1) * 0.25
                    if shared == 0:
                        a_tri = 0.5 * (wt - w0).cross(wt1 - w0)
                        c_tri = (w0 + wt + wt1) / 3.0
                        if ownA == 1:
                            a_c += a_tri; gA += (c_tri - pA).cross(a_tri)
                        else:
                            gB += (c_tri - pB).cross(a_tri)
            ti.atomic_add(facc_V[p], Vc)
            for d3 in ti.static(range(3)):
                ti.atomic_add(facc_c[p][d3], cvol[d3])
                ti.atomic_add(facc_a[p][d3], a_c[d3])
                ti.atomic_add(facc_gA[p][d3], gA[d3])
                ti.atomic_add(facc_gB[p][d3], gB[d3])


@ti.kernel
def _feng_forces(dt: ti.f32):
    # PASS 2b: one thread per pair -- the forces, from the face sums _feng_faces accumulated.
    npairs = ti.min(pair_n[None], MAX_PAIRS)
    for p in range(npairs):
        i = pair_i[p]; j = pair_j[p]
        # local frame centred on block i, as in _feng_faces
        pA = ti.Vector([0.0, 0.0, 0.0]); RA = R_f[i]; hA = he_f[i]
        pB = _mi(pos_f[j] - pos_f[i]); RB = R_f[j]; hB = he_f[j]
        Vc = facc_V[p]
        a_c = facc_a[p]
        gA = facc_gA[p]
        gB = facc_gB[p]
        cvol = facc_c[p]
        if Vc > 1e-9:
            if dbg_on[None] == 1: ti.atomic_add(dbg_ncon[None], 1)
            cntct_f[i] = 1; cntct_f[j] = 1
            ti.atomic_add(cnt_n_f[i], 1); ti.atomic_add(cnt_n_f[j], 1)
            # "resting above" count: the lower block of the pair gets +1 (load from above)
            if pos_f[j][1] > pos_f[i][1] + 0.2:
                ti.atomic_add(n_above[i], 1)
            elif pos_f[i][1] > pos_f[j][1] + 0.2:
                ti.atomic_add(n_above[j], 1)
            if (vel_f[i] - vel_f[j]).norm() > PRM[_P_WAKE_VEL]:
                if sleep_f[i] == 1 and nowake_f[i] == 0: sleep_f[i] = 0
                if sleep_f[j] == 1 and nowake_f[j] == 0: sleep_f[j] = 0
            # size-scaled stiffness: Achar = mean box face area (cube: (2he)^2)
            AcA = (4.0/3.0)*(hA[0]*hA[1] + hA[1]*hA[2] + hA[2]*hA[0])
            AcB = (4.0/3.0)*(hB[0]*hB[1] + hB[1]*hB[2] + hB[2]*hB[0])
            # --- Hertzian-VOLUME elastic contact (Poschel & Schwager 2005, eq. 2.10/2.12
            #     adapted to the Feng overlap volume). P&S Hertz F~delta^1.5 is sphere-
            #     derived; here w(V)=(2/5)k V^2.5 -> wprime = k V^1.5, which for a face
            #     contact V=A*delta gives |F| ~ delta^1.5 (Hertz depth-stiffening:
            #     sublinear penetration under load) WITHOUT leaving Feng's -grad w(V_c)
            #     potential -> conservative + continuous (no SAT min-axis flip). One fixed
            #     potential, no mass/stiffness scaling, no regimes. ---
            mi = mass_f[i]; mj = mass_f[j]
            m_red = mi*mj / (mi + mj)
            kH = PRM[_P_FENG_KHERTZ] / (AcA * AcB)     # size-normalized Hertz-volume stiffness
            wprime = kH * Vc * ti.sqrt(Vc)           # = kH * V_c^{3/2}
            cons = PRM[_P_CAP_CONSERVATIVE] > 0.5
            if cons:   # T-11: saturate w'(V) itself -> still an exact gradient
                aA = 4.0 * ti.max(hA[0]*hA[1], ti.max(hA[1]*hA[2], hA[2]*hA[0]))
                aB = 4.0 * ti.max(hB[0]*hB[1], ti.max(hB[1]*hB[2], hB[2]*hB[0]))
                amax = aA if mi <= mj else aB
                wmax = PRM[_P_VMAX_CONTACT] * ti.min(mi, mj) / dt / amax
                if wprime > wmax:
                    wprime = wmax
                    if dbg_on[None] == 1:            # same probe as the force rescale
                        ti.atomic_add(dbg_cap[None], 1)
                        ti.atomic_add(dbg_capblk[i], 1); ti.atomic_add(dbg_capblk[j], 1)
            Fi = -wprime * a_c                   # conservative, F_j = -F_i
            tii = -wprime * gA
            tjj = -wprime * gB
            # deep-overlap safeguard: bound the per-contact spring impulse so a deep
            # overlap pushes apart gently (no flings). Only bites at pathological depth.
            fm = Fi.norm()
            fcap = PRM[_P_VMAX_CONTACT] * ti.min(mi, mj) / dt
            if fm > fcap and not cons:
                if dbg_on[None] == 1:
                    ti.atomic_add(dbg_cap[None], 1)
                    ti.atomic_add(dbg_capblk[i], 1); ti.atomic_add(dbg_capblk[j], 1)
                sc = fcap / fm
                Fi = Fi * sc; tii = tii * sc; tjj = tjj * sc
            # normal dashpot: P&S (2.12) viscoelastic / Kuwabara-Kono form, damping
            # ~ sqrt(V)*vn; repulsive-only (P&S eq. 2.16 -> no tensile/sticky contact)
            an = a_c.norm()
            if an > 1e-9:
                nhat = a_c / an
                cc = cvol / Vc
                ra = cc - pA; rb = cc - pB
                va = vel_f[i] + omega_f[i].cross(ra)
                vb = vel_f[j] + omega_f[j].cross(rb)
                vn = (va - vb).dot(nhat)
                rxna = ra.cross(nhat); rxnb = rb.cross(nhat)
                me = 1.0/(1.0/mi + rxna.dot(Iiw_f[i]@rxna)
                          + 1.0/mj + rxnb.dot(Iiw_f[j]@rxnb))
                # ringing-margin probe: omega*dt of this contact (k_tan = dF/d(delta)
                # = 1.5*kH*|a_c|^2*sqrt(V)). Use the LINEAR reduced mass m_red (not the
                # rotational me, which -> 0 for glancing contacts and gives spurious
                # outliers); m_red is the right mass for load-bearing face contacts. >2 = ring.
                if ti.static(DEBUG_OMDT):
                    omdt = ti.sqrt(1.5*kH*an*an*ti.sqrt(Vc)/m_red)*dt
                    ti.atomic_max(max_omdt[None], omdt)
                    ti.atomic_max(dbg_omdt_blk[i], omdt); ti.atomic_max(dbg_omdt_blk[j], omdt)
                De = PRM[_P_D_HERTZ] * ti.sqrt(Vc) * ti.sqrt(me)   # ~ sqrt(V) damping (P&S)
                Fd = ti.min(De*ti.abs(vn), me*ti.abs(vn)/dt)
                if vn > 0:
                    Fd = -Fd                     # approaching: add to repulsion
                else:
                    # separating: clamp so the dashpot can't exceed the spring and
                    # pull blocks back together (no tensile/sticky contact -> no clumps)
                    Fd = ti.min(Fd, wprime * an)
                # --- Rayleigh dashpot on V_c (T-05/T-12), when ZETA_N > 0 ---
                # R = 1/2 c Vdot^2 -> Q = -c Vdot dV/dq: along the elastic force's
                # own generalized direction, so no contact point is involved and
                # power = -c Vdot^2 <= 0. dV/dpA = a_c, dV/dpB = -a_c, dV/dthA = gA,
                # dV/dthB = gB (the elastic law applies -w' times these).
                # c = 2 zeta sqrt(k_V m_V): k_V = dw'/dV, m_V the exact mobility
                # inverse along (a_c,-a_c,gA,gB). On a flat face this is D =
                # 2 zeta sqrt(k_n m) (Tsuji et al. 1992). Clamped (i) to m_V/dt so
                # one explicit step arrests Vdot, never reverses it, and (ii) so
                # the total stays repulsive (P&S 2.16), as the point dashpot does.
                zn = PRM[_P_ZETA_N]
                useR = 0.0
                if zn > 0.0:
                    useR = 1.0
                Vdot = (a_c.dot(vel_f[i] - vel_f[j]) + gA.dot(omega_f[i])
                        + gB.dot(omega_f[j]))
                invV = (an*an*(1.0/mi + 1.0/mj) + gA.dot(Iiw_f[i] @ gA)
                        + gB.dot(Iiw_f[j] @ gB))
                mV = 1.0 / ti.max(invV, 1e-12)
                kV = 1.5 * kH * ti.sqrt(Vc)
                cR = ti.min(2.0 * zn * ti.sqrt(kV * mV), mV / dt)
                fV = ti.max(cR * Vdot, -wprime)
                Fdv = Fd * nhat * (1.0 - useR)
                Fi += Fdv - (useR * fV) * a_c
                tii += ra.cross(Fdv) - (useR * fV) * gA
                tjj += rb.cross(-Fdv) - (useR * fV) * gB
                Fn_mag = ti.max(0.0, wprime*an - Fd)   # net normal repulsion magnitude
                if zn > 0.0:
                    Fn_mag = ti.max(0.0, (wprime + fV) * an)
                Ffr = pair_friction(i, j, pair_s[p], dt, nhat, va - vb, vn, ra, rb, cc,
                                    me, Fn_mag, kH, an, Vc)
                Fi += Ffr
                tii += ra.cross(Ffr)
                tjj += rb.cross(-Ffr)
            for d3 in ti.static(range(3)):
                ti.atomic_add(F_f[i][d3],  Fi[d3])
                ti.atomic_add(F_f[j][d3], -Fi[d3])
                ti.atomic_add(tau_f[i][d3], tii[d3])
                ti.atomic_add(tau_f[j][d3], tjj[d3])
            fq = Fi.norm_sqr()
            ti.atomic_add(fsq_f[i], fq); ti.atomic_add(fsq_f[j], fq)
