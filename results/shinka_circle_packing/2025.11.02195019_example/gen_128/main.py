# EVOLVE-BLOCK-START
import numpy as np
import math
import itertools

def enhanced_greedy_initial(n,
                            base_samples=6000,
                            local_samples=30,
                            gap_triple_samples=400,
                            seed=0):
    """
    Greedy placement with three sample streams:
      1) global random
      2) multi‐scale local around existing circles
      3) gap‐driven: circumcenters of random triples
    """
    rnd = np.random.RandomState(seed)
    centers = []
    radii = []
    for k in range(n):
        # 1) Global random points
        pts = rnd.rand(base_samples, 2)

        if k > 0:
            arr_c = np.array(centers)
            arr_r = np.array(radii)
            # 2) local multi‐scale around each circle
            local = []
            scales = [0.02, 0.05, 0.1]
            for c, r in zip(arr_c, arr_r):
                for s in scales:
                    ang = rnd.rand(local_samples)*2*np.pi
                    rl = r + s*(1 + rnd.rand(local_samples)*0.5)
                    x = c[0] + rl*np.cos(ang)
                    y = c[1] + rl*np.sin(ang)
                    pts_l = np.stack([x,y],1)
                    mask = (pts_l[:,0]>=0)&(pts_l[:,0]<=1)&(pts_l[:,1]>=0)&(pts_l[:,1]<=1)
                    local.append(pts_l[mask])
            if local:
                pts = np.vstack([pts] + local)

            # 3) gap‐driven: pick random triples, compute circumcenters
            if k >= 3:
                idxs = list(range(k))
                # sample a few random triples
                for (i,j,l) in rnd.choice(
                        list(itertools.combinations(idxs,3)),
                        size=min(gap_triple_samples,
                                 math.comb(k,3)),
                        replace=False):
                    A, B, C = arr_c[i], arr_c[j], arr_c[l]
                    # compute circumcenter
                    d = 2*( (A[0]*(B[1]-C[1]) +
                             B[0]*(C[1]-A[1]) +
                             C[0]*(A[1]-B[1])) )
                    if abs(d) < 1e-8: continue
                    ux = ((np.dot(A,A)*(B[1]-C[1]) +
                           np.dot(B,B)*(C[1]-A[1]) +
                           np.dot(C,C)*(A[1]-B[1]))/d)
                    uy = ((np.dot(A,A)*(C[0]-B[0]) +
                           np.dot(B,B)*(A[0]-C[0]) +
                           np.dot(C,C)*(B[0]-A[0]))/d)
                    if 0 <= ux <= 1 and 0 <= uy <= 1:
                        pts = np.vstack([pts, [ux, uy]])

        # Evaluate all candidates
        best_r = -1.0
        best_p = None
        if k == 0:
            # only walls
            xs, ys = pts[:,0], pts[:,1]
            rs = np.minimum.reduce([xs, ys, 1-xs, 1-ys])
            idx = np.argmax(rs)
            best_r, best_p = rs[idx], pts[idx]
        else:
            arr_c = np.array(centers)
            arr_r = np.array(radii)
            for p in pts:
                # border‐limit
                r0 = min(p[0], p[1], 1-p[0], 1-p[1])
                # circle‐limit
                d = np.linalg.norm(arr_c - p, axis=1) - arr_r
                r1 = d.min()
                r = min(r0, r1)
                if r > best_r:
                    best_r, best_p = r, p
        centers.append(best_p)
        radii.append(max(best_r,1e-8))

    return np.array(centers), np.array(radii)


def compute_radius_at(i, centers, radii):
    """Maximal radius for circle i against walls+others."""
    x,y = centers[i]
    r = min(x,y,1-x,1-y)
    if len(centers)>1:
        oth = np.delete(centers,i,0)
        rr  = np.delete(radii,i)
        d = np.linalg.norm(oth-centers[i],axis=1)-rr
        r = min(r, d.min())
    return max(r,0.0)


def adaptive_annealing(centers, radii,
                       iters=20000,
                       T0=0.08,
                       Tend=1e-5,
                       seed=1):
    """
    SA with per‐circle adaptive steps:
      step_i ∝ radii[i]/max(radii)
    """
    rnd = np.random.RandomState(seed)
    n = centers.shape[0]
    curr_c = centers.copy()
    curr_r = radii.copy()
    curr_E = curr_r.sum()
    best_c, best_r, best_E = curr_c.copy(), curr_r.copy(), curr_E
    T = T0
    decay = (Tend/T0)**(1.0/iters)
    multi_p = 0.05
    for it in range(iters):
        # cooling
        T = max(T*decay, Tend)
        if rnd.rand()<multi_p:
            idxs = rnd.choice(n,3,replace=False)
        else:
            idxs = [rnd.randint(n)]
        c_new = curr_c.copy()
        # adaptive step sizes
        maxr = curr_r.max()
        for i in idxs:
            scale = 0.02*(0.5 + 0.5*(curr_r[i]/maxr))
            c_new[i] += rnd.randn(2)*scale
        np.clip(c_new,0,1,out=c_new)
        # recompute radii for moved circles
        r_new = curr_r.copy()
        for i in idxs:
            r_new[i] = compute_radius_at(i, c_new, r_new)
            if r_new[i]<=1e-8:
                c_new = curr_c; r_new = curr_r; break
        E_new = r_new.sum()
        dE = E_new - curr_E
        if dE>0 or rnd.rand()<math.exp(dE/T):
            curr_c, curr_r, curr_E = c_new, r_new, E_new
            if E_new>best_E:
                best_c, best_r, best_E = c_new.copy(), r_new.copy(), E_new

    return best_c, best_r


def local_greedy_repack(centers, radii,
                        n_sweeps=3,
                        local_samples=20):
    """
    Sweep small local perturbations per circle, greedily re‐pick best spot.
    """
    c = centers.copy()
    r = radii.copy()
    n = len(c)
    for _ in range(n_sweeps):
        for i in range(n):
            fixed = np.delete(c, i, 0)
            fr = np.delete(r, i)
            samples = [c[i]]
            for _ in range(local_samples):
                off = 0.04*(np.random.rand(2)-0.5)
                p = np.clip(c[i]+off,0,1)
                samples.append(p)
            best_r,best_p=-1,None
            for p in samples:
                r0 = min(p[0],p[1],1-p[0],1-p[1])
                d = np.linalg.norm(fixed-p,axis=1)-fr
                r1 = d.min() if len(d)>0 else r0
                rr = min(r0,r1)
                if rr>best_r:
                    best_r,best_p=rr,p
            if best_r>0:
                c[i]=best_p; r[i]=best_r
    # finalize radii
    for i in range(n):
        r[i]=compute_radius_at(i,c,r)
    return c,r


def micro_adjust(centers, radii, steps=50, lr=0.015):
    """
    Final tiny physics‐based push:
      repulsive pair & boundary forces, accept only valid moves.
    """
    c = centers.copy()
    for _ in range(steps):
        r = np.array([compute_radius_at(i,c,radii) for i in range(len(c))])
        forces = np.zeros_like(c)
        # pairwise repulsion
        for i in range(len(c)):
            for j in range(i+1,len(c)):
                dv = c[j]-c[i]
                dist = np.linalg.norm(dv)+1e-8
                overlap = r[i]+r[j]-dist
                if overlap>0:
                    f = (overlap/dist)*dv
                    forces[j]+=f; forces[i]-=f
        # boundary repulsion
        for i in range(len(c)):
            x,y=c[i]
            ri=r[i]
            if x<ri: forces[i,0]+= (ri-x)
            if 1-x<ri: forces[i,0]-=(ri-(1-x))
            if y<ri: forces[i,1]+= (ri-y)
            if 1-y<ri: forces[i,1]-=(ri-(1-y))
        # step and clamp
        c += lr*forces
        np.clip(c,0,1,out=c)
    # final radii
    r = np.array([compute_radius_at(i,c,radii) for i in range(len(c))])
    return c,r


def construct_packing():
    n=26
    c,r = enhanced_greedy_initial(n, seed=42)
    c,r = adaptive_annealing(c, r, seed=999)
    c,r = local_greedy_repack(c, r)
    c,r = micro_adjust(c, r)
    return c,r

# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii