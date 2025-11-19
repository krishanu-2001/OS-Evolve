# EVOLVE-BLOCK-START
import numpy as np

def construct_packing():
    """
    Build and optimize a 26-circle packing in a unit square.
    Returns:
        centers: np.ndarray shape (26,2)
        radii:   np.ndarray shape (26,)
    """
    optimizer = CirclePackingOptimizer(n=26, margin=0.02, seed=2025)
    return optimizer.run()

class CirclePackingOptimizer:
    def __init__(self, n, margin, seed):
        self.n = n
        self.margin = margin
        self.rng = np.random.default_rng(seed)
        # modular stages
        self.initializer = Initializer(n, margin, self.rng)
        self.gapfiller  = GapFiller(n, margin, self.rng)
        self.annealer   = Annealer(n, margin, self.rng)
        self.relaxer    = Relaxer(n, margin, self.rng)
        self.repacker   = Repacker(n, margin, self.rng)

    def run(self):
        # Stage 1: initialization
        centers = self.initializer.initialize()
        # Stage 2: localized gap filling
        centers = self.gapfiller.fill(centers)
        # Stage 3: adaptive simulated annealing
        centers = self.annealer.anneal(centers)
        # Stage 4: force‐based fine‐tuning
        centers = self.relaxer.relax(centers)
        # Stage 5: final local greedy repacking
        centers = self.repacker.repack(centers)
        # compute final radii
        radii = compute_max_radii(centers, self.margin)
        return centers, radii

class Initializer:
    def __init__(self, n, margin, rng):
        self.n = n
        self.margin = margin
        self.rng = rng
        self.layouts = [
            [6,5,6,5,4],[5,6,5,6,4],[6,6,5,5,4],
            [5,5,6,6,4],[6,5,5,6,4]
        ]

    def initialize(self):
        r = self.rng.random()
        if r < 0.3:
            return self._hex_layout()
        elif r < 0.5:
            return self._corner_layout()
        elif r < 0.7:
            return self._random_layout()
        else:
            return self._greedy_layout()

    def _hex_layout(self):
        layout = self.rng.choice(self.layouts)
        maxc = max(layout)
        dx = (1-2*self.margin)/maxc
        h  = dx*np.sqrt(3)/2
        c = np.zeros((self.n,2))
        idx=0
        for row,oc in enumerate(layout):
            x0 = self.margin + (maxc-oc)*dx/2
            y  = self.margin + row*h
            for j in range(oc):
                c[idx] = [x0+j*dx, y]
                idx+=1
        jitter = (self.rng.random((self.n,2))-0.5)*dx*0.1
        return np.clip(c+jitter, self.margin,1-self.margin)

    def _corner_layout(self):
        c = np.zeros((self.n,2))
        m = self.margin+0.01
        # four corners
        corners = np.array([[m,m],[m,1-m],[1-m,m],[1-m,1-m]])
        c[:4]=corners
        # rest along edges
        rest = self.n-4
        per = rest//4; ex = rest%4
        idx=4
        for side in range(4):
            cnt = per + (1 if ex>side else 0)
            for k in range(cnt):
                t=(k+1)/(cnt+1)
                if side==0: c[idx]=[m+t*(1-2*m),m]
                if side==1: c[idx]=[1-m,m+t*(1-2*m)]
                if side==2: c[idx]=[1-m-t*(1-2*m),1-m]
                if side==3: c[idx]=[m,1-m-t*(1-2*m)]
                idx+=1
        jitter=(self.rng.random((self.n,2))-0.5)*0.01
        return np.clip(c+jitter,self.margin,1-self.margin)

    def _random_layout(self):
        return self.rng.uniform(self.margin,1-self.margin,(self.n,2))

    def _greedy_layout(self, samples=5000):
        centers=[]; radii=[]
        for k in range(self.n):
            pts=self.rng.uniform(self.margin,1-self.margin,(samples,2))
            best_r=-1; best_p=None
            if k>0:
                arr_c=np.array(centers); arr_r=np.array(radii)
            for p in pts:
                r=min(p[0]-self.margin,p[1]-self.margin,
                      1-self.margin-p[0],1-self.margin-p[1])
                if k>0:
                    d=np.linalg.norm(arr_c-p,axis=1)-arr_r
                    r=min(r,d.min())
                if r>best_r:
                    best_r, best_p=r,p
            centers.append(best_p); radii.append(max(best_r,1e-6))
        return np.array(centers)

class GapFiller:
    def __init__(self, n, margin, rng):
        self.n=n; self.margin=margin; self.rng=rng

    def fill(self, centers, local_samples=200):
        c=centers.copy()
        for i in range(self.n):
            other = np.delete(c,i,axis=0)
            other_r = compute_max_radii(other,self.margin)
            best_r = compute_max_radii(c,self.margin)[i]
            best_p = c[i].copy()
            angles=self.rng.uniform(0,2*np.pi,local_samples)
            offsets=self.rng.uniform(best_r*0.2,best_r*0.8,local_samples)
            pts = best_p + np.stack([np.cos(angles),np.sin(angles)],axis=1)*offsets[:,None]
            pts = np.clip(pts,self.margin,1-self.margin)
            for p in pts:
                r=min(p[0]-self.margin,p[1]-self.margin,
                      1-self.margin-p[0],1-self.margin-p[1])
                if other.shape[0]>0:
                    d=np.linalg.norm(other-p,axis=1)-other_r
                    r=min(r,d.min())
                if r>best_r:
                    best_r,best_p=r,p
            c[i]=best_p
        return c

class Annealer:
    def __init__(self, n, margin, rng):
        self.n=n; self.margin=margin; self.rng=rng
        self.iters=12000; self.cluster_prob=0.07; self.multi_k=(2,4)

    def anneal(self, centers):
        c=centers.copy()
        r=compute_max_radii(c,self.margin); e=r.sum()
        best_c, best_e = c.copy(), e
        T0, T1 = 0.05, 1e-4
        stagn=0
        for k in range(self.iters):
            frac=1-k/self.iters
            T=T0*frac + T1*(1-frac)
            if self.rng.random()<self.cluster_prob:
                kchg=self.rng.integers(*self.multi_k)
                idxs=self.rng.choice(self.n,kchg,replace=False)
            else:
                idxs=[self.rng.integers(self.n)]
            cand=c.copy()
            step=frac*(1-2*self.margin)
            delta=self.rng.normal(scale=step,size=(len(idxs),2))
            cand[idxs]+=delta; cand=np.clip(cand,self.margin,1-self.margin)
            rc=compute_max_radii(cand,self.margin); ec=rc.sum()
            dE=ec-e
            if dE>0 or self.rng.random()<np.exp(dE/T):
                c, r, e = cand, rc, ec
                if ec>best_e:
                    best_c, best_e, stagn = cand.copy(), ec, 0
                else:
                    stagn+=1
            else:
                stagn+=1
            # adaptive temp bump
            if stagn>200:
                T0*=1.1; stagn=0
        return best_c

class Relaxer:
    def __init__(self,n,margin,rng):
        self.n=n; self.margin=margin; self.rng=rng; self.steps=50

    def relax(self, centers):
        c=centers.copy()
        for _ in range(self.steps):
            r=compute_max_radii(c,self.margin)
            forces=np.zeros_like(c)
            for i in range(self.n):
                for j in range(i+1,self.n):
                    dv=c[j]-c[i]; d=np.linalg.norm(dv)
                    min_d=r[i]+r[j]+1e-6
                    if d<min_d:
                        dirv=dv/d if d>0 else self.rng.normal(2)
                        overlap=min_d-d
                        f=0.15*overlap*dirv
                        forces[i]-=f; forces[j]+=f
            # boundary
            for i in range(self.n):
                x,y=c[i]; ri=r[i]
                if x-ri<self.margin: forces[i,0]+=0.2*(self.margin-(x-ri))
                if x+ri>1-self.margin: forces[i,0]-=0.2*((x+ri)-(1-self.margin))
                if y-ri<self.margin: forces[i,1]+=0.2*(self.margin-(y-ri))
                if y+ri>1-self.margin: forces[i,1]-=0.2*((y+ri)-(1-self.margin))
            c+=0.1*forces
            c=np.clip(c,self.margin,1-self.margin)
        return c

class Repacker:
    def __init__(self,n,margin,rng):
        self.n=n; self.margin=margin; self.rng=rng

    def repack(self, centers, samples=150):
        c=centers.copy()
        for i in range(self.n):
            r_all=compute_max_radii(c,self.margin)
            best_r, best_p = r_all[i], c[i].copy()
            other=np.delete(c,i,axis=0)
            other_r=np.delete(r_all,i)
            # sample local + global
            pts_local = best_p + self.rng.normal(scale=best_r*0.3,size=(samples,2))
            pts_global= self.rng.uniform(self.margin,1-self.margin,(samples,2))
            for p in np.vstack((pts_local,pts_global)):
                p=np.clip(p,self.margin,1-self.margin)
                r=min(p[0]-self.margin,p[1]-self.margin,
                      1-self.margin-p[0],1-self.margin-p[1])
                if other.shape[0]>0:
                    d=np.linalg.norm(other-p,axis=1)-other_r
                    r=min(r,d.min())
                if r>best_r:
                    best_r,best_p=r,p
            c[i]=best_p
        return c

def compute_max_radii(centers, margin):
    """
    Given centers, compute maximal non-overlapping radii with boundary margin.
    """
    n=centers.shape[0]
    xs,ys=centers[:,0],centers[:,1]
    radii=np.minimum.reduce([xs-margin,ys-margin,1-margin-xs,1-margin-ys])
    for _ in range(30):
        changed=False
        for i in range(n):
            for j in range(i+1,n):
                dv=centers[j]-centers[i]
                d=np.hypot(dv[0],dv[1])
                if d>1e-12:
                    ri,rj=radii[i],radii[j]
                    if ri+rj>d:
                        scale=d/(ri+rj)
                        radii[i]*=scale; radii[j]*=scale
                        changed=True
        if not changed: break
    return np.clip(radii,0,None)

# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii = construct_packing()
    # Calculate the sum of radii
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii