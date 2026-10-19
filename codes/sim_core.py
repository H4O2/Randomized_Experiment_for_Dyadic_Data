"""Finite-population dyadic simulations with fixed, known undirected support.

Every undirected edge supports two ordered outcomes, including numeric zeros.
All sums below operate on this fixed support. No empirical data are simulated.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import networkx as nx

MASTER_SEED = 20261018
METHODS = ['HT', 'HA', 'DM', 'TR', 'PI2']
VARS = ['V1', 'V2_avg', 'V2_max']

def seed(label):
    return int.from_bytes(hashlib.sha256(f'{MASTER_SEED}:{label}'.encode()).digest()[:8], 'little')

@dataclass
class Population:
    n: int
    u: np.ndarray
    v: np.ndarray
    alpha: np.ndarray
    beta: np.ndarray
    gamma: np.ndarray
    lam: np.ndarray
    cluster: np.ndarray | None = None
    def __post_init__(self):
        self.m = len(self.u)
        self.ou = np.r_[self.u, self.v]
        self.ov = np.r_[self.v, self.u]
        self.degree = np.bincount(np.r_[self.u, self.v], minlength=self.n)
    def target(self, zeta=0.):
        return float(np.sum(self.beta + self.gamma + self.lam)/self.n + zeta*2*self.m/self.n)
    def pair_potentials(self):
        a = self.alpha+self.beta+self.gamma+self.lam
        return a[:self.m]+a[self.m:], self.alpha[:self.m]+self.alpha[self.m:]
    def save(self, path):
        d = {k:getattr(self,k) for k in ['n','u','v','alpha','beta','gamma','lam']}
        if self.cluster is not None: d['cluster']=self.cluster
        np.savez_compressed(path, **d)

def graph_edges(g):
    e=np.array(sorted(tuple(sorted(e)) for e in g.edges()), dtype=int)
    return e[:,0],e[:,1]

def population(g, distr, label, cluster=None, coefficients=None):
    u,v=graph_edges(g); n=g.number_of_nodes(); rng=np.random.default_rng(seed(label)); k=2*len(u)
    if coefficients is not None:
        ou=np.r_[u,v]; ov=np.r_[v,u]; a,b,c,d=[x[ou,ov] for x in coefficients]
    elif distr in ('uniform','cluster'):
        a=rng.uniform(0,1,k); b=rng.uniform(0,1 if distr=='cluster' else .5,k)
        c=rng.uniform(-.5,0,k); d=rng.uniform(0,.5,k)
    elif distr=='binomial':
        a=np.zeros(k); b=rng.binomial(1,.2,k).astype(float); c=rng.binomial(1,.1,k).astype(float); d=rng.binomial(1,.1,k).astype(float)
    elif distr=='lognormal':
        a=rng.lognormal(1,.5,k); b=rng.lognormal(.2,.1,k); c=rng.lognormal(.2,.1,k); d=rng.lognormal(.2,.1,k)
    else: raise ValueError(distr)
    return Population(n,u,v,a,b,c,d,cluster)

def ba(n,m,label):
    return nx.barabasi_albert_graph(n,m,seed=seed(label))

def load_caltech(path):
    # Supplied file has a single-% header; parse the coordinate data explicitly.
    rows=[]; shape=None
    with open(path) as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith('%'): continue
            vals=[int(x) for x in line.split()[:3]]
            if shape is None: shape=vals; continue
            rows.append((vals[0]-1,vals[1]-1))
    if shape is None or len(shape)!=3 or shape[0]!=shape[1] or len(rows)!=shape[2]:
        raise ValueError('Invalid MatrixMarket dimensions/coordinate count')
    if any(i==j or min(i,j)<0 or max(i,j)>=shape[0] for i,j in rows): raise ValueError('Invalid edge')
    g=nx.Graph(); g.add_nodes_from(range(shape[0])); g.add_edges_from(rows)
    if g.number_of_edges()!=shape[2]: raise ValueError('Duplicate undirected edges')
    return g

def neighbour_sum(pop, edge_weights):
    return np.bincount(pop.u,weights=edge_weights,minlength=pop.n)+np.bincount(pop.v,weights=edge_weights,minlength=pop.n)

def outcomes(pop,z,zeta=0.):
    peer=(np.bincount(pop.u,weights=z[pop.v],minlength=pop.n)+np.bincount(pop.v,weights=z[pop.u],minlength=pop.n))/np.maximum(pop.degree,1)
    return pop.alpha+pop.beta*z[pop.ou]+pop.gamma*z[pop.ov]+pop.lam*z[pop.ou]*z[pop.ov]+zeta*peer[pop.ou],peer

def estimate(pop,z,p,design='Bernoulli',zeta=0.,with_variance=True):
    """Complete uses exact joint probabilities; variance formulas are Bernoulli only."""
    z=np.asarray(z,dtype=float); n=pop.n; m=pop.m; q=1-p
    if design=='complete':
        n1=int(z.sum()); p=n1/n; q=1-p
        if min(n1,n-n1)<2: raise ValueError('Need at least two units in each arm')
        pt=n1*(n1-1)/(n*(n-1)); pc=(n-n1)*(n-n1-1)/(n*(n-1))
    elif design=='cluster':
        same=pop.cluster[pop.u]==pop.cluster[pop.v]
        pt=np.where(same,p,p*p); pc=np.where(same,q,q*q)
    elif design=='Bernoulli': pt=p*p; pc=q*q
    else: raise ValueError(design)
    y,peer=outcomes(pop,z,zeta); s=y[:m]+y[m:]
    tt=z[pop.u]*z[pop.v]; cc=(1-z[pop.u])*(1-z[pop.v])
    tnum=np.sum(tt*s/pt); cnum=np.sum(cc*s/pc)
    nt=z.sum()/p; nc=(n-z.sum())/q
    ht=(tnum-cnum)/n
    ha=(tnum/nt if nt else 0.)-(cnum/nc if nc else 0.)
    yi=np.bincount(pop.ou,weights=y,minlength=n)+np.bincount(pop.ov,weights=y,minlength=n)
    dm=np.sum(yi*(z/p-(1-z)/q))/n
    it=(z==1)&(peer>=p)&(pop.degree>0)
    ic=(z==0)&((1-peer)>=q)&(pop.degree>0)
    tr=(yi[it].mean() if it.any() else 0.)-(yi[ic].mean() if ic.any() else 0.)
    pi2=np.nan
    if design=='Bernoulli':
        x=(z-p)/(p*q)
        t=x+np.bincount(pop.u,weights=x[pop.v],minlength=n)+np.bincount(pop.v,weights=x[pop.u],minlength=n)
        x2=x*x
        ss=x2+np.bincount(pop.u,weights=x2[pop.v],minlength=n)+np.bincount(pop.v,weights=x2[pop.u],minlength=n)
        score=t+(1-2*p)*(t*t-ss)/2
        # Aggregate unit total has all-treated/all-control contrast 2*tau.
        pi2=np.sum(yi*score)/(2*n)
    out=dict(zip(METHODS,[ht,ha,dm,tr,pi2]))
    out['empty_arm']=float(nt==0 or nc==0); out['empty_threshold']=float(not it.any() or not ic.any())
    if with_variance and design=='Bernoulli':
        diag=np.sum(tt*s*s/p**4+cc*s*s/q**4)/n**2
        at=np.bincount(pop.u,weights=s*z[pop.v],minlength=n)+np.bincount(pop.v,weights=s*z[pop.u],minlength=n)
        bt=np.bincount(pop.u,weights=s*s*z[pop.v],minlength=n)+np.bincount(pop.v,weights=s*s*z[pop.u],minlength=n)
        ac=np.bincount(pop.u,weights=s*(1-z[pop.v]),minlength=n)+np.bincount(pop.v,weights=s*(1-z[pop.u]),minlength=n)
        bc=np.bincount(pop.u,weights=s*s*(1-z[pop.v]),minlength=n)+np.bincount(pop.v,weights=s*s*(1-z[pop.u]),minlength=n)
        tri_t=np.sum(z*(at*at-bt))/p**4/n**2
        tri_c=np.sum((1-z)*(ac*ac-bc))/q**4/n**2
        square=2*np.sum(tt*s*s/p**2+cc*s*s/q**2)/n**2
        out.update(V1=diag+tri_t+tri_c,
                   V2_avg=diag+pop.degree.mean()*square+q*tri_t+p*tri_c,
                   V2_max=diag+pop.degree.max()*square+q*tri_t+p*tri_c)
    return out

def true_variance(pop,p):
    """Exact Bernoulli HT variance, dyadic (zeta=0) model, fixed population."""
    q=1-p; a,b=pop.pair_potentials(); t=a-b
    diag=np.sum(a*a/p**2+b*b/q**2-t*t)
    def cross(x):
        sx=neighbour_sum(pop,x)
        return np.sum(sx*sx)-2*np.sum(x*x)
    return float((diag+cross(a)/p+cross(b)/q-cross(t))/pop.n**2)

def expected_v1(pop,p):
    a,b=pop.pair_potentials(); t=a-b
    return true_variance(pop,p)+(np.sum(neighbour_sum(pop,t)**2)-np.sum(t*t))/pop.n**2

def expected_v2(pop,p,delta):
    a,b=pop.pair_potentials(); q=1-p
    def cross(x): return np.sum(neighbour_sum(pop,x)**2)-2*np.sum(x*x)
    return float((np.sum(a*a/p**2+b*b/q**2)+2*delta*np.sum(a*a+b*b)+q*cross(a)/p+p*cross(b)/q)/pop.n**2)

def run(pop,p,reps,label,design='Bernoulli',zeta=0.,n1=None):
    rng=np.random.default_rng(seed(label)); rows=[]
    for _ in range(reps):
        if design=='complete':
            z=np.zeros(pop.n); z[rng.choice(pop.n,int(np.floor(p*pop.n)) if n1 is None else n1,replace=False)]=1
        elif design=='cluster':
            z=rng.binomial(1,p,int(pop.cluster.max())+1)[pop.cluster]
        else: z=(rng.random(pop.n)<p).astype(float)
        rows.append(estimate(pop,z,p,design,zeta))
    return {k:np.array([r[k] for r in rows]) for k in rows[0]}
