"""Reproduce corrected simulation figures and all Monte Carlo draws.

Usage: python run_all.py --suite all --reps 500 --variance-reps 10000
Figures/results default to THIS package's ../fig and ../results. No workspace
files are touched unless the caller explicitly chooses their output directory.
"""
import os
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ.setdefault(key,'1')
import argparse,json,time,hashlib,platform,sys
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import sim_core as sc

BASE=Path(__file__).resolve().parent.parent
COLORS=['#2171b5','#e6550d','#31a354','#756bb1','#636363']
LABELS={'HT':r'$\widehat{\tau}_{HT}$','HA':r'$\widehat{\tau}_{H\acute{a}jek}$','DM':'Unit aggregate','TR':'Exposure threshold','PI2':'SNIPE(2) / 2','V1':r'$\widehat{\sigma}^{2}$','V2_avg':r'$\widehat{\sigma}^{2}(d_1^{\mathcal{G}})$','V2_max':r'$\widehat{\sigma}^{2}(D_{\mathcal{G}})$'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.labelsize':15,'legend.fontsize':11,'pdf.fonttype':42,'ps.fonttype':42,'savefig.bbox':'tight'})

def write_json(path,d): path.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def savefig(fig,out,name):
    fig.tight_layout();fig.savefig(out/'fig'/f'{name}.pdf');fig.savefig(out/'fig'/f'{name}.png',dpi=160);plt.close(fig)

def save_cell(out,label,pop,p,raw,design='Bernoulli',zeta=0.,details=None):
    dest=out/'results'/label;dest.mkdir(parents=True,exist_ok=True)
    pop.save(dest/'population.npz');np.savez_compressed(dest/'replicates.npz',**raw)
    pd.DataFrame(raw).to_csv(dest/'replicates.csv',index=False,float_format='%.15g')
    tau=pop.target(zeta); b=len(raw['HT']);summ=[]
    for k in sc.METHODS:
        if k not in raw or np.isnan(raw[k]).all():continue
        err=raw[k]-tau
        summ.append(dict(method=k,target=tau,mean=float(raw[k].mean()),bias=float(err.mean()),relative_bias=float(err.mean()/tau),mse=float(np.mean(err*err)),sd=float(raw[k].std(ddof=1)),bias_mcse=float(err.std(ddof=1)/np.sqrt(b)),mse_mcse=float((err*err).std(ddof=1)/np.sqrt(b))))
    varsum=[]
    for k in sc.VARS:
        if k not in raw:continue
        covered=np.abs(raw['HT']-tau)<=1.959963984540054*np.sqrt(np.maximum(raw[k],0))
        cov=float(covered.mean())
        varsum.append(dict(estimator=k,mean=float(raw[k].mean()),coverage=cov,coverage_mcse=float(np.sqrt(cov*(1-cov)/b)),negative_count=int((raw[k]<0).sum()),se_ratio=float(np.sqrt(max(raw[k].mean(),0))/raw['HT'].std(ddof=1))))
    info=dict(label=label,n=pop.n,undirected_edges=pop.m,ordered_edges=2*pop.m,average_degree=float(pop.degree.mean()),max_degree=int(pop.degree.max()),degree_second_moment=float(np.mean(pop.degree**2)),treatment_probability=p,design=design,zeta=zeta,target=tau,replications=b,master_seed=sc.MASTER_SEED,empty_arm_count=int(raw['empty_arm'].sum()),empty_threshold_count=int(raw['empty_threshold'].sum()),point_summary=summ,variance_summary=varsum)
    if design=='Bernoulli' and zeta==0:info['exact_HT_variance']=sc.true_variance(pop,p)
    if details:info.update(details)
    write_json(dest/'metadata.json',info);pd.DataFrame(summ).to_csv(dest/'summary.csv',index=False)
    return dict(pop=pop,p=p,raw=raw,tau=tau,info=info)

def diagnostics(cells,xs,xlabel,out,prefix):
    pos=np.arange(len(xs));fig,ax=plt.subplots(figsize=(9,5));handles=[]
    for j,k in enumerate(sc.METHODS[:4]):
        a=[(c['raw'][k]-c['tau'])/c['tau']*100 for c in cells]
        box=ax.boxplot(a,positions=pos+(j-1.5)*.18,widths=.17,patch_artist=True,showfliers=False)
        for patch in box['boxes']:patch.set_facecolor(COLORS[j]);patch.set_alpha(.75)
        for med in box['medians']:med.set_color('black')
        handles.append(Line2D([0],[0],color=COLORS[j],lw=5,label=LABELS[k]))
    ax.set_xticks(pos,xs);ax.set_xlabel(xlabel);ax.set_ylabel('Relative estimation error (%)');ax.axhline(0,color='black',lw=.8);ax.grid(axis='y',alpha=.25);ax.legend(handles=handles,ncol=2)
    savefig(fig,out,prefix+'_boxplot')
    fig,ax=plt.subplots(figsize=(7,5))
    for j,k in enumerate(sc.METHODS[:4]):ax.plot(xs,[np.mean((c['raw'][k]-c['tau'])**2) for c in cells],'-o',color=COLORS[j],label=LABELS[k])
    ax.set_xticks(xs);ax.set_xlabel(xlabel);ax.set_ylabel('Mean squared error');ax.grid(alpha=.25);ax.legend();savefig(fig,out,prefix+'_mse')
    if 'V1' in cells[0]['raw']:
        for metric,ylabel in [('coverage','Coverage probability'),('se_ratio','RMS estimated SE / Monte Carlo SD')]:
            fig,ax=plt.subplots(figsize=(7,5))
            for j,k in enumerate(sc.VARS):
                yy=[next(v[metric] for v in c['info']['variance_summary'] if v['estimator']==k) for c in cells]
                ax.plot(xs,yy,'-o',color=COLORS[j],label=LABELS[k])
            ax.axhline(.95 if metric=='coverage' else 1,color='black',ls='--',label='Nominal 95%' if metric=='coverage' else 'Reference 1')
            ax.set_xticks(xs);ax.set_xlabel(xlabel);ax.set_ylabel(ylabel);ax.grid(alpha=.25);ax.legend()
            if metric=='coverage':ax.set_ylim(min(.85,min(v['coverage'] for c in cells for v in c['info']['variance_summary'])-.025),1.015)
            savefig(fig,out,prefix+'_'+metric)
    if not np.isnan(cells[0]['raw']['PI2']).all():
        fig,ax=plt.subplots(figsize=(7,5))
        for j,k in enumerate(sc.METHODS):ax.plot(xs,[np.mean((c['raw'][k]-c['tau'])**2) for c in cells],'-o',color=COLORS[j],label=LABELS[k])
        ax.set_yscale('log');ax.set_xticks(xs);ax.set_xlabel(xlabel);ax.set_ylabel('Mean squared error (log scale)');ax.grid(alpha=.25);ax.legend();savefig(fig,out,prefix+'_snipe2_mse')

def main_suites(out,reps,distributions):
    for distr in distributions:
        pops={}
        def getpop(n,m):
            key=(n,m)
            if key not in pops:pops[key]=sc.population(sc.ba(n,m,f'main/graph/{n}/{m}'),distr,f'main/{distr}/population/{n}/{m}')
            return pops[key]
        for sweep,xs,pars,xlabel in [
            ('n',[200,500,1000,1500,2000],[(n,10,.3) for n in [200,500,1000,1500,2000]],'Population size'),
            ('p',[.1,.2,.3,.4,.5],[(1000,10,p) for p in [.1,.2,.3,.4,.5]],'Treatment probability'),
            ('d',[4,10,20,30,40],[(1000,m,.3) for m in [2,5,10,15,20]],'Nominal average degree (2m)')]:
            cells=[]
            for n,m,p in pars:
                label=f'main/{distr}/{sweep}/n{n}_m{m}_p{p:g}';pop=getpop(n,m)
                # Reuse assignment uniforms for all probability configurations.
                alabel=f'main/{distr}/n{n}_m{m}/assignment'
                raw=sc.run(pop,p,reps,alabel)
                cells.append(save_cell(out,label,pop,p,raw,details={'assignment_seed':sc.seed(alabel),'population_seed':sc.seed(f'main/{distr}/population/{n}/{m}'),'graph_seed':sc.seed(f'main/graph/{n}/{m}'),'attachment_parameter':m,'nominal_average_degree':2*m}))
                print(f'{label}: done ({reps} replications)',flush=True)
            diagnostics(cells,xs,xlabel,out,f'{distr}_{sweep}')

def caltech_suites(out,reps,variance_reps,variance_only=False):
    graph=sc.load_caltech(Path(__file__).with_name('socfb-Caltech36.mtx'))
    pop=sc.population(graph,'uniform','caltech/population')
    probs=[.1,.2,.3,.4,.5]
    for design in ([] if variance_only else ['Bernoulli','complete']):
        cells=[]
        for p in probs:
            label=f'caltech/{design}/p{p:g}';alabel=f'caltech/{design}/assignment';raw=sc.run(pop,p,reps,alabel,design)
            cells.append(save_cell(out,label,pop,p,raw,design,details={'assignment_seed':sc.seed(alabel),'population_seed':sc.seed('caltech/population'),'actual_treatment_fraction':np.floor(p*pop.n)/pop.n if design=='complete' else p,'treated_count':int(np.floor(p*pop.n)) if design=='complete' else None}))
            print(label+': done',flush=True)
        # Existing paper uses a single two-panel figure for each design.
        fig,axs=plt.subplots(1,2,figsize=(13,5));pos=np.arange(len(probs));handles=[]
        for j,k in enumerate(sc.METHODS[:4]):
            vals=[(c['raw'][k]-c['tau'])/c['tau']*100 for c in cells]
            b=axs[0].boxplot(vals,positions=pos+(j-1.5)*.18,widths=.17,patch_artist=True,showfliers=False)
            for patch in b['boxes']:patch.set_facecolor(COLORS[j]);patch.set_alpha(.75)
            for med in b['medians']:med.set_color('black')
            handles.append(Line2D([0],[0],color=COLORS[j],lw=4,label=LABELS[k]))
            axs[1].plot(probs,[np.mean((c['raw'][k]-c['tau'])**2) for c in cells],'-o',color=COLORS[j],label=LABELS[k])
        axs[0].set_xticks(pos,probs);axs[0].set_ylabel('Relative estimation error (%)');axs[0].axhline(0,color='black',lw=.8);axs[0].legend(handles=handles,ncol=2);axs[1].set_ylabel('Mean squared error');axs[1].legend()
        for ax in axs:ax.set_xlabel('Nominal treatment probability');ax.grid(axis='y',alpha=.25)
        savefig(fig,out,design)
    # Same underlying coefficients on common ordered edges across perturbations.
    rng=np.random.default_rng(sc.seed('caltech/variance/coefficients'));n=pop.n
    coefficients=[rng.uniform(a,b,(n,n)) for a,b in [(0,1),(0,.5),(-.5,0),(0,.5)]]
    original=set(tuple(sorted(e)) for e in graph.edges());edges=sorted(original)
    rng=np.random.default_rng(sc.seed('caltech/variance/perturbation'))
    remove_order=rng.permutation(len(edges));missing=[(i,j) for i in range(n) for j in range(i+1,n) if (i,j) not in original]
    add_order=rng.permutation(len(missing))
    basepop=sc.population(graph,'uniform','unused',coefficients=coefficients)
    for sweep,xs in [('degree',[-2,-1,0,1,2,3,5]),('p',probs)]:
        cells=[];actual=[]
        for x in xs:
            g=graph.copy();p=.3 if sweep=='degree' else x
            if sweep=='degree':
                k=round(abs(x)*n/2)
                if x<0:g.remove_edges_from(edges[i] for i in remove_order[:k])
                elif x>0:g.add_edges_from(missing[i] for i in add_order[:k])
            pp=sc.population(g,'uniform','unused',coefficients=coefficients) if sweep=='degree' else basepop
            label=f'variance/{sweep}/{x:g}';alabel='caltech/variance/assignment'
            raw=sc.run(pp,p,variance_reps,alabel);c=save_cell(out,label,pp,p,raw,details={'assignment_seed':sc.seed(alabel),'coefficient_seed':sc.seed('caltech/variance/coefficients'),'perturbation_seed':sc.seed('caltech/variance/perturbation'),'requested_degree_change':x if sweep=='degree' else 0,'actual_degree_change':float(pp.degree.mean()-basepop.degree.mean())})
            cells.append(c);actual.append(float(pp.degree.mean()-basepop.degree.mean()) if sweep=='degree' else p);print(label+': done',flush=True)
        fig,ax=plt.subplots(figsize=(8,5))
        ax.plot(actual,[sc.true_variance(c['pop'],c['p']) for c in cells],'-o',color='black',label='Exact variance (HT)')
        ax.plot(actual,[c['raw']['HT'].var(ddof=1) for c in cells],'--x',color='gray',label='Monte Carlo variance (HT)')
        ax.plot(actual,[c['raw']['HA'].var(ddof=1) for c in cells],'-s',color=COLORS[4],label='Monte Carlo variance (Hajek)')
        for j,k in enumerate(sc.VARS):ax.plot(actual,[c['raw'][k].mean() for c in cells],'-o',color=COLORS[j],label='Mean '+LABELS[k])
        ax.set_xlabel('Change in average degree' if sweep=='degree' else 'Treatment probability');ax.set_ylabel('Variance');ax.grid(alpha=.25);ax.legend(ncol=2,fontsize=10);savefig(fig,out,'variance_vs_'+sweep)

def sensitivity_suites(out,reps):
    pop=sc.population(sc.ba(1000,10,'sensitivity/graph'),'uniform','sensitivity/population')
    zetas=[0,.01,.02,.05,.1,.2];p=.3;cells=[]
    for zeta in zetas:
        label=f'sensitivity/zeta{zeta:g}';raw=sc.run(pop,p,reps,'sensitivity/assignment',zeta=zeta)
        cells.append(save_cell(out,label,pop,p,raw,zeta=zeta,details={'assignment_seed':sc.seed('sensitivity/assignment'),'population_seed':sc.seed('sensitivity/population'),'graph_seed':sc.seed('sensitivity/graph')}));print(label+': done',flush=True)
    for metric in ['bias','mse','coverage']:
        fig,ax=plt.subplots(figsize=(8,5))
        methods=sc.VARS if metric=='coverage' else sc.METHODS[:4]
        for j,k in enumerate(methods):
            if metric=='coverage': y=[next(v['coverage'] for v in c['info']['variance_summary'] if v['estimator']==k) for c in cells]
            elif metric=='bias': y=[np.mean(c['raw'][k]-c['tau'])/c['tau']*100 for c in cells]
            else:y=[np.mean((c['raw'][k]-c['tau'])**2) for c in cells]
            ax.plot(zetas,y,'-o',color=COLORS[j],label=LABELS[k])
        if metric=='coverage':ax.axhline(.95,color='black',ls='--',label='Nominal 95%');ax.set_ylim(0,1.015)
        if metric=='bias':ax.axhline(0,color='black',lw=.8)
        ax.set_xlabel(r'Upstream peer-effect magnitude $\zeta$');ax.set_ylabel({'bias':'Relative bias (%)','mse':'Mean squared error','coverage':'Coverage probability'}[metric]);ax.grid(alpha=.25);ax.legend();savefig(fig,out,'sensitivity_'+metric)

def greedy_matching(pop,rng):
    used=np.zeros(pop.n,dtype=bool);chosen=[]
    for j in rng.permutation(pop.m):
        u=pop.u[j];v=pop.v[j]
        if not used[u] and not used[v]:chosen.append(j);used[u]=used[v]=True
    return np.array(chosen,dtype=int)

def subsample_suites(out,reps):
    # Same fixed graph/coefficients/assignments as sensitivity, nested Poisson samples.
    pop=sc.population(sc.ba(1000,10,'sensitivity/graph'),'uniform','sensitivity/population')
    p=.3;q=1-p;zetas=[0,.01,.02,.05,.1,.2];pis=[.25,.5,.75,1.];n=pop.n
    rz=np.random.default_rng(sc.seed('sensitivity/assignment'));rs=np.random.default_rng(sc.seed('subsample/sampling'));rm=np.random.default_rng(sc.seed('subsample/matching'))
    full=np.zeros((reps,len(zetas)));dyad=np.zeros((reps,len(zetas),len(pis)));unit=np.zeros_like(dyad);match=np.zeros_like(full);mtarget=np.zeros_like(full);msize=np.zeros(reps,dtype=int)
    dyadempty=np.zeros((reps,len(pis)),dtype=int);unitempty=np.zeros_like(dyadempty)
    for b in range(reps):
        # Matching is drawn from fixed G independently, before treatment/outcomes.
        chosen=greedy_matching(pop,rm);msize[b]=len(chosen)
        z=(rz.random(n)<p).astype(float);du=rs.random(pop.m);uu=rs.random(n)
        tt=z[pop.u]*z[pop.v]/p**2;cc=(1-z[pop.u])*(1-z[pop.v])/q**2
        for j,zeta in enumerate(zetas):
            y,_=sc.outcomes(pop,z,zeta);s=y[:pop.m]+y[pop.m:];c=s*(tt-cc);full[b,j]=c.sum()/n
            row=np.bincount(pop.ou,weights=y*np.r_[tt-cc,tt-cc],minlength=n)
            for k,pi in enumerate(pis):
                dm=du<pi;um=uu<pi;dyad[b,j,k]=c[dm].sum()/(n*pi);unit[b,j,k]=row[um].sum()/(n*pi)
                dyadempty[b,k]=not dm.any();unitempty[b,k]=not um.any()
            a,bb=pop.pair_potentials();match[b,j]=c[chosen].mean();mtarget[b,j]=(a-bb)[chosen].mean()+2*zeta
    dest=out/'results'/'subsampling';dest.mkdir(parents=True,exist_ok=True);pop.save(dest/'population.npz')
    target=np.array([pop.target(z) for z in zetas])
    np.savez_compressed(dest/'replicates.npz',full_HT=full,dyad_HT=dyad,unit_HT=unit,matching=match,matching_target=mtarget,matching_size=msize,global_target=target,zeta=zetas,pi=pis,dyad_empty=dyadempty,unit_empty=unitempty)
    rows=[]
    for j,zeta in enumerate(zetas):
        for k,pi in enumerate(pis):
            for name,arr in [('Full HT',full[:,j]),('Dyad Poisson HT',dyad[:,j,k]),('Unit Poisson HT',unit[:,j,k])]:
                err=arr-target[j];rows.append(dict(zeta=zeta,pi=pi,method=name,target=target[j],bias=err.mean(),relative_bias=err.mean()/target[j],mse=np.mean(err**2),bias_mcse=err.std(ddof=1)/np.sqrt(reps),mse_mcse=(err**2).std(ddof=1)/np.sqrt(reps)))
    pd.DataFrame(rows).to_csv(dest/'summary.csv',index=False)
    mrows=[]
    for j,zeta in enumerate(zetas):
        err=match[:,j]-mtarget[:,j];mrows.append(dict(zeta=zeta,mean_selected_target=mtarget[:,j].mean(),bias=err.mean(),mse=np.mean(err**2),bias_mcse=err.std(ddof=1)/np.sqrt(reps)))
    pd.DataFrame(mrows).to_csv(dest/'matching_summary.csv',index=False)
    write_json(dest/'metadata.json',dict(n=n,replications=reps,assignment_seed=sc.seed('sensitivity/assignment'),sampling_seed=sc.seed('subsample/sampling'),matching_seed=sc.seed('subsample/matching'),zeta=zetas,pi=pis,matching_target='Mean full-treatment minus full-control contrast of the selected unordered pairs; not the global per-unit target',empty_dyad_samples=int(dyadempty.sum()),empty_unit_samples=int(unitempty.sum()),mean_matching_size=float(msize.mean())))
    fig,axs=plt.subplots(1,2,figsize=(13,5));k=pis.index(.5)
    for h,(name,a) in enumerate([('Full HT',full),('Dyad Poisson HT',dyad[:,:,k]),('Unit Poisson HT',unit[:,:,k])]):
        err=a-target;axs[0].plot(zetas,err.mean(axis=0),'-o',color=COLORS[h],label=name);axs[1].plot(zetas,(err**2).mean(axis=0),'-o',color=COLORS[h],label=name)
    for ax in axs:ax.set_xlabel(r'Upstream peer-effect magnitude $\zeta$');ax.grid(alpha=.25);ax.legend()
    axs[0].set_ylabel('Bias for global target');axs[1].set_ylabel('MSE for global target');savefig(fig,out,'subsampling_robustness')
    fig,axs=plt.subplots(1,2,figsize=(11,4.6));err=match-mtarget
    axs[0].errorbar(zetas,err.mean(0),yerr=1.96*err.std(0,ddof=1)/np.sqrt(reps),fmt='-o');axs[0].set_ylabel('Bias for selected-pair target');axs[0].axhline(0,color='black',lw=.8)
    axs[1].plot(zetas,np.mean(err**2,axis=0),'-o');axs[1].set_ylabel('MSE for selected-pair target')
    for ax in axs:ax.set_xlabel(r'Upstream peer-effect magnitude $\zeta$');ax.grid(alpha=.25)
    savefig(fig,out,'matching_selected_target');print('subsampling: done',flush=True)

def cluster_suites(out,reps):
    complete=[];cluster=[];ls=[100,200,300,400,500]
    for ell in ls:
        n=5*ell+1;hub=n-1;g=nx.Graph();g.add_nodes_from(range(n));rng=np.random.default_rng(sc.seed(f'cluster/graph/{ell}'))
        for k in range(ell):
            nodes=list(range(5*k,5*k+5));g.add_edges_from((i,j) for i in nodes for j in nodes if i<j);g.add_edge(hub,int(rng.choice(nodes)))
        ids=np.r_[np.repeat(np.arange(ell),5),ell]
        pop=sc.population(g,'cluster',f'cluster/population/{ell}',ids)
        for design,cells in [('complete',complete),('cluster',cluster)]:
            label=f'cluster/{design}/L{ell}';alabel=label+'/assignment';n1=int(np.floor(5*ell/3))
            raw=sc.run(pop,1/3,reps,alabel,design,n1=n1)
            cells.append(save_cell(out,label,pop,1/3,raw,design,details={'L':ell,'assignment_seed':sc.seed(alabel),'population_seed':sc.seed(f'cluster/population/{ell}'),'graph_seed':sc.seed(f'cluster/graph/{ell}'),'partition':'Planted five-node cliques plus singleton hub (new explicit implementation; no Leiden)','treated_count':n1 if design=='complete' else None,'actual_treatment_fraction':n1/n if design=='complete' else 1/3}))
            print(label+': done',flush=True)
    for design,cells in [('complete',complete),('cluster',cluster)]:
        diagnostics(cells,ls,'Number of five-node cliques',out,'star_'+design)
        # Preserve the included legacy filename; its ylabel correctly says error.
        for ext in ['pdf','png']:(out/'fig'/f'star_{design}_boxplot.{ext}').rename(out/'fig'/f'star_{design}_relative_bias.{ext}')

def manifest(out):
    versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,networkx=nx.__version__,matplotlib=matplotlib.__version__,master_seed=sc.MASTER_SEED)
    write_json(out/'results'/'versions.json',versions)
    files=[]
    for folder in ['codes','fig','results']:
        for p in sorted((out/folder).rglob('*')):
            if p.is_file() and p.name!='manifest.json' and '__pycache__' not in p.parts:files.append(dict(path=str(p.relative_to(out)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
    write_json(out/'manifest.json',{'master_seed':sc.MASTER_SEED,'files':files})

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--suite',choices=['all','main','caltech','variance','sensitivity','subsampling','cluster'],default='all');parser.add_argument('--reps',type=int,default=500);parser.add_argument('--variance-reps',type=int,default=10000);parser.add_argument('--distributions',nargs='+',choices=['uniform','binomial','lognormal'],default=['uniform','binomial','lognormal']);parser.add_argument('--output',type=Path,default=BASE);args=parser.parse_args()
    if min(args.reps,args.variance_reps)<2:parser.error('At least two replications required')
    out=args.output.resolve();(out/'fig').mkdir(parents=True,exist_ok=True);(out/'results').mkdir(parents=True,exist_ok=True);t0=time.time()
    if args.suite in ['all','main']:main_suites(out,args.reps,args.distributions)
    if args.suite in ['all','caltech','variance']:caltech_suites(out,args.reps,args.variance_reps,variance_only=args.suite=='variance')
    if args.suite in ['all','sensitivity']:sensitivity_suites(out,args.reps)
    if args.suite in ['all','subsampling']:subsample_suites(out,args.reps)
    if args.suite in ['all','cluster']:cluster_suites(out,args.reps)
    manifest(out);print(f'Completed {args.suite} in {time.time()-t0:.1f} seconds. Output: {out}',flush=True)

if __name__=='__main__':main()
