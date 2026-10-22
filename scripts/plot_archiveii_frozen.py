"""Render verified real family-OOD results; no simulated or incomplete inputs."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.archiveii_frozen_protocol import FAMILIES,MODES,SEEDS
ALL_MODES=('RiNALMo_official',)+MODES
COLORS=('#686868','#56A9CF','#173E69','#CD722C')
MARKERS=('X','s','^','D')
NAMES=('RiNALMo','E0','E1','E2')
DISPLAY=('5S rRNA','16S rRNA','23S rRNA','tRNA','SRP','tmRNA','RNaseP','grp1','Telomerase')

def read_csv(path):
    with path.open(encoding='utf8',newline='') as stream:return list(csv.DictReader(stream))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path,record):path.write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def validate(report):
    validation=json.loads((report/'artifact_validation.json').read_text(encoding='utf8'))
    assert validation['passed'] and validation['independently_verified_files']==90 and validation['predictions']==38650
    runs=read_csv(report/'run_results.csv');family=read_csv(report/'family_results.csv');paired=read_csv(report/'paired_E_vs_E0.csv')
    expected={(f,m,str(s)) for f in FAMILIES for m in MODES for s in SEEDS}|{(f,'RiNALMo_official','') for f in FAMILIES}
    assert len(runs)==90 and {(r['family'],r['mode'],r['seed']) for r in runs}==expected
    assert len(family)==36 and len(paired)==20
    assert {(r['family'],r['mode']) for r in family}=={(f,m) for f in FAMILIES for m in ALL_MODES}
    for row in runs:
        for metric in ('f1','exact_f1'):
            assert math.isfinite(float(row[metric])) and 0<=float(row[metric])<=1
    for row in family:
        selected=[r for r in runs if r['family']==row['family'] and r['mode']==row['mode']]
        assert len(selected)==int(row['n_models'])
        for metric in ('f1','exact_f1'):
            values=[float(r[metric]) for r in selected]
            assert abs(statistics.mean(values)-float(row[metric+'_mean']))<1e-12
            if len(values)==3:assert abs(statistics.stdev(values)-float(row[metric+'_sample_sd']))<1e-12
            else:assert row[metric+'_sample_sd']==''
    for row in paired:
        mode=row['contrast'].split(' - ')[0]
        for metric in ('f1','exact_f1'):
            values=[]
            for seed in SEEDS:
                def score(method):
                    selected=[float(r[metric]) for r in runs if r['mode']==method and int(r['seed'] or 0)==seed
                        and (r['family']==row['family'] or row['family']=='ALL_9_equal_family')]
                    return statistics.mean(selected)
                value=score(mode)-score(MODES[0]);values.append(value)
                assert abs(value-float(row[f'{metric}_seed{seed}']))<1e-12
            assert abs(statistics.mean(values)-float(row[metric+'_mean']))<1e-12
            assert abs(statistics.stdev(values)-float(row[metric+'_sample_sd']))<1e-12
    return runs,family,paired

class Figures:
    def __init__(self,report):
        self.report=report;self.directory=report/'figures';self.directory.mkdir(exist_ok=True)
        self.qa=self.directory/'qa';self.qa.mkdir(exist_ok=True);self.records=[]
        self.qa_dir=Path(os.environ['RNA_FIGURE_QA_DIR']).resolve() if os.environ.get('RNA_FIGURE_QA_DIR') else None
        plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':9,
            'axes.labelsize':9,'axes.titlesize':11,'xtick.labelsize':8,'ytick.labelsize':8.5,'legend.fontsize':8.5,
            'axes.linewidth':.7,'svg.fonttype':'none','svg.hashsalt':'ARCHIVEII-FROZEN-E-v2','pdf.fonttype':42,
            'ps.fonttype':42,'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
    def start(self,title,xlabel,labels):
        fig=plt.figure(figsize=(183/25.4,155/25.4));ax=fig.add_axes((.27,.16,.69,.65))
        fig.text(.27,.955,title,fontsize=11,weight='bold',ha='left',va='top')
        ax.set_yticks(range(len(labels)),labels);ax.set_ylim(len(labels)-.55,-.55)
        ax.set_xlabel(xlabel,labelpad=6);ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='x',color='#E5E5E5',linewidth=.6,zorder=0);ax.set_axisbelow(True)
        ax.tick_params(direction='out',length=3,width=.6)
        return fig,ax
    def footer(self,fig,first,second):
        fig.text(.045,.045,first,fontsize=7.5,ha='left',va='bottom')
        fig.text(.045,.018,second,fontsize=7.5,ha='left',va='bottom')
    def export(self,fig,name,claim,source,rule):
        fig.canvas.draw();alignment='Single plot area; panel alignment not applicable.'
        if self.qa_dir:
            sys.path.insert(0,str(self.qa_dir))
            from audit_panel_alignment import require_matplotlib_panel_alignment
            alignment=require_matplotlib_panel_alignment(fig,json_out=self.qa/f'{name}.alignment.json',strict=True)
        fig.savefig(self.directory/f'{name}.svg',metadata={'Date':None})
        fig.savefig(self.directory/f'{name}.pdf',metadata={'CreationDate':None,'ModDate':None})
        fig.savefig(self.directory/f'{name}.png',dpi=300);plt.close(fig)
        checks={}
        if self.qa_dir:
            commands={'text':[sys.executable,str(self.qa_dir/'audit_pdf_text.py'),f'{name}.pdf','--min-pt','7','--json'],
                'collision':[sys.executable,str(self.qa_dir/'audit_figure_collisions.py'),f'{name}.pdf',
                    '--json-out',f'qa/{name}.collision.json','--overlay-pdf',f'qa/{name}.collision.pdf','--json']}
            for kind,command in commands.items():
                result=subprocess.run(command,cwd=self.directory,text=True,encoding='utf8',capture_output=True)
                (self.qa/f'{name}.{kind}.stdout.json').write_text(result.stdout,encoding='utf8')
                if result.stderr:(self.qa/f'{name}.{kind}.stderr.txt').write_text(result.stderr,encoding='utf8')
                checks[kind]={'exit_code':result.returncode}
                if result.returncode:raise RuntimeError(f'{name} {kind} audit failed: '+result.stdout[-1200:])
        self.records.append(dict(name=name,claim=claim,source=source,display_rule=rule,alignment=alignment,automatic_checks=checks,
            files={f'{name}.{ext}':digest(self.directory/f'{name}.{ext}') for ext in ('svg','pdf','png')}))

def family_figure(figures,runs,aggregates,metric,name,title,limits):
    labels=[f"{label} (n={next(r['n'] for r in aggregates if r['family']==family)})" for family,label in zip(FAMILIES,DISPLAY)]
    fig,ax=figures.start(title,'RNA-macro F1',labels)
    offsets=(-.27,-.09,.09,.27)
    for index,(mode,color,marker,offset) in enumerate(zip(ALL_MODES,COLORS,MARKERS,offsets)):
        for position,family in enumerate(FAMILIES):
            row=next(r for r in aggregates if r['family']==family and r['mode']==mode)
            mean=float(row[metric+'_mean']);sd=float(row[metric+'_sample_sd'] or 0)
            y=position+offset
            if mode!='RiNALMo_official':
                selected=sorted([r for r in runs if r['family']==family and r['mode']==mode],key=lambda r:int(r['seed']))
                ax.scatter([float(r[metric]) for r in selected],[y+j for j in (-.025,0,.025)],s=10,
                    facecolors='white',edgecolors=color,linewidths=.65,zorder=4)
            ax.errorbar(mean,y,xerr=sd if mode!='RiNALMo_official' else None,fmt=marker,color=color,
                markersize=4 if mode!='RiNALMo_official' else 4.5,capsize=2,elinewidth=.9,zorder=5)
    ax.set_xlim(*limits);ax.set_xticks((0,.2,.4,.6,.8,1.0))
    handles=[Line2D([],[],color=c,marker=m,linestyle='None',markersize=4,label=n) for n,c,m in zip(NAMES,COLORS,MARKERS)]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.255,.90),ncols=4,frameon=False,columnspacing=1.8,handletextpad=.6)
    figures.footer(fig,'E: mean +/- sample SD and seed dots (17, 23, 41); RiNALMo: one fixed official checkpoint.',
        'Official ArchiveII folds and full CT truth. E encoder frozen; reference encoder fine-tuned.')
    figures.export(fig,name,'Compare all four models on every held-out family using '+metric+'.',
        'family_results.csv; run_results.csv','All 36 family-method summaries; 81 E seed scores plus 9 fixed-reference scores. Shared F1 limits across metric views.')

def paired_figure(figures,paired):
    labels=list(DISPLAY)+['9-family mean'];fig,ax=figures.start('Contribution of P/Q relative to E0','Paired flexible F1 difference from E0',labels)
    ax.axvline(0,color='#808080',linewidth=.8,linestyle='--',zorder=1)
    ax.axhline(8.5,color='#AAAAAA',linewidth=.6,zorder=1)
    low=high=0.
    for contrast,color,marker,offset,label in zip((MODES[1]+' - E0_no_thermo',MODES[2]+' - E0_no_thermo'),
        COLORS[2:],MARKERS[2:],(-.12,.12),('E1 - E0','E2 - E0')):
        for position,family in enumerate(tuple(FAMILIES)+('ALL_9_equal_family',)):
            row=next(r for r in paired if r['family']==family and r['contrast']==contrast)
            mean=float(row['f1_mean']);sd=float(row['f1_sample_sd']);low=min(low,mean-sd);high=max(high,mean+sd)
            ax.scatter([float(row[f'f1_seed{seed}']) for seed in SEEDS],[position+offset+j for j in (-.025,0,.025)],
                s=11,facecolors='white',edgecolors=color,linewidths=.65,zorder=4)
            ax.errorbar(mean,position+offset,xerr=sd,fmt=marker,color=color,markersize=4.5,capsize=2,elinewidth=.9,zorder=5)
    pad=max(.01,(high-low)*.08);ax.set_xlim(low-pad,high+pad)
    handles=[Line2D([],[],color=c,marker=m,linestyle='None',markersize=4,label=l) for c,m,l in zip(COLORS[2:],MARKERS[2:],('E1 - E0','E2 - E0'))]
    fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.255,.90),ncols=2,frameon=False,columnspacing=2)
    figures.footer(fig,'Matched seeds: mean +/- sample SD plus all three seed differences; SD is not a confidence interval.',
        'Bottom row first averages nine families within each seed. No significance claims from three seeds.')
    figures.export(fig,'paired_flexible_vs_E0','Isolate the additional physical-feature contribution under paired E training.',
        'paired_E_vs_E0.csv; run_results.csv','All 18 family contrasts plus 2 nine-family aggregate contrasts; all 60 seed differences retained.')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',type=Path,default=ROOT/'reports/archiveii_frozen_2026-10-22')
    args=parser.parse_args();report=args.report.resolve();runs,aggregates,paired=validate(report)
    figures=Figures(report)
    observed={family:{mode:float(next(r for r in aggregates if r['family']==family and r['mode']==mode)['f1_mean']) for mode in ALL_MODES} for family in FAMILIES}
    write_json(figures.directory/'figure_contract.json',dict(question='How do frozen E models generalize to each held-out family, and what is the paired contribution of P/Q?',
        observed_flexible_F1=observed,archetype='quantitative comparison and paired effects',backend='Python/Matplotlib',
        roles={'family_flexible_f1':'Primary model comparison','paired_flexible_vs_E0':'Controlled physical-feature contribution','family_exact_f1':'Supporting metric sensitivity'},
        export=dict(width_mm=183,height_mm=155,minimum_font_pt=7,formats=['svg','pdf','png'],png_dpi=300,editable_text=True),
        data_integrity='All 90 model-family-seed results retained; all 20 paired contrasts; no data removal or smoothing.',
        uncertainty='E: sample SD of paired seeds17/23/41; fixed published baseline has no seed-SD estimate; no significance/CI claims.',
        scope='Common official test CT membership; E uses frozen generic embeddings and original training, published baseline has a fine-tuned encoder.'))
    if figures.qa_dir:
        result=subprocess.run([sys.executable,str(figures.qa_dir/'validate_figure.py'),'scripts/plot_archiveii_frozen.py','--backend','python','--json'],
            cwd=ROOT,text=True,encoding='utf8',capture_output=True)
        (figures.qa/'source_preflight.json').write_text(result.stdout,encoding='utf8')
        if result.returncode:raise RuntimeError('Source preflight failed: '+result.stdout)
    lower=min([0.]+[float(row[metric+'_mean'])-float(row[metric+'_sample_sd'] or 0) for row in aggregates for metric in ('f1','exact_f1')])-.02
    upper=max([1.]+[float(row[metric+'_mean'])+float(row[metric+'_sample_sd'] or 0) for row in aggregates for metric in ('f1','exact_f1')])+.02
    family_figure(figures,runs,aggregates,'f1','family_flexible_f1','Family-OOD performance on ArchiveII',(lower,upper))
    paired_figure(figures,paired)
    family_figure(figures,runs,aggregates,'exact_f1','family_exact_f1','Exact-pairing sensitivity on ArchiveII',(lower,upper))
    versions={}
    for package in ('numpy','matplotlib','PyMuPDF','pillow','fonttools','contourpy','kiwisolver','cycler','packaging','pyparsing','python-dateutil','six'):
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]=None
    write_json(figures.directory/'figure_manifest.json',dict(figures=figures.records,
        sources={name:digest(report/name) for name in ('family_results.csv','run_results.csv','paired_E_vs_E0.csv')},
        plotting_source_sha256=digest(Path(__file__)),qa_enabled=bool(figures.qa_dir),
        runtime=dict(python=platform.python_version(),packages=versions),manual_review='Pending inspection of every actual final PNG.'))
    print(json.dumps(dict(figure_count=len(figures.records),directory=str(figures.directory))))

if __name__=='__main__':main()
