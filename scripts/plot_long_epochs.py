"""Render the completed, independently validated LONG-50 study (no synthetic data).

Usage: python scripts/plot_long_epochs.py --report reports/long_epochs_2026-10-21
Optional: RNA_FIGURE_QA_DIR points at locally installed nature-figure QA scripts.
All drawing and exports use Matplotlib; the numerical training runtime is untouched.
"""
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

MODES=('B0','E0_no_thermo','E1_bpp','E2_bpp_stack')
LABELS=('B0','E0','E1','E2')
COLORS=('#686868','#56A9CF','#173E69','#CD722C')
MARKERS=('o','s','^','D')
SEEDS=(17,23,41)
BUDGETS=(10,20,30,40,50)
ROOT=Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(encoding='utf8',newline='') as stream:
        return list(csv.DictReader(stream))


def write_json(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_data(report):
    validation=json.loads((report/'artifact_validation.json').read_text(encoding='utf8'))
    if not validation.get('passed'):
        raise ValueError('Refusing to plot results without successful independent validation')
    epochs=read_csv(report/'epoch_curves.csv')
    aggregate=read_csv(report/'budget_aggregate.csv')
    results=read_csv(report/'budget_results.csv')
    if len(epochs)!=600 or len(aggregate)!=20 or len(results)!=180:
        raise ValueError('Require every epoch, every budget and every cohort')
    expected={(m,s,e) for m in MODES for s in SEEDS for e in range(1,51)}
    observed=[(r['mode'],int(r['seed']),int(r['epoch'])) for r in epochs]
    if len(set(observed))!=600 or set(observed)!=expected:
        raise ValueError('Missing or duplicate epoch rows')
    for row in epochs:
        for key in ('validation_f1','best_so_far_f1','validation_bce','train_bce_online','cumulative_training_minutes'):
            if not math.isfinite(float(row[key])):raise ValueError(f'Nonfinite {key}')
    expected_results={(m,s,b,c) for m in MODES for s in SEEDS for b in BUDGETS for c in ('VL0','TS0','bprna_new')}
    keys=[(r['mode'],int(r['seed']),int(r['budget']),r['split']) for r in results]
    if len(set(keys))!=180 or set(keys)!=expected_results:raise ValueError('Incomplete budget result grid')
    for row in results:
        if int(row['selected_epoch'])>int(row['budget']):raise ValueError('Future checkpoint leakage')
    for mode in MODES:
        for budget in BUDGETS:
            row=[r for r in aggregate if r['mode']==mode and int(r['budget'])==budget]
            if len(row)!=1:raise ValueError('Incomplete aggregate grid')
            for cohort in ('VL0','TS0','bprna_new'):
                values=[float(r['rna_macro_exact_f1']) for r in results if r['mode']==mode and int(r['budget'])==budget and r['split']==cohort]
                for suffix,value in (('mean',statistics.mean(values)),('sd',statistics.stdev(values))):
                    if not math.isclose(float(row[0][f'{cohort}_rna_macro_exact_f1_{suffix}']),value,abs_tol=1e-12):
                        raise ValueError('Published aggregation differs from seed results')
    return epochs,aggregate,results


class Figures:
    def __init__(self,report):
        self.report=report
        self.destination=report/'figures'
        self.destination.mkdir(exist_ok=True)
        self.qa=self.destination/'qa'
        self.qa.mkdir(exist_ok=True)
        qa_dir=os.environ.get('RNA_FIGURE_QA_DIR')
        self.qa_dir=Path(qa_dir).resolve() if qa_dir else None
        self.records=[]
        plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':9,
            'axes.labelsize':9,'axes.titlesize':11,'xtick.labelsize':8,'ytick.labelsize':8,
            'legend.fontsize':8.5,'axes.linewidth':.7,'lines.linewidth':1.6,
            'svg.fonttype':'none','svg.hashsalt':'LONG-50-v1','pdf.fonttype':42,'ps.fonttype':42,
            'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})

    def start(self,title,xlabel,ylabel,footer):
        fig=plt.figure(figsize=(183/25.4,118/25.4))
        ax=fig.add_axes((.14,.23,.82,.55))
        fig.text(.14,.94,title,fontsize=11,weight='bold',ha='left',va='top')
        fig.text(.14,.055,footer,fontsize=7.5,ha='left',va='bottom')
        fig.text(.14,.018,'TR0 train; VL0 selection; seeds 17, 23, 41. F1 uses complete truth.',fontsize=7.5,ha='left',va='bottom')
        ax.set_xlabel(xlabel,labelpad=6);ax.set_ylabel(ylabel,labelpad=6)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',color='#E5E5E5',linewidth=.6,zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(direction='out',length=3,width=.6)
        return fig,ax

    def legend(self,fig,labels=LABELS,colors=COLORS,markers=MARKERS):
        handles=[Line2D([],[],color=c,marker=m,markersize=4,linewidth=1.6,label=l) for l,c,m in zip(labels,colors,markers)]
        fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.13,.868),ncols=len(labels),
            frameon=False,handlelength=2,columnspacing=2.2)

    def export(self,fig,name,claim,source,display_rule):
        fig.canvas.draw()
        alignment='Single plot area: comparable-panel alignment is not applicable.'
        if self.qa_dir:
            sys.path.insert(0,str(self.qa_dir))
            from audit_panel_alignment import require_matplotlib_panel_alignment
            alignment=require_matplotlib_panel_alignment(fig,json_out=self.qa/f'{name}.alignment.json',strict=True)
        fig.savefig(self.destination/f'{name}.svg',metadata={'Date':None})
        fig.savefig(self.destination/f'{name}.pdf',metadata={'CreationDate':None,'ModDate':None})
        fig.savefig(self.destination/f'{name}.png',dpi=300)
        plt.close(fig)
        checks={}
        if self.qa_dir:
            commands={
                'text':[sys.executable,str(self.qa_dir/'audit_pdf_text.py'),f'{name}.pdf','--min-pt','7','--json'],
                'collision':[sys.executable,str(self.qa_dir/'audit_figure_collisions.py'),f'{name}.pdf',
                    '--json-out',f'qa/{name}.collision.json','--overlay-pdf',f'qa/{name}.collision.pdf','--json']}
            for kind,command in commands.items():
                completed=subprocess.run(command,text=True,encoding='utf8',capture_output=True,cwd=self.destination)
                checks[kind]={'exit_code':completed.returncode}
                (self.qa/f'{name}.{kind}.stdout.json').write_text(completed.stdout,encoding='utf8')
                if completed.stderr:(self.qa/f'{name}.{kind}.stderr.txt').write_text(completed.stderr,encoding='utf8')
                if completed.returncode:raise RuntimeError(f'{kind} QA failed for {name}: {completed.stdout[-1200:]}')
        self.records.append(dict(name=name,claim=claim,source=source,display_rule=display_rule,
            alignment=alignment,automatic_checks=checks,
            files={f'{name}.{s}':digest(self.destination/f'{name}.{s}') for s in ('svg','pdf','png')}))


def epoch_figure(figures,epochs,field,name,title,ylabel,claim,footer):
    fig,ax=figures.start(title,'Completed training epoch',ylabel,footer)
    for mode,color,marker in zip(MODES,COLORS,MARKERS):
        tracks=[]
        for seed in SEEDS:
            values=sorted([r for r in epochs if r['mode']==mode and int(r['seed'])==seed],key=lambda r:int(r['epoch']))
            track=[float(r[field]) for r in values];tracks.append(track)
            ax.plot(range(1,51),track,color=color,alpha=.2,linewidth=.65,zorder=1)
        means=[statistics.mean(v) for v in zip(*tracks)]
        deviations=[statistics.stdev(v) for v in zip(*tracks)]
        ax.fill_between(range(1,51),[m-s for m,s in zip(means,deviations)],[m+s for m,s in zip(means,deviations)],color=color,alpha=.09,linewidth=0,zorder=2)
        ax.plot(range(1,51),means,color=color,marker=marker,markevery=[9,19,29,39,49],markersize=4,zorder=3)
    ax.set_xlim(.5,50.5);ax.set_xticks((1,10,20,30,40,50))
    figures.legend(fig)
    figures.export(fig,name,claim,'epoch_curves.csv',f'All 600 rows; field={field}; faint individual seeds, bold mean, band sample SD. No smoothing.')


def budget_figure(figures,aggregate,cohort,name,title):
    fig,ax=figures.start(title,'Predeclared training budget (epochs)','RNA-macro exact F1',
        'Mean +/- sample SD (n=3). Each budget selects checkpoints using VL0 only.')
    for mode,color,marker in zip(MODES,COLORS,MARKERS):
        rows=sorted([r for r in aggregate if r['mode']==mode],key=lambda r:int(r['budget']))
        ax.errorbar(BUDGETS,[float(r[f'{cohort}_rna_macro_exact_f1_mean']) for r in rows],
            yerr=[float(r[f'{cohort}_rna_macro_exact_f1_sd']) for r in rows],color=color,marker=marker,
            markersize=4,capsize=3,elinewidth=.9,zorder=3)
    ax.set_xlim(8,52);ax.set_xticks(BUDGETS);figures.legend(fig)
    figures.export(fig,name,f'{cohort} generalization at all five predeclared budgets.',
        'budget_aggregate.csv; budget_results.csv',f'All 20 method-budget summaries for {cohort}; n=3 paired seeds per point; sample SD.')


def cost_figure(figures,aggregate):
    fig,ax=figures.start('Validation benefit relative to measured training cost',
        'Cumulative training segment time per seed (minutes)','Selected VL0 RNA-macro exact F1',
        'Left to right: budgets 10, 20, 30, 40, 50. Mean +/- sample SD; inherited cost included.')
    for mode,color,marker in zip(MODES,COLORS,MARKERS):
        rows=sorted([r for r in aggregate if r['mode']==mode],key=lambda r:int(r['budget']))
        x=[float(r['cumulative_training_minutes_mean']) for r in rows]
        y=[float(r['VL0_rna_macro_exact_f1_mean']) for r in rows]
        ax.errorbar(x,y,xerr=[float(r['cumulative_training_minutes_sd']) for r in rows],
            yerr=[float(r['VL0_rna_macro_exact_f1_sd']) for r in rows],color=color,marker=marker,
            capsize=2,elinewidth=.65,markersize=4,zorder=3)
    figures.legend(fig)
    figures.export(fig,'validation_vs_time','Selected validation performance versus actual cumulative training cost.',
        'budget_aggregate.csv','All 20 method-budget points; both axes use n=3 mean +/- sample SD. Queue and validation time excluded.')


def paired_figure(figures,results,cohort,title,name):
    fig,ax=figures.start(title,'Method contrast at the fixed 50-epoch budget','Paired exact F1 difference',
        'One dot per paired seed; large marker is mean +/- sample SD (n=3).')
    ax.axhline(0,color='#808080',linewidth=.8,linestyle='--',zorder=1)
    for index,mode in enumerate(MODES[1:]):
        differences=[]
        for seed in SEEDS:
            scores={r['mode']:float(r['rna_macro_exact_f1']) for r in results if int(r['budget'])==50 and int(r['seed'])==seed and r['split']==cohort}
            differences.append(scores[mode]-scores['B0'])
        ax.scatter([index+o for o in (-.11,0,.11)],differences,s=20,facecolor='white',edgecolor=COLORS[index+1],linewidth=.9,zorder=4)
        ax.errorbar(index,statistics.mean(differences),yerr=statistics.stdev(differences),color=COLORS[index+1],
            marker=MARKERS[index+1],markersize=5,capsize=5,linewidth=1.2,zorder=5)
    ax.set_xlim(-.5,2.5);ax.set_xticks(range(3),('E0 - B0','E1 - B0','E2 - B0'))
    fig.text(.14,.84,'Same seed, data, optimization and selection rules; B0 has the original PairHead.',fontsize=8,ha='left')
    figures.export(fig,name,f'Paired {cohort} performance difference from B0 at the fixed final budget.',
        'budget_results.csv',f'All 12 seed-method scores at predeclared budget 50 on {cohort}; three matched contrasts. Other budgets retained in source CSV.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=ROOT/'reports/long_epochs_2026-10-21')
    args=parser.parse_args();report=args.report.resolve()
    epochs,aggregate,results=validate_data(report)
    # Observed conclusions are recorded before styling/rendering; never guessed.
    observed={}
    for mode in MODES:
        rows=[r for r in aggregate if r['mode']==mode];by_budget={int(r['budget']):r for r in rows}
        best_epochs=[]
        for seed in SEEDS:
            track=sorted([r for r in epochs if r['mode']==mode and int(r['seed'])==seed],key=lambda r:int(r['epoch']))
            best_epochs.append(int(max(track,key=lambda r:float(r['validation_f1']))['epoch']))
        observed[mode]={'best_epochs':best_epochs,
            'last_epoch_VL0_F1_change_10_to_50':float(by_budget[50]['validation_f1_mean'])-float(by_budget[10]['validation_f1_mean']),
            'selected_TS0_F1_change_30_to_50':float(by_budget[50]['TS0_rna_macro_exact_f1_mean'])-float(by_budget[30]['TS0_rna_macro_exact_f1_mean']),
            'selected_bprna_new_F1_change_30_to_50':float(by_budget[50]['bprna_new_rna_macro_exact_f1_mean'])-float(by_budget[30]['bprna_new_rna_macro_exact_f1_mean'])}
    figures=Figures(report)
    if figures.qa_dir:
        checked=subprocess.run([sys.executable,str(figures.qa_dir/'validate_figure.py'),
            'scripts/plot_long_epochs.py','--backend','python','--json'],cwd=ROOT,text=True,encoding='utf8',capture_output=True)
        (figures.qa/'source_preflight.json').write_text(checked.stdout,encoding='utf8')
        if checked.returncode:raise RuntimeError('Plotting source preflight failed: '+checked.stdout)
    write_json(figures.destination/'figure_contract.json',dict(question='Does training longer improve generalization enough to justify the cost, and how does B0 compare with E?',
        observed_evidence=observed,archetype='quantitative trajectory and paired comparison figures',backend='Python/Matplotlib',
        export=dict(width_mm=183,height_mm=118,minimum_font_pt=7,formats=['svg','pdf','png'],editable_text=True),
        uncertainty='n=3 fixed paired seeds; sample SD, not confidence intervals; no significance claims',
        data_integrity='All 600 epoch rows and all 20 budget aggregates retained; no outliers excluded or smoothed.',
        interpretation='Raw epoch F1 and selection envelope are distinct. Online train BCE and end-of-epoch validation BCE use different parameter states.'))
    epoch_figure(figures,epochs,'validation_f1','validation_f1_by_epoch','Validation performance of the epoch-end model',
        'VL0 RNA-macro exact F1','Raw validation trajectories reveal late improvement or deterioration.',
        'Thin lines: individual seeds; bold lines: mean; band: sample SD (n=3).')
    epoch_figure(figures,epochs,'best_so_far_f1','validation_best_so_far','Best validation score available within each prefix',
        'Best-so-far VL0 exact F1','The selection envelope is monotonic by definition, not proof of continued optimization.',
        'Prefix maximum per seed; monotonicity follows from selection. Mean +/- sample SD (n=3).')
    epoch_figure(figures,epochs,'train_bce_online','training_loss_by_epoch','Optimization across all 50 training epochs',
        'Online training candidate BCE','Training loss describes optimization, independently of held-out generalization.',
        'Online loss averages successive parameter states. Mean +/- sample SD; all seeds shown.')
    epoch_figure(figures,epochs,'validation_bce','validation_loss_by_epoch','Held-out validation loss across training',
        'VL0 candidate BCE','Validation BCE helps assess whether lower training loss generalizes.',
        'End-of-epoch model on all VL0 sequences. Mean +/- sample SD; all seeds shown.')
    budget_figure(figures,aggregate,'TS0','TS0_f1_vs_budget','TS0 performance across fixed training budgets')
    budget_figure(figures,aggregate,'bprna_new','bprna_new_f1_vs_budget','bpRNA-new performance across fixed training budgets')
    cost_figure(figures,aggregate)
    paired_figure(figures,results,'TS0','Paired differences from B0 on TS0','TS0_paired_vs_B0')
    paired_figure(figures,results,'bprna_new','Paired differences from B0 on bpRNA-new','bprna_new_paired_vs_B0')
    versions={}
    for package in ('numpy','matplotlib','PyMuPDF','pillow','fonttools','contourpy','kiwisolver','cycler','packaging','pyparsing','python-dateutil','six'):
        try:versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:versions[package]=None
    write_json(figures.destination/'figure_manifest.json',dict(figures=figures.records,
        source_sha256={n:digest(report/n) for n in ('epoch_curves.csv','budget_aggregate.csv','budget_results.csv')},
        plotting_source_sha256=digest(Path(__file__)),qa_enabled=figures.qa_dir is not None,
        runtime=dict(python=platform.python_version(),system=platform.system(),machine=platform.machine(),packages=versions),
        manual_review='Pending inspection of the final PNG previews; automatic audits do not replace visual review.'))
    print(json.dumps(dict(figures=len(figures.records),directory=str(figures.destination))))


if __name__=='__main__':main()
