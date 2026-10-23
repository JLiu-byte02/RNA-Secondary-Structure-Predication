"""Python-only, source-bound pilot figures; no uncertainty inferred from one seed."""
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.direction_a_protocol import ROOT,file_sha,write_json

REPORT=ROOT/'reports/direction_a_pilot_2026-10-23'
METHODS=('base','p_only','mfe','mea','fixed_mix','residual','global_gate','local_gate','safe_local_gate')
LABELS=('Base','P-only','MFE','MEA','Fixed mixture','Residual CNN','Global gate','Local gate','Protected local gate')
LEARNED=('residual','global_gate','local_gate','safe_local_gate')
COLORS=('#383838','#999999','#2878A0','#C37B39')

def save(fig,name,source_files,claim):
    # Each figure contains exactly one quantitative axis: panel alignment is inapplicable.
    assert len(fig.axes)==1
    fig.canvas.draw();directory=REPORT/'figures';directory.mkdir(exist_ok=True)
    for ext in ('pdf','svg','png'):fig.savefig(directory/(name+'.'+ext),dpi=300,facecolor='white')
    write_json(directory/(name+'.alignment.json'),dict(status='NOT_APPLICABLE',reason='Single quantitative axis',axes=1))
    write_json(directory/(name+'.provenance.json'),dict(core_question=claim,archetype='quantitative grid: single axis',backend='Python/matplotlib',
        sources={s:file_sha(REPORT/s) for s in source_files},width_mm=float(fig.get_figwidth()*25.4),height_mm=float(fig.get_figheight()*25.4),
        statistics='One seed17; predeclared SRP/tmRNA pseudo-OOD development cases. RNA macro within each family. No confidence interval or significance claim.',
        data_exclusions=0,font_minimum_pt=7,editable_text=True))
    plt.close(fig)

def run():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    numeric=json.loads((REPORT/'numeric_source.json').read_text());metrics=numeric['metrics']
    assert len(metrics)==18 and {(r['case'],r['method']) for r in metrics}=={(c,m) for c in ('srp','tmRNA') for m in METHODS}
    sources=['numeric_source.json','family_metrics.csv']
    for name,key,label,claim in [
        ('pilot_performance','exact_f1','Exact pairing F1','How do all predeclared corrections compare with simple baselines on each development family?'),
        ('pilot_harmful_edits','harmful_edit_rate','Mean harmful edit fraction per RNA','Does better predictive quality come with fewer damaging structural edits? Zero-edit RNAs contribute zero.')]:
        fig,ax=plt.subplots(figsize=(183/25.4,100/25.4));fig.subplots_adjust(left=.31,right=.97,bottom=.15,top=.86)
        for case,color,marker,offset,labelcase in [('srp','#2878A0','o',-.11,'SRP (n=826)'),('tmRNA','#777777','s',.11,'tmRNA (n=417)')]:
            rows={r['method']:r for r in metrics if r['case']==case};values=[float(rows[m][key]) for m in METHODS]
            assert all(0<=v<=1 for v in values)
            ax.scatter(values,np.arange(len(METHODS))+offset,c=color,s=23,marker=marker,label=labelcase,zorder=3)
        ax.set_yticks(range(len(METHODS)),LABELS);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xlabel(label)
        ax.grid(axis='x',color='#e8e8e8',lw=.5);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0)
        fig.legend(loc='upper right',bbox_to_anchor=(.98,.99),ncol=2,fontsize=8)
        save(fig,name,sources,claim)
    with (REPORT/'learning_curves.csv').open(newline='') as handle:curves=list(csv.DictReader(handle))
    assert len(curves)==120
    for case in ('srp','tmRNA'):
        fig,ax=plt.subplots(figsize=(183/25.4,90/25.4));fig.subplots_adjust(left=.11,right=.97,bottom=.17,top=.77)
        for method,color,style in zip(LEARNED,COLORS,('-','--','-.',':')):
            rows=sorted([r for r in curves if r['case']==case and r['method']==method],key=lambda r:int(r['epoch']))
            assert [int(r['epoch']) for r in rows]==list(range(1,16))
            ax.plot([int(r['epoch']) for r in rows],[float(r['calibration_exact_f1']) for r in rows],color=color,ls=style,lw=1.4,label=LABELS[METHODS.index(method)])
        ax.set_xlabel('Correction epoch');ax.set_ylabel('Calibration family-macro exact F1');ax.set_xlim(1,15);ax.set_ylim(0,1);ax.set_xticks([1,5,10,15])
        fig.legend(loc='upper center',bbox_to_anchor=(.54,.99),ncol=2,fontsize=8)
        save(fig,'calibration_'+case,['learning_curves.csv'],'How does calibration performance evolve over all15 correction epochs in the '+case+' case? This is not development-test performance.')
    print(json.dumps(dict(figures=4,source_rows=len(metrics)+len(curves),excluded_rows=0)))

if __name__=='__main__':run()
