"""Full12-method pilot comparison: all cases retained, no invented error bars."""
import csv,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.direction_a_protocol import ROOT,file_sha,write_json

def run():
    old=ROOT/'reports/direction_a_pilot_2026-10-23/numeric_source.json';report=ROOT/'reports/direction_a_revision1_2026-10-23';new=report/'numeric_source.json'
    rows=[dict(case=r['case'],method=r['method'],exact_f1=r['exact_f1'],harmful_fraction=r['harmful_edit_rate']) for r in json.loads(old.read_text())['metrics']]
    rows.extend(dict(case=r['case'],method=r['method'],exact_f1=r['exact_f1'],harmful_fraction=r['harmful_fraction']) for r in json.loads(new.read_text())['metrics'])
    methods=('base','p_only','mfe','mea','fixed_mix','residual','global_gate','local_gate','safe_local_gate','risk_fixed_mix','risk_global_gate','risk_local_gate')
    labels=('Base','P-only','MFE','MEA','Fixed mixture','Residual CNN','Global gate','Local gate','Protected local gate','Risk-calibrated fixed mix','Risk-calibrated global gate','Risk-calibrated local gate')
    assert len(rows)==24 and {(r['case'],r['method']) for r in rows}=={(c,m) for c in ('srp','tmRNA') for m in methods}
    with (report/'combined_metrics.csv').open('w',newline='',encoding='utf8') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':8,'xtick.labelsize':7,'ytick.labelsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    directory=report/'figures';directory.mkdir(exist_ok=True)
    for name,key,label in [('performance_all','exact_f1','Exact pairing F1'),('harmful_edits_all','harmful_fraction','Mean harmful edit fraction per RNA')]:
        fig,ax=plt.subplots(figsize=(7.204724409448819,5.118110236220473));fig.subplots_adjust(left=.35,right=.97,bottom=.12,top=.89)
        for case,color,marker,offset,display in [('srp','#2878A0','o',-.11,'SRP (n=826)'),('tmRNA','#777777','s',.11,'tmRNA (n=417)')]:
            values={r['method']:r for r in rows if r['case']==case};scores=[values[m][key] for m in methods]
            assert all(0<=v<=1 for v in scores);ax.scatter(scores,np.arange(len(methods))+offset,c=color,marker=marker,s=22,label=display,zorder=3,clip_on=False)
        ax.set_yticks(range(len(methods)),labels);ax.invert_yaxis();ax.set_xlim(0,1);ax.set_xlabel(label)
        ax.grid(axis='x',color='#e8e8e8',lw=.5);ax.set_axisbelow(True);ax.tick_params(axis='y',length=0)
        fig.legend(loc='upper right',bbox_to_anchor=(.98,.99),ncol=2,fontsize=8);fig.canvas.draw();assert len(fig.axes)==1
        fig.savefig(f'{directory}/{name}.pdf',facecolor='white');fig.savefig(f'{directory}/{name}.svg',facecolor='white');fig.savefig(f'{directory}/{name}.png',dpi=300,facecolor='white');plt.close(fig)
        write_json(directory/(name+'.alignment.json'),dict(status='NOT_APPLICABLE',reason='Single quantitative axis'))
        write_json(directory/(name+'.provenance.json'),dict(question='Compare predictive quality and harmful edits across every original and revised policy',
            archetype='quantitative grid: single axis',backend='Python/matplotlib',source_sha256=dict(original=file_sha(old),revision=file_sha(new),combined_csv=file_sha(report/'combined_metrics.csv')),
            rows=24,excluded_rows=0,width_mm=183,height_mm=130,statistics='One seed17, two pseudo-OOD development cases; RNA macro within family; no confidence interval or significance claim',font_minimum_pt=7))
    print(json.dumps(dict(figure_count=2,source_rows=24,excluded_rows=0)))

if __name__=='__main__':run()
