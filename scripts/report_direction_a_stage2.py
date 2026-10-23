"""Build Stage2 development evidence from audited outputs, without fitting models."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/direction_a_stage2_2026-10-23'
OUT = ROOT / 'outputs/direction_a_stage2'


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf8') as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')


def read_csv(name):
    return list(csv.DictReader((REPORT / name).open(encoding='utf8')))


def build():
    for phase in ('matched', 'utility'):
        v = json.loads((REPORT / (phase + '_verification.json')).read_text())
        assert v['passed'] and v['official_predictions_verified'] == 14916
    assert json.loads((OUT / 'controller.json').read_text())['status'] == 'completed'
    jobs = json.loads((REPORT / 'jobs.json').read_text())
    accounting = {r[0]: r for line in (REPORT / 'resource_sacct.psv').read_text().splitlines()
                  if len(r := line.split('|')) >= 7}
    resources = []
    for job in jobs:
        r = accounting[job['job_id']]
        assert r[1].split()[0] in ('COMPLETED', 'FAILED', 'CANCELLED')
        seconds, cpus = int(r[3]), int(r[4])
        resources.append(dict(job_id=r[0], tag=job['tag'], phase=job['phase'], state=r[1],
                              exit_code=r[2], elapsed_s=seconds, allocated_cpus=cpus,
                              allocated_gpu_s=seconds * job['gpus'], allocated_cpu_core_s=seconds * cpus))
    write_csv(REPORT / 'resource_usage.csv', resources)
    totals = {p: dict(gpu_hours=sum(r['allocated_gpu_s'] for r in resources if r['phase'] == p) / 3600,
                      cpu_core_hours=sum(r['allocated_cpu_core_s'] for r in resources if r['phase'] == p) / 3600)
              for p in ('matched', 'mechanism')}
    totals['scope'] = 'Allocated job wall time; includes failures and CPU allocation on GPU jobs. Queue time and lightweight controller/local reporting are excluded.'
    write_json(REPORT / 'resource_summary.json', totals)
    means = []
    for phase in ('matched', 'utility'):
        rows = read_csv(phase + '_metrics.csv')
        methods = list(dict.fromkeys(r['method'] for r in rows))
        assert len(rows) == 24 and len(methods) == 12
        for m in methods:
            subset = [r for r in rows if r['method'] == m]
            assert {r['case'] for r in subset} == {'srp', 'tmRNA'}
            means.append(dict(phase=phase, method=m, exact_f1=sum(float(r['exact_f1']) for r in subset) / 2,
                              harmful_rate=sum(float(r['harmful_rate']) for r in subset) / 2))
    write_csv(REPORT / 'case_equal_summary.csv', means)
    curves, summaries, selections = [], [], []
    for case in ('srp', 'tmRNA'):
        directory = OUT / 'utility/formal' / case
        seal = json.loads((directory / 'choices_sealed.json').read_text())
        selections.append(dict(case=case, seal_sha256=seal['sha256'], runtime=seal['identity']['runtime'],
                               base_threshold=seal['base_threshold'], proposal_threshold=seal['proposal_threshold'],
                               choices=seal['choices'], training_rows=seal['training_rows'], parameters=seal['parameters']))
        for path in sorted(directory.glob('*_curve.json')):
            values = json.loads(path.read_text()); kind = path.stem.removesuffix('_curve')
            assert [r['epoch'] for r in values] == list(range(1, 151))
            best = min(values, key=lambda r: r['calibration_mse'])
            summaries.append(dict(case=case, model=kind, selected_epoch=best['epoch'],
                                  selected_calibration_mse=best['calibration_mse'],
                                  final_train_mse=values[-1]['train_mse'], final_calibration_mse=values[-1]['calibration_mse']))
            curves.extend(dict(case=case, model=kind, **r) for r in values)
    write_csv(REPORT / 'utility_learning_curves.csv', curves)
    write_csv(REPORT / 'utility_learning_summary.csv', summaries)
    write_json(REPORT / 'selection_summary.json', selections)
    print(json.dumps(dict(jobs=len(resources), totals=totals, curves=len(curves), official_views=29832)))


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 8, 'ytick.labelsize': 7,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'legend.frameon': False})
    directory = REPORT / 'figures'; directory.mkdir(exist_ok=True)
    for phase in ('matched', 'utility'):
        rows = read_csv(phase + '_metrics.csv'); methods = list(dict.fromkeys(r['method'] for r in rows))
        fig, ax = plt.subplots(figsize=(183 / 25.4, 130 / 25.4))
        fig.subplots_adjust(left=.31, right=.97, bottom=.12, top=.89)
        for case, color, marker, offset, label in [('srp', '#2878A0', 'o', -.12, 'SRP (n=826)'),
                                                  ('tmRNA', '#686868', 's', .12, 'tmRNA (n=417)')]:
            values = {r['method']: float(r['exact_f1']) for r in rows if r['case'] == case}
            assert all(0 <= v <= 1 for v in values.values())
            ax.scatter([values[m] for m in methods], np.arange(12) + offset,
                       s=24, color=color, marker=marker, label=label, zorder=3)
        ax.set_yticks(range(12), [m.replace('_', ' ') for m in methods]); ax.invert_yaxis()
        ax.set_xlim(0, 1); ax.set_xlabel('Exact pairing F1')
        ax.grid(axis='x', color='#e8e8e8', linewidth=.6); ax.set_axisbelow(True)
        fig.legend(loc='upper right', bbox_to_anchor=(.98, .99), ncol=2)
        fig.canvas.draw(); assert len(fig.axes) == 1
        stem = directory / (phase + '_performance')
        fig.savefig(str(stem) + '.pdf', facecolor='white')
        fig.savefig(str(stem) + '.svg', facecolor='white')
        fig.savefig(str(stem) + '.png', dpi=600, facecolor='white')
        plt.close(fig)
        write_json(Path(str(stem) + '.alignment.json'), dict(status='NOT_APPLICABLE', reason='One quantitative axis'))
        write_json(Path(str(stem) + '.provenance.json'), dict(
            conclusion='No stable improvement over the calibrated fixed mixture in this development pilot.',
            question='Do learned methods improve both held-out development families?',
            archetype='quantitative grid, single axis', backend='Python/matplotlib', source_csv=phase + '_metrics.csv',
            source_sha256=hashlib.sha256((REPORT / (phase + '_metrics.csv')).read_bytes()).hexdigest(),
            source_rows=24, plotted_rows=24, excluded_rows=0, width_mm=183, height_mm=130, font_floor_pt=7,
            statistics='Seed17 only. RNA-macro within each family. No error bars or significance claim. Two phases use different upstream training roles; compare methods within a phase only.'))


if __name__ == '__main__':
    build(); plot()
