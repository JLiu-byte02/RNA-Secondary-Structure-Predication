"""Pure protocol checks for FULL-TR0-v1 (no model or test-score dependencies)."""
import hashlib
from itertools import combinations

SPLITS = ('TR0', 'VL0', 'TS0', 'bprna_new')
COUNTS = dict(TR0=10682, VL0=1286, TS0=1288, bprna_new=5388)
SEEDS = [17, 23, 41]


def validate_config(config):
    if (config['train_split'], config['select_split'], config['threshold_split']) != ('TR0','VL0','VL0'):
        raise ValueError('Full-stage training and selection must use TR0 / VL0 / VL0')
    if config['seeds'] != SEEDS or config['test_splits'] != ['TS0','bprna_new']:
        raise ValueError('Keep the three original paired seeds and two fixed test sets')


def audit_cohorts(manifest, splits, expected_counts=True):
    ids, sequences, counts = {}, {}, {}
    for name in SPLITS:
        ids[name] = set(splits[name])
        counts[name] = len(splits[name])
        if len(ids[name]) != counts[name] or not ids[name] or not ids[name].issubset(manifest):
            raise ValueError('Missing, empty or duplicate cohort: '+name)
        if expected_counts and counts[name] != COUNTS[name]:
            raise ValueError('Unexpected cohort size: '+name)
        sequences[name] = set()
        for identifier in splits[name]:
            row = manifest[identifier]
            sequence = row['sequence']
            if not sequence or set(sequence)-set('ACGU'):
                raise ValueError('Invalid sequence: '+identifier)
            sha = hashlib.sha256(sequence.encode()).hexdigest()
            if row.get('sequence_hash',sha) != sha:
                raise ValueError('Sequence hash mismatch: '+identifier)
            sequences[name].add(sha)
    for first, second in combinations(SPLITS,2):
        if ids[first] & ids[second]:
            raise ValueError('ID overlap: '+first+'/'+second)
        if sequences[first] & sequences[second]:
            raise ValueError('sequence overlap: '+first+'/'+second)
    return dict(passed=True, counts=counts, cross_split_sequence_overlap=0,
                train_split='TR0', selection_split='VL0', test_splits=['TS0','bprna_new'])


def shared_threshold(curves):
    if len(curves) != len(SEEDS) or any(set(c)!=set(curves[0]) for c in curves):
        raise ValueError('Require all three matching validation threshold grids')
    return max(sorted(curves[0]), key=lambda t:sum(c[t] for c in curves)/len(curves))
