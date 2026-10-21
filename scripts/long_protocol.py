"""Fixed training horizons; held-out scores never select a horizon or checkpoint."""
from scripts.full_protocol import validate_config

MODES=('B0','E0_no_thermo','E1_bpp','E2_bpp_stack')
BUDGETS=(10,20,30,40,50)
SEEDS=(17,23,41)

def validate_long_config(config):
    validate_config(config)
    if config['max_epochs']!=50 or config['early_stopping'] is not False or config['budgets']!=list(BUDGETS):
        raise ValueError('Require the predeclared complete 50-epoch trajectory and five horizons')

def select_epoch(rows,budget):
    available=[r for r in rows if r['epoch']<=budget]
    if sorted(r['epoch'] for r in available)!=list(range(1,budget+1)):
        raise ValueError('Budget requires every epoch exactly once')
    return max(available,key=lambda r:(r['score'],-r['epoch']))
