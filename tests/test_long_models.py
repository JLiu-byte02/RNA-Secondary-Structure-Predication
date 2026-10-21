import copy
import io
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts._common import set_seed,build_candidates
from scripts.long_models import make_model
from scripts.run_thermo_fast import state_hash
from scripts.run_long_epochs import train_epoch


class LongModels(unittest.TestCase):
    def test_b0_is_head_only_with_identical_seed_initialization(self):
        for seed in (17,23,41):
            set_seed(seed);b0=make_model('B0')
            self.assertEqual(sum(p.numel() for p in b0.parameters()),175553)
            self.assertFalse(hasattr(b0,'correction'))
            for mode in ('E0_no_thermo','E1_bpp','E2_bpp_stack'):
                set_seed(seed);e=make_model(mode)
                self.assertEqual(state_hash(b0.head.state_dict()),state_hash(e.head.state_dict()))

    def test_b0_predictions_ignore_all_physical_and_distance_channels(self):
        set_seed(17);model=make_model('B0').eval()
        embedding=torch.randn(9,1280);mask=torch.ones(9,9,dtype=torch.bool)
        first=model(embedding,torch.zeros(2,9,9),torch.zeros(9,9),mask)
        second=model(embedding,torch.randn(2,9,9)*100,torch.randn(9,9)*100,mask)
        self.assertTrue(torch.equal(first,second));self.assertTrue(torch.equal(first,first.T))

    def test_training_resumes_model_and_adamw_state_exactly(self):
        set_seed(17)
        ids=list('abcde');sequence='GGGAAACCC'
        ci,cj,labels=build_candidates(sequence,{(0,8),(1,7),(2,6)})
        records={sid:dict(sequence=sequence,cand_i=ci,cand_j=cj,labels=labels,
            physical=torch.zeros(2,9,9),distance=torch.zeros(9,9),valid=torch.ones(9,9,dtype=torch.bool)) for sid in ids}
        embeddings={sid:torch.randn(9,1280) for sid in ids}
        config=dict(gradient_accumulation=4,clip_grad_norm=1.)
        set_seed(17);continuous=make_model('B0');optimizer=torch.optim.AdamW(continuous.parameters(),lr=1e-4,weight_decay=1e-4)
        train_epoch(continuous,optimizer,ids,embeddings,records,17,1,config,torch.device('cpu'))
        saved=io.BytesIO();torch.save(dict(model=continuous.state_dict(),optimizer=optimizer.state_dict()),saved)
        train_epoch(continuous,optimizer,ids,embeddings,records,17,2,config,torch.device('cpu'))
        set_seed(17);resumed=make_model('B0');other=torch.optim.AdamW(resumed.parameters(),lr=1e-4,weight_decay=1e-4)
        saved.seek(0);state=torch.load(saved,weights_only=False)
        resumed.load_state_dict(state['model']);other.load_state_dict(state['optimizer'])
        train_epoch(resumed,other,ids,embeddings,records,17,2,config,torch.device('cpu'))
        self.assertEqual(state_hash(continuous.state_dict()),state_hash(resumed.state_dict()))

if __name__=='__main__':unittest.main(verbosity=2)
