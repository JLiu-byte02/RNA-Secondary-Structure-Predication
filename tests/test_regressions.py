import importlib.util,json,sys,tempfile,unittest
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R))
def load(name):
 s=importlib.util.spec_from_file_location(name,R/'scripts'/f'{name}.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
class Regression(unittest.TestCase):
 def test_product_control_geometry(self):
  import torch
  m=load('run_thermo_fast');self.assertTrue(hasattr(m,'marginal_product'),'Missing required conditional product control')
  p=torch.zeros(9,9);p[0,8]=p[8,0]=.6;p[1,7]=p[7,1]=.4
  q=m.marginal_product(p);self.assertAlmostEqual(q[0,8].item(),.24,places=6);self.assertTrue(torch.equal(q,q.T))
 def test_eligible_distinct_from_primary_full_truth(self):
  m=load('eval_thermo_fast')
  with tempfile.TemporaryDirectory(dir=R/'tmp') as tmp:
   f=Path(tmp)/'p.jsonl';f.write_text(json.dumps(dict(rna_id='x',sequence='GGGAAACCC',family_id='f',reference_pairs=[[0,8],[1,7]],full_reference_pairs=[[0,8],[1,7]],eligible_reference_pairs=[[0,8]],predicted_pairs=[[0,8]]))+'\n')
   rec=m.load_predictions(f)['x'];self.assertIn('eligible_reference',rec);self.assertEqual(len(rec['full_reference']-rec['eligible_reference']),1)
 def test_decoder_retains_crossings_and_excludes_noncanonical(self):
  import numpy as np
  from src.decoder import decode
  s=np.full((9,9),-100.);s[0,6]=s[6,0]=10.;s[2,8]=s[8,2]=9.
  self.assertEqual(set(decode(s,'AAGAAAUAC',.5)),{(0,6),(2,8)})
if __name__=='__main__':unittest.main(verbosity=2)
