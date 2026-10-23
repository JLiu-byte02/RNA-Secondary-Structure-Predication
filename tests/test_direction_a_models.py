import unittest
import torch
from scripts.direction_a_models import LocalCorrection,physical_entropy,correction_loss

class ModelTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(17)
    def sample(self):
        base=torch.randn(9,9);base=(base+base.T)/2;base.requires_grad_()
        p=torch.eye(9).roll(4,0)*.2;p=(p+p.T)/2
        valid=torch.ones(9,9,dtype=torch.bool);valid.fill_diagonal_(False)
        return base,p,valid
    def test_convex_bounds_symmetry_and_frozen_base(self):
        s,p,v=self.sample();m=LocalCorrection('local_gate');out,g=m(s,p,v)
        physical=torch.logit(p.clamp(1e-4,1-1e-4))
        self.assertTrue(torch.allclose(out,out.T))
        self.assertTrue(((out[v]>=torch.minimum(s.detach(),physical)[v]-1e-6)&(out[v]<=torch.maximum(s.detach(),physical)[v]+1e-6)).all())
        self.assertTrue(((g>=0)&(g<=1)).all())
        out[v].sum().backward();self.assertIsNone(s.grad)
        self.assertTrue(any(t.grad is not None and t.grad.abs().max()>0 for t in m.parameters()))
    def test_invalid_pairs_have_no_loss_gradient(self):
        s,p,v=self.sample();m=LocalCorrection('safe_local_gate');out,g=m(s,p,v)
        self.assertTrue((out[~v]==-100).all())
        out.retain_grad();ci=torch.tensor([0,1]);cj=torch.tensor([4,5]);labels=torch.tensor([1.,0.])
        loss,details=correction_loss(out,s,ci,cj,labels,.2,.1);loss.backward()
        self.assertTrue((out.grad[~v]==0).all());self.assertTrue(torch.isfinite(loss))
    def test_zero_p_entropy_finite_and_unpaired(self):
        entropy=physical_entropy(torch.zeros(9,9));self.assertTrue(torch.isfinite(entropy).all())
        self.assertLess(float(entropy.abs().max()),1e-5)
    def test_harm_surrogate_increases_on_changes_to_correct_candidates(self):
        base=torch.tensor([[3.,-3.]])
        i=torch.tensor([0,0]);j=torch.tensor([0,1]);labels=torch.tensor([1.,0.])
        _,same=correction_loss(base.clone(),base,i,j,labels,.5,1.)
        _,changed=correction_loss(-base,base,i,j,labels,.5,1.)
        self.assertEqual(same['harm'],0.)
        self.assertGreater(changed['harm'],.8)

if __name__=='__main__':unittest.main()
