import importlib,importlib.util,unittest

class EditPacketTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.direction_a_edit_packets'),'Edit-packet diagnostic is missing')
        return importlib.import_module('scripts.direction_a_edit_packets')
    def test_alternating_cycle_is_one_atomic_edit(self):
        m=self.module();base={(0,4),(2,6)};proposal={(0,6),(2,4)}
        packets=m.edit_packets(base,proposal);self.assertEqual(len(packets),1)
        self.assertEqual(set(packets[0]['removed']),base);self.assertEqual(set(packets[0]['added']),proposal)
    def test_stem_merge_can_bridge_an_unchanged_pair(self):
        m=self.module();base={(1,8)};proposal={(0,9),(1,8),(2,7)}
        self.assertEqual(len(m.edit_packets(base,proposal,False)),2)
        self.assertEqual(len(m.edit_packets(base,proposal,True)),1)
    def test_oracle_keeps_good_swap_and_rejects_false_addition(self):
        m=self.module();base={(0,5),(1,4)};proposal={(0,6),(1,4),(2,8)};truth={(0,6),(1,4)}
        result=m.oracle(base,proposal,truth,False)
        self.assertEqual(set(result['prediction']),truth);self.assertEqual(result['f1'],1.)
    def test_exact_f1_oracle_does_not_confuse_f1_with_net_correct_edits(self):
        m=self.module();base={(0,100)}|{(200+i,230+i) for i in range(10)}
        proposal=base|{(10+i,50-i) for i in range(10)};truth={(0,100),(10,50)}|{(300+i,330+i) for i in range(8)}
        result=m.oracle(base,proposal,truth,True)
        self.assertEqual(set(result['prediction']),proposal);self.assertAlmostEqual(result['f1'],4/31)

if __name__=='__main__':unittest.main()
