import importlib,importlib.util,unittest

class UtilityTests(unittest.TestCase):
    def m(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.direction_a_utility'),'Utility inference missing')
        return importlib.import_module('scripts.direction_a_utility')
    def test_whole_packet_preserves_matching_with_crossings(self):
        m=self.m();base={(0,5),(1,7)};packets=[dict(removed=[(0,5)],added=[(0,6)]),dict(removed=[],added=[(2,8)])]
        self.assertEqual(m.apply_packets(base,packets,[True,False]),{(0,6),(1,7)})
        self.assertEqual(m.apply_packets(base,packets,[False,False]),base)
        with self.assertRaises(ValueError):m.apply_packets(base,[dict(removed=[],added=[(0,6)])],[True])
    def test_training_target_is_change_in_true_pairs_not_new_pairs(self):
        m=self.m();p=dict(removed=[(0,5),(1,6)],added=[(0,6)])
        self.assertEqual(m.target(p,{(1,6)}),-1/3)
        self.assertEqual(m.target(p,{(0,6)}),1/3)
    def test_inference_counts_obey_known_packet_cardinality(self):
        m=self.m();p=dict(removed=[(0,5)],added=[(0,6),(1,7)])
        tp,fp=m.predict_changes(p,100.)
        self.assertEqual(tp,2);self.assertEqual(tp+fp,1)
        tp,fp=m.predict_changes(p,-100.)
        self.assertEqual(tp,-1);self.assertEqual(tp+fp,1)
    def test_features_have_no_truth_argument_and_support_same_sequence_states(self):
        m=self.m();p=dict(removed=[(0,5)],added=[(0,6)])
        scores=[[0.]*9 for _ in range(9)];physical=[[0.]*9 for _ in range(9)]
        f=m.features(p,{(0,5)},scores,physical,[0.]*9)
        self.assertEqual(len(f),18);self.assertEqual(f,m.features(p,{(0,5)},scores,physical,[0.]*9))
        import inspect
        self.assertNotIn('truth',inspect.signature(m.features).parameters)

if __name__=='__main__':unittest.main()
