import importlib,importlib.util,unittest

class ErrorMechanismTests(unittest.TestCase):
    def test_crossing_pairs_distinguish_pseudoknot_from_nested_stems(self):
        self.assertIsNotNone(importlib.util.find_spec('scripts.analyze_direction_a_errors'),'Structure error analysis is missing')
        fn=importlib.import_module('scripts.analyze_direction_a_errors').crossing_pairs
        self.assertEqual(fn({(0,4),(2,6),(7,9)}),{(0,4),(2,6)})
        self.assertEqual(fn({(0,9),(2,7),(3,6)}),set())

if __name__=='__main__':unittest.main()
