import copy
import unittest
from scripts.direction_a_protocol import family_blocks,group_calibration,pilot_roles,validate_roles

class RolesTests(unittest.TestCase):
    def records(self):
        return {f'{f}_{i}':dict(family=f,sequence=f'{f}_{i//2}') for f in ('5s','16s','23s','tRNA','srp','tmRNA','RNaseP','grp1','telomerase') for i in range(12)}
    def test_oof_and_development_are_family_disjoint(self):
        records=self.records();roles=pilot_roles(records,list(records),'5s','srp')
        self.assertTrue(validate_roles(records,roles))
        a,b=set(roles['block_a']),set(roles['block_b'])
        self.assertFalse(a&b)
        self.assertEqual({records[s]['family'] for s in roles['development']},{'srp'})
        self.assertNotIn('5s',{records[s]['family'] for key in ('block_a','block_b','development') for s in roles[key]})
        bad=copy.deepcopy(roles);bad['block_a'].append(roles['development'][0])
        with self.assertRaises(AssertionError):validate_roles(records,bad)
    def test_identical_sequences_do_not_cross_calibration(self):
        records=self.records();fit,cal=group_calibration(records,list(records),.2)
        self.assertFalse({records[s]['sequence'] for s in fit}&{records[s]['sequence'] for s in cal})
        self.assertEqual(set(fit+cal),set(records))
        self.assertEqual((fit,cal),group_calibration(records,list(reversed(records)),.2))
    def test_cross_family_duplicate_fails(self):
        records=self.records();records['16s_0']['sequence']=records['tRNA_0']['sequence']
        with self.assertRaises(ValueError):group_calibration(records,list(records),.2)

if __name__=='__main__':unittest.main()
