import unittest
import numpy as np
from diagnose_completion_basics import check_invariants
from prepare_dataset import fit_transform, normalize


class AuditTests(unittest.TestCase):
    def test_detects_wrong_partition_and_normalization(self):
        raw=np.array([[-2.,0,0],[2.,0,0],[0.,0,0],[.1,0,0]])
        mask=np.array([False,False,True,True])
        offset,scale=fit_transform(raw[~mask])
        full=normalize(raw,offset,scale)
        sample=dict(target=full,input=full[~mask],local_target=full[mask],mask=mask,
                    center=np.zeros(3),radius=np.array(.2),offset=offset,scale=np.array(scale))
        self.assertTrue(all(check_invariants(sample).values()))
        corrupted={**sample,'local_target':full[mask]+.01}
        self.assertFalse(check_invariants(corrupted)['partition'])
        wrong={**sample,'scale':np.array(-2.)}
        self.assertFalse(check_invariants(wrong)['visible_only_normalization'])


if __name__=='__main__':
    unittest.main()
