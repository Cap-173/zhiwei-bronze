from pathlib import Path
import tempfile
import unittest
import json
from object_cross_validation import prepare, grouped_folds
from training_data import read_manifest


class ObjectCVTests(unittest.TestCase):
    def test_group_coverage_and_determinism(self):
        ids=[f'm{i}' for i in range(12)]
        folds=grouped_folds(ids)
        self.assertEqual(folds,grouped_folds(ids[::-1]))
        self.assertEqual(sorted(mid for fold in folds for mid in fold),sorted(ids))
        self.assertEqual([len(fold) for fold in folds],[3]*4)
        with self.assertRaises(ValueError):
            grouped_folds(['a','a'],2)

    def test_test_files_never_read_and_data_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'source';source.mkdir()
            entries=[]
            for i in range(6):
                directory=source/f'm{i}';directory.mkdir()
                (directory/'training.npz').write_bytes(f'unchanged payload {i}'.encode())
                entries.append(dict(model_id=f'm{i}',split='train' if i<4 else 'val',path=f'm{i}'))
            # The original test sample deliberately has no file: any read fails.
            entries.append(dict(model_id='excluded',split='test',path='missing_test'))
            (source/'manifest.json').write_text(json.dumps(dict(status='complete',samples=entries,seed=42)))
            protocol=prepare(source,root/'cv',3)
            heldout=[]
            for fold in protocol['folds']:
                self.assertFalse(set(fold['train_models'])&set(fold['heldout_models']))
                self.assertNotIn('excluded',fold['train_models']+fold['heldout_models'])
                heldout+=fold['heldout_models']
                data=read_manifest(Path(fold['dataset']))
                for entry in data['samples']:
                    self.assertEqual((Path(fold['dataset'])/entry['path']/'training.npz').read_bytes(),
                                     (source/entry['original_sample_path']/'training.npz').read_bytes())
            self.assertEqual(sorted(heldout),[f'm{i}' for i in range(6)])
            with self.assertRaises(FileExistsError):
                prepare(source,root/'cv',3)


if __name__=='__main__':
    unittest.main()
