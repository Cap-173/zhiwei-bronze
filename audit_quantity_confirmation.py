"""Check saved paired-seed results, metric aggregation and report image links."""
import json
from pathlib import Path
import re

import numpy as np

from main import ROOT
from run_quantity_confirmation import checkpoint_path, validate_checkpoint
from run_quantity_experiment import dump


def main():
    output = ROOT / 'outputs/repeats/bronze_quantity_confirmation'
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    protocol = summary['protocol']
    evaluations, geometry = summary['evaluations'], summary['geometry']
    rows = 0
    for seed in [42] + protocol['confirmation_seeds']:
        for count in protocol['counts']:
            validate_checkpoint(checkpoint_path(count, seed), count, seed, protocol)
            label = f'{count}_seed{seed}'
            for role, entries in protocol['roles'].items():
                expected = [entry['path'] for entry in entries]
                scores = evaluations[label][role]['samples']
                quality = geometry[label][role]['samples']
                assert [r['path'] for r in scores] == expected
                assert [r['path'] for r in quality] == expected
                for r in scores:
                    values = [r[k] for k in ('local_cd_squared', 'local_fscore', 'precision', 'recall')]
                    assert np.isfinite(values).all() and values[0] >= 0
                    assert all(0 <= v <= 1 for v in values[1:])
                for r in quality:
                    assert np.isfinite([r['close_fraction'], r['mesh_distance_over_R']]).all()
                    assert 0 <= r['close_fraction'] <= 1 and r['mesh_distance_over_R'] >= 0
                assert np.isclose(np.mean([r['local_cd_squared'] for r in scores]), evaluations[label][role]['mean_local_cd_squared'])
                assert np.isclose(np.mean([r['local_fscore'] for r in scores]), evaluations[label][role]['mean_local_fscore'])
                rows += len(scores)
    for scope, seeds in [('confirmation_only', [43, 44]), ('all_three', [42, 43, 44])]:
        for role in protocol['roles']:
            for count in protocol['counts']:
                aggregate = summary['aggregates'][scope][role][str(count)]
                for field, result in aggregate.items():
                    values = [evaluations[f'{count}_seed{seed}'][role][field] if field.startswith('mean_local_')
                              else geometry[f'{count}_seed{seed}'][role]['means'][field] for seed in seeds]
                    assert np.allclose(values, result['values'])
                    assert np.isclose(np.mean(values), result['mean'])
                    assert np.isclose(np.std(values), result['std_population'])
    text = (output / 'README.md').read_text(encoding='utf-8')
    links = re.findall(r'!\[[^\]]*\]\(([^)]+)\)', text)
    assert len(links) == 5 and all((output / link).resolve().is_file() for link in links)
    result = dict(status='pass', checkpoints=6, updates_per_checkpoint=800, metric_sample_rows=rows,
                  finite_geometry_sample_rows=rows, checked_report_images=len(links),
                  scope='Numerical and artifact consistency; not a guarantee of surface correctness')
    dump(output / 'audit.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
