"""Regression checks for the paired inventory and scene-preserving manifest."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from scripts import prepare_mixed_dataset as mixed


class PairedManifestTest(unittest.TestCase):
    def test_large_scenes_are_sampled_without_crossing_splits(self):
        rows = [{'group_id': group, 'noisy_path': f'{group}/{index}',
                 'clean_path': f'{group}/{index}'}
                for group in ('a', 'b', 'c', 'd') for index in range(4)]
        with patch.object(mixed, 'QUOTAS', {'PolyU': {'train': 2, 'val': 1, 'test': 1}}):
            selected = mixed.allocate('PolyU', rows)
        self.assertEqual(len(selected), 4)
        self.assertEqual({split: sum(r['split'] == split for r in selected)
                          for split in ('train', 'val', 'test')},
                         {'train': 2, 'val': 1, 'test': 1})
        for group in ('a', 'b', 'c', 'd'):
            self.assertLessEqual(len({r['split'] for r in selected if r['group_id'] == group}), 1)

    def test_canonical_inventories_create_fixed_grouped_split(self):
        with tempfile.TemporaryDirectory() as directory:
            data = Path(directory) / 'data'
            inventory = data / 'inventory'
            inventory.mkdir(parents=True)
            quotas = {source: {'train': 2, 'val': 1, 'test': 1}
                      for source in ('SIDD', 'PolyU')}
            for source in quotas:
                rows = []
                for index in range(4):
                    for kind in ('LQ', 'GT'):
                        image = data / 'raw' / source / kind / f'{index}.png'
                        image.parent.mkdir(parents=True, exist_ok=True)
                        Image.new('RGB', (128, 128), (index, index, index)).save(image)
                    rows.append({'noisy_path': f'raw/{source}/LQ/{index}.png',
                                 'clean_path': f'raw/{source}/GT/{index}.png',
                                 'group_id': f'scene_{0 if index < 2 else index}'})
                with (inventory / f'{source}.csv').open('w', newline='', encoding='utf8') as file:
                    writer = csv.DictWriter(file, fieldnames=['noisy_path', 'clean_path', 'group_id'])
                    writer.writeheader()
                    writer.writerows(rows)
            output = data / 'splits.csv'
            with patch.object(mixed, 'DATA', data), patch.object(mixed, 'QUOTAS', quotas):
                mixed.prepare(output=output)
                with self.assertRaises(FileExistsError):
                    mixed.prepare(output=output)
            with output.open(newline='', encoding='utf8') as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 8)
            for source in quotas:
                for split, target in quotas[source].items():
                    self.assertEqual(sum(r['source'] == source and r['split'] == split
                                         for r in rows), target)
            self.assertTrue(all((data / r['noisy_path']).is_file() and
                                (data / r['clean_path']).is_file() for r in rows))
            for source in quotas:
                grouped = [r['split'] for r in rows
                           if r['source'] == source and r['group_id'] == 'scene_0']
                self.assertEqual(len(set(grouped)), 1)
            from app.data import DenoiseDataset
            noisy, clean = DenoiseDataset(output, 'train', patch=128)[0]
            self.assertTrue(noisy.equal(clean))


if __name__ == '__main__':
    unittest.main()
