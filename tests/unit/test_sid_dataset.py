"""
SID Dataset Loader Regression Tests
===================================
Covers the loader fixes recorded in CHANGELOG.md:
1. Official list files ('... ISO200 F8') parse without crashing; exposure comes from filenames.
2. `data_root` means the 'Sony/' folder in both list-file and directory-scan modes.
3. `split=` enforces the official SID scene-ID split (0 = train, 1 = test, 2 = val).
4. Amplified Bayer inputs are clipped to AMPLIFIED_INPUT_CEILING (1.0), as in Chen et al.

Pair discovery never opens the raw files, so empty placeholder .ARW files are sufficient here.
"""

import tempfile
import unittest
from pathlib import Path

import torch

from model.datasets.raw_processing import AMPLIFIED_INPUT_CEILING, apply_exposure_amplification
from model.datasets.sid_dataset import DummySIDDataset, SIDSonyDataset, _parse_exposure_time

# (short, long) filename pairs: two train scenes, one test scene, one val scene
_PAIRS = [
    ("00001_00_0.1s.ARW", "00001_00_10s.ARW"),
    ("00002_00_0.04s.ARW", "00002_00_10s.ARW"),
    ("10003_00_0.1s.ARW", "10003_00_10s.ARW"),
    ("20004_00_0.033s.ARW", "20004_00_10s.ARW"),
]


class _FakeSonyTree(unittest.TestCase):
    """Creates <tmp>/Sony/{short,long}/ with placeholder raw files and an official-style list."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.sony = self.root / "Sony"
        (self.sony / "short").mkdir(parents=True)
        (self.sony / "long").mkdir(parents=True)
        lines = []
        for short, long in _PAIRS:
            (self.sony / "short" / short).touch()
            (self.sony / "long" / long).touch()
            lines.append(f"./Sony/short/{short} ./Sony/long/{long} ISO200 F8")
        self.list_file = self.root / "Sony_all_list.txt"
        self.list_file.write_text("\n".join(lines) + "\n")

    def tearDown(self):
        self._tmp.cleanup()


class TestListFileParsing(_FakeSonyTree):
    def test_official_list_with_iso_and_aperture_columns(self):
        """'ISO200 F8' columns must not be parsed as exposure times (previously a ValueError)."""
        dataset = SIDSonyDataset(self.sony, list_file=self.list_file)
        self.assertEqual(len(dataset), len(_PAIRS))
        short_p, long_p, short_exp, long_exp = dataset.file_pairs[0]
        self.assertEqual((short_exp, long_exp), (0.1, 10.0))

    def test_list_paths_resolve_against_sony_folder(self):
        """'./Sony/short/x.ARW' resolves to '<data_root>/short/x.ARW', not '<data_root>/Sony/short/...'."""
        dataset = SIDSonyDataset(self.sony, list_file=self.list_file)
        for short_p, long_p, _, _ in dataset.file_pairs:
            self.assertTrue(short_p.is_file(), short_p)
            self.assertTrue(long_p.is_file(), long_p)

    def test_missing_list_file_fails_loudly(self):
        with self.assertRaises(FileNotFoundError):
            SIDSonyDataset(self.sony, list_file=self.root / "does_not_exist.txt")


class TestOfficialSplit(_FakeSonyTree):
    def test_directory_scan_split_has_no_overlap(self):
        """Train/val/test scene IDs must be disjoint (previously val was a subset of train)."""
        splits = {
            name: {pair[0].name for pair in SIDSonyDataset(self.sony, split=name).file_pairs}
            for name in ("train", "val", "test")
        }
        self.assertEqual(splits["train"], {"00001_00_0.1s.ARW", "00002_00_0.04s.ARW"})
        self.assertEqual(splits["test"], {"10003_00_0.1s.ARW"})
        self.assertEqual(splits["val"], {"20004_00_0.033s.ARW"})

    def test_split_applies_to_list_files_too(self):
        dataset = SIDSonyDataset(self.sony, list_file=self.list_file, split="test")
        self.assertEqual([p[0].name for p in dataset.file_pairs], ["10003_00_0.1s.ARW"])

    def test_unknown_split_rejected(self):
        with self.assertRaises(ValueError):
            SIDSonyDataset(self.sony, split="training")

    def test_wrong_data_root_fails_loudly(self):
        """Pointing data_root at the parent of 'Sony/' finds nothing and must raise, not train on 0 pairs."""
        with self.assertRaises(ValueError):
            SIDSonyDataset(self.root, split="train")


class TestExposureParsing(unittest.TestCase):
    def test_parses_fractional_and_integer_exposures(self):
        self.assertEqual(_parse_exposure_time("00001_00_0.1s.ARW"), 0.1)
        self.assertEqual(_parse_exposure_time("10003_00_10s.ARW"), 10.0)

    def test_unparseable_name_raises(self):
        """A silent 1.0 default would corrupt the amplification ratio."""
        with self.assertRaises(ValueError):
            _parse_exposure_time("ISO200")


class TestInputClipping(unittest.TestCase):
    def test_dummy_dataset_inputs_are_clipped(self):
        bayer_in, _, _ = DummySIDDataset(num_samples=2, patch_size=64)[0]
        self.assertLessEqual(float(bayer_in.max()), AMPLIFIED_INPUT_CEILING)

    def test_apply_exposure_amplification_clamps_and_broadcasts(self):
        packed = torch.full((2, 4, 8, 8), 0.004)
        ratio = torch.tensor([[100.0], [300.0]])  # (B, 1): previously broadcast against (H, W)
        out = apply_exposure_amplification(packed, ratio)
        self.assertEqual(out.shape, (2, 4, 8, 8))
        self.assertAlmostEqual(float(out[0].max()), 0.4, places=5)
        self.assertEqual(float(out[1].max()), AMPLIFIED_INPUT_CEILING)  # 1.2 clipped to 1.0

    def test_apply_exposure_amplification_unclamped(self):
        out = apply_exposure_amplification(torch.full((1, 4, 2, 2), 0.01), 300.0, max_value=None)
        self.assertAlmostEqual(float(out.max()), 3.0, places=5)


if __name__ == "__main__":
    unittest.main()
