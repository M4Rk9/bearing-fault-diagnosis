"""Synthetic fixtures test behavior; they are never diagnostic evidence."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import savemat

from src.data_pipeline import (acquire, digest, make_windows, prepare, read_signal,
                               validate_manifest)

EXAMPLE = json.loads(Path('data/manifests/real_record_example.json').read_text())[0]


def row(**changes):
    result = copy.deepcopy(EXAMPLE)
    result['sha256'] = 'a' * 64
    result.update(changes)
    return result


class ManifestTests(unittest.TestCase):
    def test_group_cannot_cross_partitions(self):
        with self.assertRaisesRegex(ValueError, 'group'):
            validate_manifest([row(), row(record_id='copy', partition='test', channel_key='X105_FE_time', sensor_end='FE')], require_coverage=False)

    def test_duplicate_file_cannot_be_relabelled_as_new_group(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate file'):
            validate_manifest([row(), row(record_id='copy', group_id='new')], require_coverage=False)

    def test_missing_coverage_rejected(self):
        with self.assertRaisesRegex(ValueError, 'four classes'):
            validate_manifest([row()])

    def test_invalid_metadata(self):
        for changes in ({'sampling_rate_hz':None}, {'sampling_rate_source':None},
                        {'sampling_rate_hz':True}, {'channel_key':'X105_FE_time'},
                        {'sha256':'xyz'}, {'relative_path':'../105.mat'}, {'rpm':float('nan')},
                        {'fault_diameter_in':-1}, {'class_name':'outer_race'},
                        {'units':'g'}, {'partition':'holdout'}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_manifest([row(**changes)], require_coverage=False)

    def test_candidates_fail_closed_on_healthy_sampling_rate(self):
        candidates = json.loads(Path('data/manifests/cwru_007_candidates.json').read_text())
        with self.assertRaisesRegex(ValueError, 'sampling rate'):
            validate_manifest(candidates, require_hash=False)


class SignalTests(unittest.TestCase):
    def test_explicit_key_and_invalid_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'sample.mat'
            savemat(path, {'other':np.arange(3000)})
            with self.assertRaisesRegex(ValueError, 'Missing requested'):
                read_signal(path, 'X105_DE_time')
            for value in (np.ones(3000), np.array([1.,np.nan]), np.ones((10,10)),
                          np.array([1+2j, 3+4j]), np.array([1.])):
                savemat(path, {'X105_DE_time':value})
                with self.assertRaises(ValueError):
                    read_signal(path, 'X105_DE_time')

    def test_boundaries_and_amplitude_branches(self):
        x=np.sin(2*np.pi*np.arange(4097)/64)
        f,c,s,length=make_windows(x,12000)
        self.assertEqual(s.tolist(),[0,1024,2048])
        self.assertEqual(length,4097)
        f2,c2,_,_=make_windows(3*x+20,12000)
        np.testing.assert_allclose(f2,3*f,atol=1e-13)
        np.testing.assert_allclose(c,c2,atol=1e-13)
        self.assertAlmostEqual(float(np.sqrt(np.mean(f*f))),1/np.sqrt(2))
        for bad in (np.ones(4096), x[:20], np.zeros((2,2048))):
            with self.assertRaises(ValueError):
                make_windows(bad,12000)

    def test_parameter_validation(self):
        for kwargs in ({'hop_size':0},{'window_size':0},{'target_rate':0},
                       {'hop_size':3000},{'window_size':2.5}):
            with self.assertRaises(ValueError):
                make_windows(np.arange(4096),12000,**kwargs)

    def test_anti_alias_and_equal_duration(self):
        t=np.arange(48000)/48000
        low=np.sin(2*np.pi*1000*t)
        high=np.sin(2*np.pi*10000*t)
        a,_,starts,length=make_windows(low,48000)
        b,_,_,_=make_windows(high,48000)
        self.assertEqual(length,12000)
        self.assertEqual(a.shape[1],2048)
        self.assertLess(np.sqrt(np.mean(b[1:-1]**2)),0.01*np.sqrt(np.mean(a[1:-1]**2)))


class IntegrationTests(unittest.TestCase):
    def test_prepare_provenance_and_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            signal=np.sin(np.arange(5000)/10)
            savemat(root/'105.mat',{'X105_DE_time':signal})
            r=row(sha256=digest(root/'105.mat'))
            summary=prepare([r],root,root/'out',require_coverage=False)
            self.assertFalse(summary['classification_ready'])
            with np.load(root/'out/train.npz',allow_pickle=False) as data:
                self.assertEqual(set(data['group_ids']),{'cwru-105'})
                self.assertEqual(data['start_samples'].tolist(),[0,1024,2048])
            with self.assertRaisesRegex(ValueError,'empty'):
                prepare([r],root,root/'out',require_coverage=False)
            savemat(root/'105.mat',{'X105_DE_time':signal*2})
            with self.assertRaisesRegex(ValueError,'Hash mismatch'):
                prepare([r],root,root/'out2',require_coverage=False)
            self.assertFalse((root/'out2').exists())

    def test_repacked_duplicate_signal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            x=np.sin(np.arange(5000)/10)
            savemat(root/'105.mat',{'X105_DE_time':x})
            savemat(root/'copy.mat',{'X105_DE_time':x,'extra':np.array([3])})
            rows=[row(sha256=digest(root/'105.mat')),
                  row(record_id='copy',group_id='copy',relative_path='copy.mat',partition='test',sha256=digest(root/'copy.mat'))]
            with self.assertRaisesRegex(ValueError,'Identical channel'):
                prepare(rows,root,root/'out',require_coverage=False)

    def test_acquire_existing_file_and_hash_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            savemat(root/'105.mat',{'X105_DE_time':np.arange(5000)})
            locked=acquire([row(sha256=None)],root)
            self.assertEqual(locked[0]['sha256'],digest(root/'105.mat'))
            with self.assertRaisesRegex(ValueError,'Hash mismatch'):
                acquire([row()],root)


if __name__=='__main__':
    unittest.main()
