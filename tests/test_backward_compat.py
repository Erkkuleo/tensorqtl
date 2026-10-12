"""map_nominal without joint_tests must give the same output as master.

Needs a separate checkout of master:
    git clone --branch master https://github.com/Erkkuleo/tensorqtl.git /content/tensorqtl_master
    TENSORQTL_MASTER_DIR=/content/tensorqtl_master pytest tests/test_backward_compat.py
"""
import glob
import gzip
import os
import subprocess
import sys

import pandas as pd
import pytest

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RUNNER = os.path.join(os.path.dirname(__file__), 'compat_runner.py')
DATA_DIR = os.path.join(REPO, 'example', 'data')
MASTER = os.environ.get('TENSORQTL_MASTER_DIR')

pytestmark = pytest.mark.skipif(not MASTER, reason='set TENSORQTL_MASTER_DIR to a checkout of master')


# ni=2 uses maf_threshold_interaction=0: with cos_t/sin_t columns and a threshold > 0,
# this branch adds the per-quadrant carrier filter (an intended change; master
# applied no interaction MAF filter for > 1 interaction).
@pytest.mark.parametrize('ni,maf_threshold_interaction', [(1, 0.05), (2, 0)])
def test_map_nominal_matches_master(tmp_path, ni, maf_threshold_interaction):
    pytest.importorskip('pgenlib')
    pytest.importorskip('pyarrow')
    out = {}
    for name, repo in [('master', MASTER), ('branch', REPO)]:
        out[name] = str(tmp_path / name)
        subprocess.run([sys.executable, RUNNER, '--repo', repo, '--data-dir', DATA_DIR,
                        '--out-dir', out[name], '--ni', str(ni),
                        '--maf-threshold-interaction', str(maf_threshold_interaction)],
                       check=True)

    parquets = sorted(os.path.basename(p) for p in glob.glob(os.path.join(out['master'], '*.parquet')))
    assert parquets
    assert parquets == sorted(os.path.basename(p) for p in glob.glob(os.path.join(out['branch'], '*.parquet')))
    for f in parquets:
        pd.testing.assert_frame_equal(pd.read_parquet(os.path.join(out['master'], f)),
                                      pd.read_parquet(os.path.join(out['branch'], f)),
                                      check_exact=True)

    top = 'compat.cis_qtl_top_assoc.txt.gz'
    with gzip.open(os.path.join(out['master'], top)) as a, gzip.open(os.path.join(out['branch'], top)) as b:
        assert a.read() == b.read()
