import boost_histogram as bh
import boost_histogram.version as bhv


def test_import():
    assert bh


def test_version():
    assert bh.__version__ == bhv.__version__
