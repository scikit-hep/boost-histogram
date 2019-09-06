import boost.histogram as bh
import boost.histogram_version as bhv


def test_import():
    assert bh


def test_version():
    assert bh.__version__ == bhv.__version__
