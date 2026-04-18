import qdatoolkit as qda
from qdatoolkit._version import __version__ as package_version


def test_package_version_is_exposed():
    assert qda.__version__ == package_version
    assert isinstance(qda.__version__, str)
    assert qda.__version__
