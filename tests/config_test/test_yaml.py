import pytest
import yaml

from gentopia.assembler.loader import Loader


class Test01:

    @pytest.mark.parametrize("file", ["data/test1.yaml"])
    def test_yaml(self, file):
        with open(file, 'r') as f:
            data = yaml.load(f, Loader)
            assert data['a'] == 1

    @pytest.mark.parametrize("file", ["data/test2.yaml"])
    def test_include(self, file):
        with open(file, 'r') as f:
            data = yaml.load(f, Loader)
            assert data['c']['a'] == 1
