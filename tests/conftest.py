import pytest
import torch

from khmer_llm.runtime import setup_runtime
from khmer_llm.selftest import tiny_config


@pytest.fixture(scope="session")
def rt():
    return setup_runtime(allow_ddp=False)


@pytest.fixture
def cfg():
    torch.manual_seed(0)
    return tiny_config()
