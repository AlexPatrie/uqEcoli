from uq.common.models import BaseClass
from uq.common.utils import get_repo_root

PARAM_CONFIG_DEMO = get_repo_root() / "examples/uq_artifacts/params/params_demo.json"

__all__ = ["get_repo_root", "BaseClass", "PARAM_CONFIG_DEMO"]
