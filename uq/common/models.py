import abc
import json
import math
import subprocess
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Optional

import numpy as np

if TYPE_CHECKING:
    from uq.sensitivity import PCESurrogate


@dataclass
class BaseClass:
    def model_dump(self) -> dict[str, Any]:
        return asdict(self)

    def export(self, f: Path) -> None:
        with open(f.__str__(), "w") as fp:
            json.dump(self.model_dump(), fp, indent=3)
