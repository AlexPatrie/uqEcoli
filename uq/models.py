from enum import StrEnum
from typing import cast


class StrEnumBase(StrEnum):
    @classmethod
    def keys(cls) -> list[str]:
        return cast(list[str], vars(cls)["_member_names_"])

    @classmethod
    def member_keys(cls) -> list[str]:
        return cast(list[str], vars(cls)["_member_names_"])

    @classmethod
    def values(cls) -> list[str]:
        vals: list[str] = []
        for key in cls.member_keys():
            val = getattr(cls, key, None)
            if val is not None:
                vals.append(val)
        return vals

    @classmethod
    def to_dict(cls) -> dict[str, str]:
        return dict(zip(cls.keys(), cls.values()))

    @classmethod
    def to_list(cls, sort: bool = False) -> list[str]:
        vals = cls.values()
        return sorted(vals) if sort else vals


class CliType(StrEnumBase):
    SIMULATOR = "simulator"
    SIMULATION = "simulation"
    PARCA = "parca"
    ANALYSIS = "analysis"
    DEMO = "demo"
    HELP = "help"
    TUI = "tui"


class ApiBaseUrl(StrEnumBase):
    RKE_PROD = "https://sms.cam.uchc.edu"
    RKE_DEV = "https://sms-dev.cam.uchc.edu"
    LOCAL_8888 = "http://localhost:8888"
    LOCAL_8000 = "http://localhost:8000"
    LOCAL_1111 = "http://localhost:1111"
    LOCAL_62505 = "http://localhost:62505"
    LOCAL_8080 = "http://localhost:8080"