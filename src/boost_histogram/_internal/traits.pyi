from typing import Any

class Traits:
    underflow: bool
    overflow: bool
    circular: bool
    growth: bool
    continuous: bool
    ordered: bool
    def __init__(
        self,
        underflow: bool = False,
        overflow: bool = False,
        circular: bool = False,
        growth: bool = False,
        continuous: bool = False,
        ordered: bool = False,
    ): ...
    @property
    def discrete(self) -> bool: ...
    def __eq__(self, other: Any) -> bool: ...
    def __ne__(self, other: Any) -> bool: ...
    def __repr__(self) -> str: ...
