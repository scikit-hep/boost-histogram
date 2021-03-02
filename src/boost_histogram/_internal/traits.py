from dataclasses import dataclass


@dataclass(order=True, frozen=True)
class Traits:
    underflow: bool = False
    overflow: bool = False
    circular: bool = False
    growth: bool = False
    continuous: bool = False
    ordered: bool = False

    @property
    def discrete(self) -> bool:
        "True if axis is not continuous"
        return not self.continuous
