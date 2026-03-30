from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class S0Config:

    n_steps: int = 20
    lr: float = 1e-3
    l2_lambda: float = 5e-4
    alpha: float | None = None
    normalize: bool = False
    grad_clip: float = 1.0
    max_length: int = 2048

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> S0Config:
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> S0Config:
        return cls.from_dict(json.loads(Path(path).read_text()))
