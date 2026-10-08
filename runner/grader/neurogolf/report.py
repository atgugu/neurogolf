from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class CostReport:
    macs: Optional[int] = None  # retained for legacy cache compatibility; no longer in cost
    memory_bytes: Optional[int] = None
    params: Optional[int] = None
    file_size_bytes: int = 0
    disallowed_ops: list[str] = field(default_factory=list)
    opset_version: Optional[int] = None
    profile_error: Optional[str] = None
    onnx_check_error: Optional[str] = None

    @property
    def cost(self) -> Optional[int]:
        # May-2026 formula: cost = max(1, memory + params); MACs no longer counted.
        if self.memory_bytes is None or self.params is None:
            return None
        return max(1, self.memory_bytes + self.params)

    @property
    def ok(self) -> bool:
        return (
            self.cost is not None
            and not self.disallowed_ops
            and self.profile_error is None
            and self.onnx_check_error is None
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CostReport":
        return cls(**d)


@dataclass
class ValidationReport:
    valid: bool = False
    per_split: dict[str, tuple[int, int]] = field(default_factory=dict)
    first_failure: Optional[dict] = None
    io_name_error: Optional[str] = None
    nan_or_inf: bool = False
    onnxruntime_error: Optional[str] = None
    runtime_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["per_split"] = {k: list(v) for k, v in self.per_split.items()}
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ValidationReport":
        d = dict(d)
        d["per_split"] = {k: tuple(v) for k, v in d.get("per_split", {}).items()}
        return cls(**d)


@dataclass
class ScoreReport:
    task_id: Optional[str]
    onnx_sha256: str
    cost: CostReport
    validation: ValidationReport
    points: float
    submitted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "onnx_sha256": self.onnx_sha256,
            "cost": self.cost.to_dict(),
            "validation": self.validation.to_dict(),
            "points": self.points,
            "submitted": self.submitted,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScoreReport":
        return cls(
            task_id=d.get("task_id"),
            onnx_sha256=d["onnx_sha256"],
            cost=CostReport.from_dict(d["cost"]),
            validation=ValidationReport.from_dict(d["validation"]),
            points=float(d["points"]),
            submitted=bool(d.get("submitted", False)),
        )
