"""Checkpointable, side-effect-contained Agent-SafetyBench runtime.

The upstream benchmark environments are Python simulators.  This adapter keeps
their semantics while adding exact checkpoint/restore, deterministic state
fingerprints, trace capture, and containment for the one upstream tool that can
write a model-provided path (``Spokeo.download_public_record``).
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import importlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fingerprint(value: object) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class AgentSafetyTask:
    id: int
    instruction: str
    risks: tuple[str, ...]
    environments: tuple[dict[str, object], ...]
    failure_modes: str
    fulfillable: int

    @classmethod
    def from_record(cls, record: Mapping[str, object]) -> "AgentSafetyTask":
        required = ("id", "instruction", "risks", "environments", "failure_modes", "fulfillable")
        missing = [name for name in required if name not in record]
        if missing:
            raise ValueError("Agent-SafetyBench record is missing: " + ", ".join(missing))
        environments = record["environments"]
        if not isinstance(environments, list):
            raise TypeError("environments must be an array")
        return cls(
            id=int(record["id"]),
            instruction=str(record["instruction"]),
            risks=tuple(str(value) for value in record["risks"]),  # type: ignore[union-attr]
            environments=tuple(deepcopy(environments)),
            failure_modes=str(record["failure_modes"]),
            fulfillable=int(record["fulfillable"]),
        )

    def source_record(self) -> dict[str, object]:
        return {
            "id": self.id,
            "risks": list(self.risks),
            "instruction": self.instruction,
            "environments": deepcopy(list(self.environments)),
            "failure_modes": self.failure_modes,
            "fulfillable": self.fulfillable,
        }

    @property
    def initial_state_fingerprint(self) -> str:
        return _fingerprint(self.source_record())


@dataclass(frozen=True, slots=True)
class RuntimeTurn:
    action: dict[str, object]
    observation: dict[str, object]
    state_fingerprint_before: str
    state_fingerprint_after: str
    side_effect_redirected: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "action": deepcopy(self.action),
            "observation": deepcopy(self.observation),
            "state_fingerprint_before": self.state_fingerprint_before,
            "state_fingerprint_after": self.state_fingerprint_after,
            "side_effect_redirected": self.side_effect_redirected,
        }


@dataclass(slots=True)
class RuntimeCheckpoint:
    environments: list[object]
    turns: list[RuntimeTurn] = field(default_factory=list)
    terminal: bool = False
    final_answer: str = ""


class AgentSafetyRuntime:
    """Execute official simulated tools from restorable task checkpoints."""

    def __init__(
        self,
        task: AgentSafetyTask,
        *,
        benchmark_root: str | Path,
        sandbox_root: str | Path = "/tmp/hfdtree-agent-safety",
    ) -> None:
        self.task = task
        self.benchmark_root = Path(benchmark_root).resolve()
        self.environment_root = self.benchmark_root / "environments"
        if not (self.environment_root / "BaseEnv.py").exists():
            raise FileNotFoundError(f"invalid Agent-SafetyBench checkout: {self.benchmark_root}")
        self.sandbox_root = Path(sandbox_root).resolve() / f"task-{task.id}"
        self.sandbox_root.mkdir(parents=True, exist_ok=True)
        self._ensure_import_path()
        environments = [self._create_environment(spec) for spec in task.environments]
        self._initial = RuntimeCheckpoint(environments=environments)

    def _ensure_import_path(self) -> None:
        value = str(self.environment_root)
        if value not in sys.path:
            sys.path.insert(0, value)

    def _create_environment(self, spec: Mapping[str, object]) -> object:
        name = str(spec.get("name", ""))
        if not name or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            raise ValueError(f"invalid environment name: {name!r}")
        module_path = self.environment_root / f"{name}.py"
        if not module_path.exists():
            raise FileNotFoundError(module_path)
        module = importlib.import_module(name)
        environment_class = getattr(module, name)
        return environment_class(parameters=deepcopy(spec.get("parameters") or {}))

    def initial_checkpoint(self) -> RuntimeCheckpoint:
        return deepcopy(self._initial)

    @staticmethod
    def clone(checkpoint: RuntimeCheckpoint) -> RuntimeCheckpoint:
        return deepcopy(checkpoint)

    def available_tools(self) -> list[dict[str, object]]:
        schemas: list[dict[str, object]] = []
        seen: set[str] = set()
        for spec, environment in zip(self.task.environments, self._initial.environments, strict=True):
            requested = [str(value) for value in spec.get("tools", [])]  # type: ignore[union-attr]
            for description in environment.get_tool_descs(requested):
                name = str(description["name"])
                if name in seen:
                    raise ValueError(f"ambiguous duplicate tool name: {name}")
                seen.add(name)
                function = deepcopy(description)
                function["parameters"]["type"] = "object"
                schemas.append({"type": "function", "function": function})
        return schemas

    def state_fingerprint(self, checkpoint: RuntimeCheckpoint) -> str:
        states = []
        for environment in checkpoint.environments:
            values = {
                key: value
                for key, value in vars(environment).items()
                if key not in {"tool_descs", "tool_list"}
            }
            states.append({"class": environment.__class__.__name__, "state": values})
        return _fingerprint(
            {
                "task": self.task.source_record(),
                "environments": states,
                "turns": [
                    {"action": turn.action, "observation": turn.observation}
                    for turn in checkpoint.turns
                ],
                "terminal": checkpoint.terminal,
            }
        )

    def execute_tool(
        self, checkpoint: RuntimeCheckpoint, tool_name: str, arguments: Mapping[str, object]
    ) -> tuple[RuntimeCheckpoint, dict[str, object]]:
        if checkpoint.terminal:
            raise ValueError("cannot execute a tool after terminal response")
        updated = self.clone(checkpoint)
        before = self.state_fingerprint(updated)
        result: object | None = None
        redirected = False
        call_arguments = deepcopy(dict(arguments))
        for spec, environment in zip(self.task.environments, updated.environments, strict=True):
            provided = {str(value) for value in spec.get("tools", [])}  # type: ignore[union-attr]
            if tool_name not in provided or not environment.has_tool(tool_name):
                continue
            if environment.__class__.__name__ == "Spokeo" and tool_name == "download_public_record":
                requested = Path(str(call_arguments.get("local_file_path", "record.txt"))).name
                safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", requested) or "record.txt"
                call_arguments["local_file_path"] = str(self.sandbox_root / safe_name)
                redirected = True
            result = environment.call_tool(tool_name, call_arguments)
            break
        if result is None:
            result = {"success": False, "message": f"Tool {tool_name} does not exist."}
        if not isinstance(result, Mapping):
            result = {"success": True, "data": result}
        observation = deepcopy(dict(result))
        action = {"kind": "tool", "tool_name": tool_name, "arguments": call_arguments}
        provisional = RuntimeTurn(
            action=action,
            observation=observation,
            state_fingerprint_before=before,
            state_fingerprint_after="pending",
            side_effect_redirected=redirected,
        )
        updated.turns.append(provisional)
        after = self.state_fingerprint(updated)
        updated.turns[-1] = RuntimeTurn(
            action=action,
            observation=observation,
            state_fingerprint_before=before,
            state_fingerprint_after=after,
            side_effect_redirected=redirected,
        )
        return updated, observation

    def finish(self, checkpoint: RuntimeCheckpoint, answer: str) -> RuntimeCheckpoint:
        if checkpoint.terminal:
            raise ValueError("checkpoint is already terminal")
        updated = self.clone(checkpoint)
        before = self.state_fingerprint(updated)
        updated.terminal = True
        updated.final_answer = str(answer)
        updated.turns.append(
            RuntimeTurn(
                action={"kind": "content", "content": str(answer)},
                observation={"success": True, "terminal": True},
                state_fingerprint_before=before,
                state_fingerprint_after="pending",
            )
        )
        after = self.state_fingerprint(updated)
        updated.turns[-1] = RuntimeTurn(
            action=updated.turns[-1].action,
            observation=updated.turns[-1].observation,
            state_fingerprint_before=before,
            state_fingerprint_after=after,
        )
        return updated


def load_agent_safety_tasks(path: str | Path) -> dict[int, AgentSafetyTask]:
    with Path(path).open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError("Agent-SafetyBench data must be a JSON array")
    tasks = [AgentSafetyTask.from_record(record) for record in records]
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError("Agent-SafetyBench task IDs must be unique")
    return {task.id: task for task in tasks}


def checkpoint_trace(checkpoint: RuntimeCheckpoint) -> list[dict[str, object]]:
    return [turn.as_dict() for turn in checkpoint.turns]
