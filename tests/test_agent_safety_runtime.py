import json

from hfdtree.agent_safety_runtime import AgentSafetyRuntime, AgentSafetyTask
from hfdtree.production_rollout import ExplicitBranchCollector, GeneratedAction, parse_generated_action
from hfdtree.trajectory_judge import parse_judgment


def _benchmark(tmp_path):
    root = tmp_path / "benchmark"
    environments = root / "environments"
    environments.mkdir(parents=True)
    (environments / "BaseEnv.py").write_text(
        """import json, os
class BaseEnv:
    def __init__(self):
        path=os.path.join(os.path.dirname(__file__), self.__class__.__name__+'.json')
        self.tool_descs=json.load(open(path))
        self.tool_list=[x['name'] for x in self.tool_descs]
    def has_tool(self,name): return name in self.tool_list
    def get_tool_descs(self,names): return [x for x in self.tool_descs if x['name'] in names]
    def call_tool(self,name,args): return getattr(self,name)(**args)
""",
        encoding="utf-8",
    )
    (environments / "Counter.json").write_text(
        json.dumps(
            [
                {
                    "name": "increment",
                    "description": "increment a simulated value",
                    "parameters": {
                        "type": "object",
                        "properties": {"amount": {"type": "integer"}},
                        "required": ["amount"],
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    (environments / "Counter.py").write_text(
        """from BaseEnv import BaseEnv
class Counter(BaseEnv):
    def __init__(self, *, parameters=None):
        super().__init__(); self.value=(parameters or {}).get('value', 0)
    def increment(self, *, amount):
        self.value += amount; return {'success': True, 'value': self.value}
""",
        encoding="utf-8",
    )
    return root


def _task():
    return AgentSafetyTask.from_record(
        {
            "id": 1,
            "instruction": "increment",
            "risks": ["none"],
            "environments": [
                {"name": "Counter", "tools": ["increment"], "parameters": {"value": 0}}
            ],
            "failure_modes": "",
            "fulfillable": 1,
        }
    )


def test_checkpoint_restore_and_state_fingerprints(tmp_path) -> None:
    runtime = AgentSafetyRuntime(_task(), benchmark_root=_benchmark(tmp_path))
    first = runtime.initial_checkpoint()
    second = runtime.clone(first)
    assert runtime.state_fingerprint(first) == runtime.state_fingerprint(second)
    advanced, observation = runtime.execute_tool(second, "increment", {"amount": 2})
    assert observation == {"success": True, "value": 2}
    assert runtime.state_fingerprint(first) != runtime.state_fingerprint(advanced)
    restored, observation = runtime.execute_tool(first, "increment", {"amount": 1})
    assert observation == {"success": True, "value": 1}
    assert advanced.environments[0].value == 2
    assert restored.environments[0].value == 1


def test_generated_action_parser_keeps_exact_tokens() -> None:
    action = parse_generated_action(
        '<tool_call>{"name":"increment","arguments":{"amount":2}}</tool_call>',
        [1, 2, 3],
        10,
        0.5,
    )
    assert action.kind == "tool"
    assert action.tool_name == "increment"
    assert action.arguments == {"amount": 2}
    assert action.token_ids == (1, 2, 3)


def test_collector_meets_rollout_budget_for_immediate_terminal_answers(tmp_path) -> None:
    class TerminalPolicy:
        def generate(self, messages, tools, **kwargs):
            return GeneratedAction(
                kind="content",
                raw_text="done",
                token_ids=(1,),
                prompt_tokens=2,
                generation_seconds=0.0,
            )

    collector = ExplicitBranchCollector(
        TerminalPolicy(), max_turns=2, branch_factor=2, rollout_width=8
    )
    record = collector.collect_task(
        _task(), benchmark_root=str(_benchmark(tmp_path)), seed=42, include_references=False
    )
    assert len(record["rollouts"]) == 8
    assert len({rollout["trajectory_id"] for rollout in record["rollouts"]}) == 8
    assert {
        rollout["turns"][0]["state_fingerprint_before"] for rollout in record["rollouts"]
    } == {record["initial_state_fingerprint"]}


def test_structured_judgment_parser_fails_closed() -> None:
    value = parse_judgment(
        '{"safe":true,"task_success":true,"refused":false,"correct_refusal":false,'
        '"confidence":0.9,"rationale":"ok"}<|im_end|>'
    )
    assert value["safe"] is True
    assert value["confidence"] == 0.9
