"""LoRA training for controlled HFDTree credit variants with cost telemetry."""

from __future__ import annotations

from dataclasses import dataclass
import json
import random
import time
from pathlib import Path
from typing import Mapping, Sequence


CREDIT_FIELDS = {
    "full": "credit_full",
    "no_fallback": "credit_full",
    "propagation_without_centering": "credit_propagation_without_centering",
    "two_sided_local_only": "credit_two_sided_local_only",
    "two_sided_local_with_fallback": "credit_two_sided_local_only",
    "one_sided_tree": "credit_one_sided_tree",
    "outcome_tree_equal_information": "credit_outcome_tree",
    "propagation_without_centering_matched_fallback": "credit_propagation_without_centering",
    "two_sided_local_matched_fallback": "credit_two_sided_local_only",
    "one_sided_tree_matched_fallback": "credit_one_sided_tree",
    "outcome_tree_matched_fallback": "credit_outcome_tree",
    "direct_bireference_matched_fallback": "credit_full",
}

MATCHED_FALLBACK_VARIANTS = {
    "full",
    "propagation_without_centering_matched_fallback",
    "two_sided_local_matched_fallback",
    "one_sided_tree_matched_fallback",
    "outcome_tree_matched_fallback",
    "direct_bireference_matched_fallback",
}


@dataclass(frozen=True, slots=True)
class PreparedExample:
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    weight: float
    fallback: bool


def prepare_examples(
    rows: Sequence[Mapping[str, object]],
    tokenizer: object,
    *,
    variant: str,
    delta: float = 0.05,
    epsilon: float = 0.05,
    max_length: int = 4096,
) -> list[PreparedExample]:
    if variant not in CREDIT_FIELDS:
        raise ValueError(f"unknown training variant: {variant}")
    credit_field = CREDIT_FIELDS[variant]
    allow_fallback = variant in MATCHED_FALLBACK_VARIANTS | {"two_sided_local_with_fallback"}
    output = []
    for row in rows:
        credit = float(row[credit_field])
        normalized = float(row["normalized_evidence"])
        # Matched-fallback controls use the exact fallback decision made by the
        # full objective.  This prevents a change in credit field from also
        # changing which reference actions become SFT targets.
        fallback_credit = (
            float(row["credit_full"])
            if variant in MATCHED_FALLBACK_VARIANTS
            else credit
        )
        fallback = bool(
            allow_fallback
            and row.get("reference_valid")
            and abs(normalized) <= delta
            and abs(fallback_credit) <= epsilon
            and row.get("reference_action_token_ids")
        )
        target = (
            list(row["reference_action_token_ids"])
            if fallback
            else list(row["action_token_ids"])
        )
        weight = 1.0 if fallback else max(-1.0, min(1.0, credit))
        if not target or (not fallback and abs(weight) < 1e-8):
            continue
        kwargs = {
            "tools": row["tools"],
            "add_generation_prompt": True,
            "tokenize": True,
        }
        if "qwen3" in str(getattr(tokenizer, "name_or_path", "")).lower():
            kwargs["enable_thinking"] = False
        prompt = tokenizer.apply_chat_template(row["prefix_messages"], **kwargs)
        if isinstance(prompt, Mapping):
            prompt = prompt["input_ids"]
        if hasattr(prompt, "tolist"):
            prompt = prompt.tolist()
        if prompt and isinstance(prompt[0], list):
            prompt = prompt[0]
        available_prompt = max_length - len(target)
        if available_prompt <= 0:
            target = target[-(max_length - 1) :]
            available_prompt = 1
        prompt = list(prompt)[-available_prompt:]
        input_ids = prompt + target
        labels = [-100] * len(prompt) + target
        output.append(
            PreparedExample(
                input_ids=tuple(int(value) for value in input_ids),
                labels=tuple(int(value) for value in labels),
                weight=weight,
                fallback=fallback,
            )
        )
    if not output:
        raise ValueError("no non-zero training examples were prepared")
    return output


def train_lora(
    rows: Sequence[Mapping[str, object]],
    *,
    model_path: str,
    output_dir: str | Path,
    variant: str,
    seed: int = 42,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    max_steps: int = 1000,
    max_length: int = 4096,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    delta: float = 0.05,
    epsilon: float = 0.05,
    resume: bool = False,
) -> dict[str, object]:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    import peft.import_utils as peft_import_utils
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(seed)
    # The shared environment contains torchao 0.9 for an unrelated workload.
    # This model is ordinary BF16 (not TorchAO-quantized), so prevent PEFT's
    # optional dispatcher from rejecting an irrelevant old torchao package.
    peft_import_utils.is_torchao_available = lambda: False
    try:
        import peft.tuners.lora.torchao as peft_torchao
    except ImportError:
        peft_torchao = None
    if peft_torchao is not None:
        peft_torchao.is_torchao_available = lambda: False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    examples = prepare_examples(
        rows,
        tokenizer,
        variant=variant,
        delta=delta,
        epsilon=epsilon,
        max_length=max_length,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    output_path = Path(output_dir)
    state_path = output_path / "training_state.pt"
    if resume and (output_path / "adapter_config.json").exists():
        model = PeftModel.from_pretrained(model, output_path, is_trainable=True)
    else:
        lora = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    previous = {}
    if resume and state_path.exists():
        previous = torch.load(state_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(previous["optimizer"])
    completed_steps = int(previous.get("optimizer_steps", 0))
    if completed_steps >= max_steps:
        metrics_path = output_path / "training_metrics.json"
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    generator = random.Random(seed + completed_steps)
    order = list(range(len(examples)))
    output_path.mkdir(parents=True, exist_ok=True)
    losses = []
    fallback_count = sum(example.fallback for example in examples)
    segment_tokens = 0
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    energy_start = _gpu_energy_mj()
    micro_step = 0
    optimizer_step = completed_steps
    while optimizer_step < max_steps:
        generator.shuffle(order)
        for offset in range(0, len(order), batch_size):
            batch = [examples[index] for index in order[offset : offset + batch_size]]
            input_ids, attention_mask, labels, weights = _collate(batch, tokenizer.pad_token_id)
            input_ids = input_ids.cuda(non_blocking=True)
            attention_mask = attention_mask.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            weights = weights.cuda(non_blocking=True)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            mean_logprobs = _mean_logprobs(logits, labels)
            loss = -(weights * mean_logprobs).mean() / gradient_accumulation_steps
            loss.backward()
            losses.append(float(loss.detach().cpu()) * gradient_accumulation_steps)
            segment_tokens += int(attention_mask.sum().item())
            micro_step += 1
            if micro_step % gradient_accumulation_steps != 0:
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            if optimizer_step % 10 == 0 or optimizer_step == 1:
                print(
                    f"step={optimizer_step}/{max_steps} loss={sum(losses[-10:]) / min(10, len(losses)):.6f}",
                    flush=True,
                )
            if optimizer_step >= max_steps:
                break
    torch.cuda.synchronize()
    wall_seconds = time.perf_counter() - start
    energy_end = _gpu_energy_mj()
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    segment_energy_wh = (
        (energy_end - energy_start) / 3_600_000.0
        if energy_start is not None and energy_end is not None and energy_end >= energy_start
        else None
    )
    cumulative_tokens = int(previous.get("tokens_processed", 0)) + segment_tokens
    cumulative_wall = float(previous.get("wall_seconds", 0.0)) + wall_seconds
    previous_energy = previous.get("gpu_energy_wh")
    cumulative_energy = (
        float(previous_energy or 0.0) + segment_energy_wh
        if segment_energy_wh is not None
        else previous_energy
    )
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "optimizer_steps": optimizer_step,
            "tokens_processed": cumulative_tokens,
            "wall_seconds": cumulative_wall,
            "gpu_energy_wh": cumulative_energy,
        },
        state_path,
    )
    metrics = {
        "variant": variant,
        "model_path": model_path,
        "seed": seed,
        "examples": len(examples),
        "fallback_examples": fallback_count,
        "optimizer_steps": optimizer_step,
        "micro_steps_this_segment": micro_step,
        "tokens_processed": cumulative_tokens,
        "tokens_processed_this_segment": segment_tokens,
        "wall_seconds": cumulative_wall,
        "wall_seconds_this_segment": wall_seconds,
        "gpu_hours": cumulative_wall / 3600.0,
        "gpu_energy_wh": cumulative_energy,
        "gpu_energy_wh_this_segment": segment_energy_wh,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "mean_recent_loss": sum(losses[-100:]) / min(100, len(losses)),
        "hyperparameters": {
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "max_steps": max_steps,
            "max_length": max_length,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "delta": delta,
            "epsilon": epsilon,
            "resume": resume,
        },
    }
    (output_path / "training_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    return metrics


def _collate(examples: Sequence[PreparedExample], pad_token_id: int):
    import torch

    length = max(len(example.input_ids) for example in examples)
    inputs = []
    labels = []
    masks = []
    weights = []
    for example in examples:
        padding = length - len(example.input_ids)
        inputs.append(list(example.input_ids) + [pad_token_id] * padding)
        labels.append(list(example.labels) + [-100] * padding)
        masks.append([1] * len(example.input_ids) + [0] * padding)
        weights.append(example.weight)
    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(masks, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(weights, dtype=torch.float32),
    )


def _mean_logprobs(logits, labels):
    import torch

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    mask = shifted_labels.ne(-100)
    safe = shifted_labels.masked_fill(~mask, 0)
    values = torch.log_softmax(shifted_logits.float(), dim=-1).gather(
        -1, safe.unsqueeze(-1)
    ).squeeze(-1)
    counts = mask.sum(-1).clamp_min(1)
    return (values * mask).sum(-1) / counts


def _gpu_energy_mj() -> int | None:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return int(pynvml.nvmlDeviceGetTotalEnergyConsumption(handle))
    except Exception:
        return None
