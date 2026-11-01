"""Mixed CE/SCST training with completed-epoch recovery and portable exports."""

from __future__ import annotations

import gc
import json
import math
import shutil
import time
from contextlib import contextmanager, nullcontext
from functools import wraps

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from .artifacts import read_jsonl, write_json
from .checkpoints import CheckpointManager
from .config import config_fingerprint, run_dir
from .data import AudioDataset, identity_collate, prepare_dataset
from .evaluation import evaluate_model
from .metrics import is_better
from .locking import RunLock
from .model import TranscriptPolicy, load_base, move_inputs, snapshot, supervised_batch
from .objective import scst_loss, sequence_log_probabilities
from .optimization import OptimizerController
from .pissa import PiSSAManager
from .semantic import PhoBERTReward


def exclusive_operation(method):
    @wraps(method)
    def invoke(self, *args, **kwargs):
        with self.exclusive_run():
            return method(self, *args, **kwargs)

    return invoke


class Trainer:
    def __init__(self, config, runtime, *, reward_factory=None):
        self.config, self.runtime = config, runtime
        self.root = run_dir(config)
        self.reward_factory = reward_factory or self._make_reward
        self.model = self.wrapped = self.processor = self.scorer = None
        self.method = self.optimizer = self.scheduler = self.scaler = None
        self.history, self.best = [], None
        self.progress = {
            "microbatches": 0,
            "attempted_updates": 0,
            "successful_updates": 0,
            "skipped_updates": 0,
        }
        self.gradient_groups = {"audio": False, "decoder": False}
        self._lease_depth = 0

    @contextmanager
    def exclusive_run(self):
        # Re-entry is only for the same Trainer/CLI operation, never a second job.
        if self._lease_depth:
            self._lease_depth += 1
            try:
                yield
            finally:
                self._lease_depth -= 1
            return
        lease = RunLock(self.root)
        try:
            self.runtime.call(
                lambda: lease.acquire() if self.runtime.main else None,
                "acquire exclusive run",
            )
            self._lease_depth = 1
            yield
        finally:
            self._lease_depth = 0
            lease.release()

    @staticmethod
    def _make_reward(config, runtime):
        return PhoBERTReward(
            config["semantic"], runtime.device, snapshot(config["semantic"], config)
        )

    def prepare(self):
        self.runtime.call(
            lambda: prepare_dataset(self.config, self.root)
            if self.runtime.main
            else None,
            "prepare manifests",
        )
        self.splits = {
            s: read_jsonl(self.root / "manifests" / f"{s}.jsonl")
            for s in ["train", "validation", "test"]
        }
        self.data_info = json.loads((self.root / "dataset.json").read_text())
        self.checkpoints = CheckpointManager(
            self.root / "checkpoints",
            config_fingerprint(self.config),
            self.data_info["fingerprint"],
        )

    def _load(self):
        self.model, self.processor = self.runtime.call(
            lambda: load_base(self.config, self.runtime), "load base"
        )
        self.scorer = self.runtime.call(
            lambda: self.reward_factory(self.config, self.runtime), "load frozen reward"
        )
        self.policy = TranscriptPolicy(
            self.processor, self.model, self.config["decode"]
        )

    def _evaluate(self, records, destination):
        return evaluate_model(
            self.model,
            self.processor,
            self.policy,
            self.scorer,
            records,
            self.config,
            self.runtime,
            destination,
        )

    def _setup_training(self, state, records):
        c, rt = self.config, self.runtime
        self.method = PiSSAManager(c["pissa"])
        self.model = rt.call(
            lambda: self.method.attach(
                self.model, state["method_state"] if state else None
            ),
            "PiSSA initialization",
        )
        adapter = self.model.peft_config["default"]
        adapter.base_model_name_or_path = c["model"]["model_id"]
        adapter.revision = c["model"]["revision"]
        rt.call(
            lambda: self.method.save_initialization(self.root / "pissa_initial")
            if rt.main
            else None,
            "save PiSSA initialization",
        )
        if not rt.main:
            self.method.initialization_path = self.root / "pissa_initial"
        self.model.config.use_cache = False
        if c["train"]["gradient_checkpointing"]:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.model.train()
        self.wrapped = (
            DistributedDataParallel(
                self.model,
                device_ids=[rt.local_rank] if rt.device.type == "cuda" else None,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
            if rt.world_size > 1
            else self.model
        )
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=c["train"]["learning_rate"],
            weight_decay=c["train"]["weight_decay"],
            betas=tuple(c["train"]["betas"]),
            eps=c["train"]["eps"],
        )
        self.sampler = DistributedSampler(
            AudioDataset(records, c["dataset"]["root"]),
            num_replicas=rt.world_size,
            rank=rt.rank,
            shuffle=True,
            seed=c["train"]["seed"],
            drop_last=False,
        )
        kwargs = (
            {"multiprocessing_context": "spawn"} if c["train"]["num_workers"] else {}
        )
        self.loader = DataLoader(
            self.sampler.dataset,
            batch_size=1,
            sampler=self.sampler,
            num_workers=c["train"]["num_workers"],
            collate_fn=identity_collate,
            pin_memory=rt.device.type == "cuda",
            **kwargs,
        )
        steps = (
            math.ceil(len(self.loader) / c["train"]["gradient_accumulation_steps"])
            * c["train"]["epochs"]
        )
        warmup = int(steps * c["train"]["warmup_ratio"])
        from transformers import get_linear_schedule_with_warmup

        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, warmup, steps)
        self.scaler = (
            torch.amp.GradScaler("cuda", init_scale=1024.0)
            if c["train"]["precision"] == "fp16"
            else None
        )
        self.controller = OptimizerController(
            self.optimizer,
            self.scheduler,
            self.scaler,
            c["train"]["gradient_clip_norm"],
            rt,
        )
        self.optimizer.zero_grad(set_to_none=True)
        if state:
            self.history, self.best, self.progress = (
                state["history"],
                state["best"],
                state["progress"],
            )
            self.checkpoints.restore(
                state,
                self.method,
                self.optimizer,
                self.scheduler,
                self.scaler,
                rank=rt.rank,
                world_size=rt.world_size,
            )

    def _autocast(self):
        return (
            torch.autocast("cuda", dtype=torch.float16)
            if self.config["train"]["precision"] == "fp16"
            else nullcontext()
        )

    def _no_sync(self, sync=False):
        return (
            self.wrapped.no_sync()
            if not sync and isinstance(self.wrapped, DistributedDataParallel)
            else nullcontext()
        )

    def _finite_loss(self, loss):
        if not self.runtime.agree(bool(torch.isfinite(loss.detach()).all())):
            raise FloatingPointError(
                "nonfinite training loss on a rank; checkpoint remains at last completed epoch"
            )

    def train_epoch(self, epoch):
        c, rt = self.config, self.runtime
        self.sampler.set_epoch(epoch - 1)
        self.model.train()
        totals = {
            "ce": 0.0,
            "scst": 0.0,
            "sample_reward": 0.0,
            "greedy_reward": 0.0,
            "nonzero_advantages": 0,
            "truncated_samples": 0,
            "chunked_rewards": 0,
            "examples": 0,
        }
        accumulation = c["train"]["gradient_accumulation_steps"]
        start = time.perf_counter()
        iterator = iter(self.loader)
        for group_start in range(0, len(self.loader), accumulation):
            group_size = min(accumulation, len(self.loader) - group_start)
            for micro in range(group_size):
                item = rt.call(lambda: next(iterator)[0], "read training audio")

                def prepare():
                    dtype = next(self.model.parameters()).dtype
                    ce = move_inputs(
                        supervised_batch(self.processor, item, c["decode"]["language"]),
                        rt.device,
                        dtype,
                    )
                    return ce, self.policy.prompt(item, rt.device, dtype)

                ce_batch, prompt = rt.call(prepare, "prepare assistant/policy inputs")

                def rollout_reward():
                    with self._autocast():
                        sampled, text = self.policy.generate(
                            self.model, prompt, sampled=True
                        )
                        _, greedy = self.policy.generate(
                            self.model, prompt, sampled=False
                        )
                    scores = self.scorer.score_pairs(
                        [
                            (item["record"]["reference"], text),
                            (item["record"]["reference"], greedy),
                        ]
                    )
                    if len(scores) != 2:
                        raise ValueError("incomplete semantic rewards")
                    return sampled, scores

                tokens, rewards = rt.call(
                    rollout_reward, "sample/greedy semantic rewards"
                )
                sampled_reward = torch.tensor([rewards[0].f1], device=rt.device)
                greedy_reward = torch.tensor([rewards[1].f1], device=rt.device)
                # Release the CE graph before replay to bound the T4 memory peak.
                with self._no_sync(), self._autocast():
                    output = self.wrapped(**ce_batch, use_cache=False)
                    ce = (
                        output.loss["loss"]
                        if isinstance(output.loss, dict)
                        else output.loss
                    )
                    self._finite_loss(ce)
                    self.controller.backward(ce, group_size)
                ce_value = float(ce.detach())
                del ce, output, ce_batch
                with self._no_sync(sync=micro == group_size - 1), self._autocast():
                    output = self.wrapped(**self.policy.replay(prompt, tokens))
                    logp, _ = sequence_log_probabilities(
                        output.logits,
                        prompt["input_ids"].shape[-1],
                        tokens,
                        self.policy.eos_ids,
                        self.policy.forbidden_ids,
                    )
                    pg = scst_loss(logp, sampled_reward, greedy_reward)
                    self._finite_loss(pg)
                    # Always backward, including zero advantage: identical DDP schedule.
                    self.controller.backward(c["scst"]["weight"] * pg, group_size)
                totals["ce"] += ce_value
                totals["scst"] += float(pg.detach())
                totals["sample_reward"] += rewards[0].f1
                totals["greedy_reward"] += rewards[1].f1
                totals["nonzero_advantages"] += int(rewards[0].f1 != rewards[1].f1)
                totals["chunked_rewards"] += sum(r.chunked for r in rewards)
                totals["truncated_samples"] += int(
                    not any(int(t) in self.policy.eos_ids for t in tokens[0].tolist())
                )
                totals["examples"] += 1
                self.progress["microbatches"] += 1
                del pg, logp, output, tokens
            if not all(self.gradient_groups.values()):
                for name, parameter in self.model.named_parameters():
                    if parameter.requires_grad and parameter.grad is not None:
                        group = "audio" if "audio" in name else "decoder"
                        self.gradient_groups[group] |= bool(
                            (
                                torch.isfinite(parameter.grad) & (parameter.grad != 0)
                            ).any()
                        )
            step = self.controller.step(self.model)
            self.progress["attempted_updates"] += 1
            self.progress["successful_updates"] += int(step["successful"])
            self.progress["skipped_updates"] += int(step["skipped"])
            if rt.main and (
                self.progress["attempted_updates"] % 10 == 0
                or group_start + group_size == len(self.loader)
            ):
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "microbatches_per_rank": group_start + group_size,
                            "updates": self.progress["successful_updates"],
                            "lr": self.optimizer.param_groups[0]["lr"],
                            **step,
                        }
                    ),
                    flush=True,
                )
        shards = rt.gather(totals)
        sums = {k: sum(s[k] for s in shards) for k in totals}
        count = sums.pop("examples")
        report = {k: v / count for k, v in sums.items()}
        report.update(
            examples_including_sampler_padding=count,
            seconds=time.perf_counter() - start,
            total_loss=report["ce"] + c["scst"]["weight"] * report["scst"],
            nonzero_gradient_groups=self.gradient_groups.copy(),
            cuda_peak_memory_bytes=torch.cuda.max_memory_allocated(rt.device)
            if rt.device.type == "cuda"
            else None,
        )
        return report

    def _recover_artifacts(self, state):
        """latest.pt is the durable commit; quarantine outputs from an interrupted next epoch."""
        completed = state["epoch"] if state else 0
        for epoch in range(1, completed + 1):
            path = self.root / "adapters" / f"epoch-{epoch:02d}"
            if not (path / "adapter_model.safetensors").is_file():
                raise ValueError(
                    "resume needs the complete run bundle, including committed epoch adapters"
                )
        for parent in ["adapters", "validation"]:
            for path in (self.root / parent).glob("epoch-*"):
                if int(path.name.split("-")[-1]) > completed:
                    orphan = (
                        self.root
                        / "orphaned"
                        / f"{time.time_ns()}-{parent}-{path.name}"
                    )
                    orphan.parent.mkdir(parents=True, exist_ok=True)
                    path.rename(orphan)
        if state:
            write_json(self.root / "history.json", state["history"])
            write_json(self.root / "best.json", state["best"])

    def save_epoch(self, epoch, validation, training):
        candidate = {
            "epoch": epoch,
            "adapter": f"adapters/epoch-{epoch:02d}",
            **validation,
        }
        if is_better(candidate, self.best):
            self.best = candidate
        self.history.append(
            {"epoch": epoch, "training": training, "validation": validation}
        )

        def export():
            if self.runtime.main:
                self.method.export(self.root / candidate["adapter"], self.processor)

        self.runtime.call(export, "export rank-16 adapter")
        rng = self.runtime.gather(self.checkpoints.capture_rng())

        def commit():
            if self.runtime.main:
                self.checkpoints.save_latest(
                    epoch=epoch,
                    method=self.method,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    history=self.history,
                    best=self.best,
                    rng_by_rank=rng,
                    progress=self.progress,
                )
                write_json(self.root / "history.json", self.history)
                write_json(self.root / "best.json", self.best)

        self.runtime.call(commit, "commit completed epoch")

    def _best_export(self, best):
        if not self.runtime.main:
            return
        destination = self.root / "best_adapter"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(self.root / best["adapter"], destination)
        write_json(
            self.root / "best_model.json",
            {
                **self.config["model"],
                "adapter_path": "best_adapter",
                "training_method": "pissa_converted_lora",
                "selected_epoch": best["epoch"],
                "selection": "validation.latest.normalized_bertscore_f1",
            },
        )

    @exclusive_operation
    def train(self, *, resume=False):
        self.prepare()
        latest = self.root / "checkpoints/latest.pt"
        if latest.exists() and not resume:
            raise FileExistsError(
                "run already has a checkpoint; use --resume or a new run_id"
            )
        if resume and not latest.is_file():
            raise FileNotFoundError("resume needs the full run bundle and latest.pt")
        state = self.checkpoints.read(latest) if resume else None
        if state and not 1 <= state["epoch"] <= self.config["train"]["epochs"]:
            raise ValueError("resume supports completed epoch boundaries only")
        self.runtime.call(
            lambda: self._recover_artifacts(state) if self.runtime.main else None,
            "recover epoch artifacts",
        )
        self._load()
        if not state:
            self._evaluate(self.splits["validation"], self.root / "validation/base")
        elif not (self.root / "validation/base/summary.json").is_file():
            raise ValueError("resume bundle is missing controlled base validation")
        self._setup_training(state, self.splits["train"])
        for epoch in range(
            (state["epoch"] if state else 0) + 1, self.config["train"]["epochs"] + 1
        ):
            training = self.train_epoch(epoch)
            validation = self._evaluate(
                self.splits["validation"],
                self.root / "validation" / f"epoch-{epoch:02d}",
            )
            self.save_epoch(epoch, validation, training)
        self.runtime.call(
            lambda: self._best_export(self.best), "publish selected adapter locally"
        )
        return self.best

    @exclusive_operation
    def evaluate_test(self):
        """Held-out data is accessed only after the final completed training epoch."""
        self.prepare()
        state = self.checkpoints.read(self.root / "checkpoints/latest.pt")
        if state["epoch"] != self.config["train"]["epochs"]:
            raise ValueError("finish all epochs before evaluating test")
        self._load()
        base = self._evaluate(self.splits["test"], self.root / "test/base")
        self.release(keep_scorer=True)
        self.model, self.processor = self.runtime.call(
            lambda: load_base(self.config, self.runtime), "reload original base"
        )
        from peft import PeftModel

        self.model = self.runtime.call(
            lambda: PeftModel.from_pretrained(
                self.model, self.root / state["best"]["adapter"], is_trainable=False
            ).eval(),
            "load best converted adapter",
        )
        self.policy = TranscriptPolicy(
            self.processor, self.model, self.config["decode"]
        )
        best = self._evaluate(self.splits["test"], self.root / "test/best")
        report = {
            "selected_epoch": state["best"]["epoch"],
            "base": base,
            "best": best,
            "config_fingerprint": config_fingerprint(self.config),
            "data_fingerprint": self.data_info["fingerprint"],
        }
        self.runtime.call(
            lambda: write_json(self.root / "test/comparison.json", report)
            if self.runtime.main
            else None,
            "write held-out comparison",
        )
        return report

    @exclusive_operation
    def smoke(self):
        """One partial accumulation group, then reload the portable adapter. No test data."""
        self.prepare()
        self._load()
        limit = min(len(self.splits["train"]), 2 * self.runtime.world_size)
        self._setup_training(None, self.splits["train"][:limit])
        report = self.train_epoch(1)
        if not self.progress["successful_updates"]:
            raise RuntimeError("smoke produced no successful optimizer update")
        if not self.runtime.agree(all(self.gradient_groups.values())):
            raise RuntimeError(
                "smoke requires nonzero audio encoder and decoder adapter gradients on every rank"
            )
        self.runtime.call(
            lambda: self.method.export(self.root / "smoke_adapter", self.processor)
            if self.runtime.main
            else None,
            "smoke portable export",
        )
        self.release(keep_scorer=True)
        self.model, self.processor = self.runtime.call(
            lambda: load_base(self.config, self.runtime), "smoke fresh base"
        )
        from peft import PeftModel

        self.model = self.runtime.call(
            lambda: PeftModel.from_pretrained(
                self.model, self.root / "smoke_adapter"
            ).eval(),
            "smoke adapter reload",
        )
        self.policy = TranscriptPolicy(
            self.processor, self.model, self.config["decode"]
        )
        selected = [r for r in self.splits["train"] if r["is_latest"]][
            : self.runtime.world_size
        ]
        report["reloaded"] = self._evaluate(selected, self.root / "smoke_reloaded")
        self.runtime.call(
            lambda: write_json(self.root / "smoke.json", report)
            if self.runtime.main
            else None,
            "write smoke report",
        )
        return report

    def release(self, *, keep_scorer=False):
        self.wrapped = self.model = self.processor = self.method = None
        self.optimizer = self.scheduler = self.scaler = None
        if hasattr(self, "controller"):
            self.controller = None
        if hasattr(self, "policy"):
            self.policy = None
        if not keep_scorer:
            self.scorer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
