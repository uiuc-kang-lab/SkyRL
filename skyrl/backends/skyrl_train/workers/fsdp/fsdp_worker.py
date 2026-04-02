from skyrl.train.utils.trainer_utils import get_rope_scaling_config, get_rope_theta_config
import ray
import torch
from loguru import logger as lora_logger
import torch.distributed
import torch.optim as optim
from transformers.trainer import get_scheduler
from transformers import AutoConfig
from torch.distributed.fsdp.api import ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
import io
from typing import TYPE_CHECKING

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from skyrl.backends.skyrl_train.workers.model_wrapper import HFModelWrapper, get_llm_for_sequence_regression
from skyrl.backends.skyrl_train.distributed.fsdp_strategy import FSDPStrategy
from skyrl.train.utils.utils import str_to_torch_dtype
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch
from skyrl.backends.skyrl_train.distributed.fsdp_utils import fsdp_version, get_init_weight_context_manager
from skyrl.backends.skyrl_train.workers.worker import (
    PolicyWorkerBase,
    CriticWorkerBase,
    RefWorkerBase,
)
from skyrl.backends.skyrl_train.weight_sync import WeightExtractor, WeightChunk, LoraLoadRequest
from skyrl.backends.skyrl_train.weight_sync.weight_extractor_utils import yield_module_grouped_chunks

if TYPE_CHECKING:
    from skyrl.train.config.config import InferenceEngineConfig


class FSDPWeightExtractor(WeightExtractor):
    """Extracts weights from FSDP-sharded models.

    Args:
        model: FSDP model to extract weights from
        group_by_module: If True, group parameters by module (e.g., for FlashRL QKV fusion)
        batch_size_threshold_gb: If > 0, batch complete modules together until threshold is reached
    """

    def __init__(self, model: torch.nn.Module, group_by_module: bool = False, batch_size_threshold_gb: float = 0.0):
        self.model = model
        self.group_by_module = group_by_module
        self.batch_size_threshold_gb = batch_size_threshold_gb

    def extract_weights(self, dtype: torch.dtype):
        """Extract weights from FSDP model.

        Args:
            dtype: Target dtype for inference

        Yields:
            WeightChunk objects (one per parameter, or grouped by module)
        """
        # Configure state_dict type for FSDP v1
        if fsdp_version(self.model) == 1:
            FSDP.set_state_dict_type(
                self.model,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        # Get state dict (handles FSDP sharding)
        params = self.model.state_dict()

        if not self.group_by_module:
            # Simple path: yield one chunk per parameter
            for name, param in params.items():
                tensor = self._gather_tensor(param).to(dtype).detach().contiguous()
                yield WeightChunk(
                    names=[name],
                    dtypes=[str(dtype)],
                    shapes=[list(tensor.shape)],
                    tensors=[tensor],
                )
        else:
            for chunk in yield_module_grouped_chunks(
                params=params,
                dtype=dtype,
                gather_tensor_fn=self._gather_tensor,
                get_shape_fn=lambda name, param, tensor: list(tensor.shape),
                batch_size_threshold_gb=self.batch_size_threshold_gb,
            ):
                yield chunk

    def get_weight_metadata(self, dtype: torch.dtype) -> dict:
        """Return weight metadata without materializing full tensors.

        Uses state_dict() to get clean parameter names (FSDP strips the
        _fsdp_wrapped_module prefix), matching extract_weights behavior.
        The sharded tensors returned by state_dict() are not gathered;
        we only read their shape.
        """
        if fsdp_version(self.model) == 1:
            FSDP.set_state_dict_type(
                self.model,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        names = []
        dtype_names = []
        shapes = []
        dtype_name = str(dtype).split(".")[-1]
        for name, param in self.model.state_dict().items():
            names.append(name)
            dtype_names.append(dtype_name)
            shapes.append(list(param.shape))
        return {"names": names, "dtype_names": dtype_names, "shapes": shapes}

    def _gather_tensor(self, param: torch.Tensor) -> torch.Tensor:
        """Gather sharded tensor into full tensor."""
        device = torch.cuda.current_device()
        return param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param


class FSDPPolicyWorkerBase(PolicyWorkerBase):
    def offload_to_cpu(self, pin_memory=True, non_blocking=True, offload_optimizer=True, offload_model=True):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(
            self.model, self.optimizer, pin_memory, non_blocking, offload_optimizer, offload_model
        )

    def backload_to_gpu(self, non_blocking=True, backload_optimizer=True, backload_model=True):
        self.strategy.backload_to_gpu(self.model, self.optimizer, non_blocking, backload_optimizer, backload_model)

    def init_model(self, model_path, num_training_steps: int = None):
        assert self.cfg.strategy in ("fsdp", "fsdp2")
        lora_cfg = self.cfg.policy.model.lora

        strategy = FSDPStrategy(
            fsdp_config=self.cfg.policy.fsdp_config,
            optimizer_config=self.cfg.policy.optimizer_config,
            model_config=self.cfg.policy.model,
            fsdp_strategy=self.cfg.strategy,
            seed=self.cfg.seed,
            micro_train_batch_size_per_gpu=self.cfg.micro_train_batch_size_per_gpu,
            num_training_steps=num_training_steps,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        self._is_lora = lora_cfg.rank > 0
        custom_lora_cfg = self.cfg.policy.model.custom_lora
        self._is_custom_lora = custom_lora_cfg.enabled

        if lora_cfg.load_in_4bit:
            # QLoRA path: BitsAndBytes NF4 quantization is incompatible with both
            # meta-tensor init (BnB needs real weights to quantize) and FSDP sharding
            # (FSDP cannot shard NF4 quantized tensors). Load directly to GPU on each
            # rank and skip FSDP wrapping.
            wrapped_model = HFModelWrapper(
                model_path,
                use_flash_attention_2=self.cfg.flash_attn,
                bf16=True,  # BnB requires bf16 compute dtype
                load_in_4bit=True,
                lora_rank=lora_cfg.rank,
                lora_alpha=lora_cfg.alpha,
                lora_dropout=lora_cfg.dropout,
                lora_init_method=lora_cfg.init_method,
                target_modules=lora_cfg.target_modules,
                exclude_modules=lora_cfg.exclude_modules,
                layers_to_transform=lora_cfg.layers_to_transform,
                device_map="auto",
                sequence_parallel_size=self.cfg.policy.sequence_parallel_size,
                use_sample_packing=self.cfg.use_sample_packing,
                use_torch_compile=self.cfg.policy.use_torch_compile,
                rope_scaling=get_rope_scaling_config(self.cfg),
                rope_theta=get_rope_theta_config(self.cfg),
                model_config_kwargs=self.cfg.policy.model_config_kwargs,
                custom_lora_config=custom_lora_cfg if self._is_custom_lora else None,
            )
            self._seq_parallel_monkey_patch(model=wrapped_model.model)

            if self.cfg.gradient_checkpointing:
                wrapped_model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": self.cfg.gradient_checkpointing_use_reentrant}
                )

            # Create optimizer and scheduler directly (mirrors FSDPStrategy._fsdp_init_train_model)
            optim_config = self.cfg.policy.optimizer_config
            optimizer = optim.AdamW(
                wrapped_model.parameters(),
                lr=optim_config.lr,
                betas=tuple(optim_config.adam_betas),
                weight_decay=optim_config.weight_decay,
            )
            scheduler = get_scheduler(
                optim_config.scheduler,
                optimizer,
                num_warmup_steps=optim_config.num_warmup_steps,
                num_training_steps=num_training_steps,
            )
            self.model = wrapped_model
            self.optimizer = optimizer
            self.scheduler = scheduler
        else:
            # Standard FSDP path
            model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            init_context = get_init_weight_context_manager(
                use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.strategy.device_mesh
            )
            with init_context():
                wrapped_model = HFModelWrapper(
                    model_path,
                    use_flash_attention_2=self.cfg.flash_attn,
                    # NOTE (sumanthrh): Model initialization should always be in fp32
                    # during training
                    bf16=False,
                    lora_rank=lora_cfg.rank,
                    lora_alpha=lora_cfg.alpha,
                    lora_dropout=lora_cfg.dropout,
                    lora_init_method=lora_cfg.init_method,
                    target_modules=lora_cfg.target_modules,
                    exclude_modules=lora_cfg.exclude_modules,
                    layers_to_transform=lora_cfg.layers_to_transform,
                    sequence_parallel_size=self.cfg.policy.sequence_parallel_size,
                    use_sample_packing=self.cfg.use_sample_packing,
                    use_torch_compile=self.cfg.policy.use_torch_compile,
                    rope_scaling=get_rope_scaling_config(self.cfg),
                    rope_theta=get_rope_theta_config(self.cfg),
                    model_config_kwargs=self.cfg.policy.model_config_kwargs,
                    custom_lora_config=custom_lora_cfg if self._is_custom_lora else None,
                )
                # in-place patch
                self._seq_parallel_monkey_patch(model=wrapped_model.model)

                if self.cfg.gradient_checkpointing:
                    wrapped_model.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": self.cfg.gradient_checkpointing_use_reentrant}
                    )

            self.model, self.optimizer, self.scheduler = strategy.prepare(
                (wrapped_model, None, None),
            )

        assert (
            self.optimizer is not None and self.scheduler is not None
        ), "Model preparation should create optimizer and scheduler"

    async def init_weight_sync_state(self, inference_engine_client, inference_engine_cfg: "InferenceEngineConfig"):
        # Call super first to set _transfer_strategy_cls and create sender/receivers
        await super().init_weight_sync_state(inference_engine_client, inference_engine_cfg)

        # Initialize weight extractor
        # TODO(haochen): Now module grouping (in order to support FlashRL) is only enabled for the CUDA IPC
        # transfer strategy, we can enable it for other strategies as well.
        from skyrl.backends.skyrl_train.weight_sync import CudaIpcTransferStrategy

        group_by_module = self._transfer_strategy_cls is CudaIpcTransferStrategy
        self.weight_extractor = FSDPWeightExtractor(
            self.model.model,
            group_by_module=group_by_module,
            batch_size_threshold_gb=(
                inference_engine_cfg.weight_transfer_threshold_cuda_ipc_GB if group_by_module else 0.0
            ),
        )

        # For custom LoRA: build a map from state_dict weight key → layer info
        # (scheme name + keys for v and buffers) so we can apply deltas
        # inline during weight extraction using gathered tensors only.
        if self._is_custom_lora:
            from skyrl.backends.skyrl_train.custom_lora import build_custom_lora_delta_map

            self._custom_lora_delta_map = build_custom_lora_delta_map(self.model.model)

    async def _save_lora_adapters_and_sync(self, peft_model, lora_sync_path, inference_engine_client, needs_lm_prefix=False):
        """Collect LoRA parameters, save and call inference engine to load.

        Args:
            needs_lm_prefix: When True, remap PEFT adapter keys so that vLLM can
                resolve them correctly for models whose vLLM class is
                ForConditionalGeneration (e.g. Qwen3.5 with language_model_only=True).
                See ``remap_peft_lora_keys_for_conditional_gen`` for details.
        """
        import os
        import json
        from dataclasses import asdict
        from safetensors.torch import save_file
        from skyrl.backends.skyrl_train.distributed.fsdp_utils import (
            collect_lora_params,
            remap_peft_lora_keys_for_conditional_gen,
        )

        lora_params = collect_lora_params(module=self.model.model)

        if needs_lm_prefix:
            lora_params = remap_peft_lora_keys_for_conditional_gen(lora_params)

        if torch.distributed.get_rank() == 0:
            os.makedirs(lora_sync_path, exist_ok=True)

            peft_config = asdict(peft_model.peft_config.get("default", {}))
            peft_config["task_type"] = peft_config["task_type"].value
            peft_config["peft_type"] = peft_config["peft_type"].value
            peft_config["target_modules"] = list(peft_config["target_modules"])
            if isinstance(peft_config.get("exclude_modules"), set):
                peft_config["exclude_modules"] = list(peft_config["exclude_modules"])

            # --- LoRA weight sync diagnostics ---
            lora_keys = list(lora_params.keys())
            print(f"[LoRA sync] Saving {len(lora_keys)} adapter tensors to {lora_sync_path}")
            print(f"[LoRA sync] Sample keys (first 4): {lora_keys[:4]}")
            # Log L2 norms for a few lora_B weights to verify they change across steps
            lora_b_keys = [k for k in lora_keys if "lora_B" in k][:4]
            for k in lora_b_keys:
                norm = lora_params[k].float().norm().item()
                print(f"[LoRA sync] {k} | shape={tuple(lora_params[k].shape)} | L2={norm:.6f}")

            # Save LoRA parameters and config
            save_file(lora_params, os.path.join(lora_sync_path, "adapter_model.safetensors"))
            with io.open(os.path.join(lora_sync_path, "adapter_config.json"), "w", encoding="utf-8") as f:
                json.dump(peft_config, f, ensure_ascii=False, indent=4)

            # Send LoRA disk loading request to inference engine
            lora_request = LoraLoadRequest(lora_path=lora_sync_path)
            print(f"[LoRA sync] Sending add_lora request to inference engine (path={lora_sync_path})")
            result = await inference_engine_client.update_named_weights(lora_request)
            print(f"[LoRA sync] add_lora result: {result}")

        torch.distributed.barrier()

    async def broadcast_to_inference_engines(self, inference_engine_client, inference_engine_cfg):
        use_prefix_cache = inference_engine_cfg.enable_prefix_caching
        generator_dtype = str_to_torch_dtype(inference_engine_cfg.model_dtype)
        cache_reset_task = None
        if use_prefix_cache and torch.distributed.get_rank() == 0:
            # clear prefix cache
            cache_reset_task = inference_engine_client.reset_prefix_cache()

        torch.cuda.empty_cache()

        # Check if this is a LoRA model
        peft_model = getattr(self.model.model, "_fsdp_wrapped_module", self.model.model)

        if self._is_lora:
            assert hasattr(peft_model, "peft_config"), "LoRA model should have peft_config"

            # assume base model is already synced, sync LoRA adapters
            lora_sync_path = self.cfg.policy.model.lora.lora_sync_path
            # When language_model_only=True (e.g. Qwen3.5), vLLM uses
            # ForConditionalGeneration whose LoRA layers are registered under
            # "language_model.model.*", but PEFT on the CausalLM training model
            # produces keys with "model.*". Remap before saving to disk.
            needs_lm_prefix = inference_engine_cfg.engine_init_kwargs.get("language_model_only", False)
            await self._save_lora_adapters_and_sync(
                peft_model, lora_sync_path, inference_engine_client, needs_lm_prefix=needs_lm_prefix
            )
            # Await prefix cache reset here — the non-LoRA path does this at the
            # end of the function, but we return early above, so we must handle it
            # explicitly. Without this, stale KV cache entries (keyed by old LoRA
            # IDs) accumulate across steps when enable_prefix_caching=True.
            if cache_reset_task is not None:
                await cache_reset_task
            return

        if self._is_custom_lora:
            # Custom LoRA weight sync — FSDP-safe, memory-efficient.
            #
            # Under FSDP, module attributes (self.v, self.U_scaled, …) are
            # flat shards outside forward/summon_full_params.  We must NOT
            # read them.  Instead we use gathered tensors from state_dict().
            #
            # Single state_dict() call, two-phase iteration:
            #   Phase 1: gather only tiny custom LoRA tensors (v, buffers)
            #            into a lookup dict.  Base weights are NOT gathered.
            #   Phase 2: stream base-model entries one at a time, applying
            #            deltas inline for adapted weights, skipping
            #            custom LoRA internals.
            from skyrl.backends.skyrl_train.custom_lora.schemes import get_scheme

            delta_map = self._custom_lora_delta_map
            skip_keys: set[str] = set()
            for info in delta_map.values():
                skip_keys.add(info.v_key)
                skip_keys.update(info.buffer_keys.values())

            # Single state_dict() call — returns sharded/DTensor references
            if fsdp_version(self.model.model) == 1:
                FSDP.set_state_dict_type(
                    self.model.model,
                    state_dict_type=StateDictType.SHARDED_STATE_DICT,
                    state_dict_config=ShardedStateDictConfig(),
                )
            sd = self.model.model.state_dict()

            # Phase 1: gather only the tiny custom LoRA tensors (v, U_scaled,
            # V, P).  These are a few MB total — negligible.
            gathered: dict[str, torch.Tensor] = {}
            for name, param in sd.items():
                if name in skip_keys:
                    gathered[name] = (
                        self.weight_extractor._gather_tensor(param)
                        .to(generator_dtype).detach().contiguous()
                    )

            # Phase 2: stream base-model weights, applying deltas inline.
            dtype_str = str(generator_dtype)

            def _stream_with_deltas():
                for name, param in sd.items():
                    if name in skip_keys:
                        continue
                    tensor = (
                        self.weight_extractor._gather_tensor(param)
                        .to(generator_dtype).detach().contiguous()
                    )
                    if name in delta_map:
                        info = delta_map[name]
                        scheme = get_scheme(info.scheme_name)
                        v_full = gathered[info.v_key]
                        buffers = {
                            buf_name: gathered[sd_key]
                            for buf_name, sd_key in info.buffer_keys.items()
                        }
                        scheme.merge_into(tensor, v_full, buffers)
                    yield WeightChunk(
                        names=[name],
                        dtypes=[dtype_str],
                        shapes=[list(tensor.shape)],
                        tensors=[tensor],
                    )

            # Build metadata that excludes custom LoRA internal keys
            filtered_meta = {"names": [], "dtype_names": [], "shapes": []}
            for name, param in sd.items():
                if name not in skip_keys:
                    filtered_meta["names"].append(name)
                    filtered_meta["dtype_names"].append(dtype_str.split(".")[-1])
                    filtered_meta["shapes"].append(list(param.shape))

            await self._weight_transfer_sender.send_chunks(
                _stream_with_deltas(),
                weight_metadata=filtered_meta,
            )

            if cache_reset_task is not None:
                await cache_reset_task
            torch.cuda.empty_cache()
            torch.distributed.barrier()
            return

        # Extract and send weights using the sender created at init time
        weight_iterator = self.weight_extractor.extract_weights(generator_dtype)
        weight_metadata = self.weight_extractor.get_weight_metadata(generator_dtype)
        await self._weight_transfer_sender.send_chunks(
            weight_iterator,
            weight_metadata=weight_metadata,
        )

        if cache_reset_task is not None:
            await cache_reset_task
        torch.cuda.empty_cache()
        torch.distributed.barrier()

    def get_weight_statistics(self):
        """Compute lightweight statistics for model weights"""
        raise NotImplementedError()

    def _set_pad_token_id(self, pad_token_id):
        # NOTE (sumanthrh): self.model -> HFModelWrapper; self.model.model -> AutoModelForCausalLM
        self.model.model.config.pad_token_id = pad_token_id

    def forward(
        self,
        data: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """Run forward pass on data in inference mode.

        Reshard the model after forward pass to redistribute memory and allow for offloading to cpu.
        """
        output = super().forward(data)
        # unshard the root FSDP module (https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes)
        if self._world_size > 1 and fsdp_version(self.model.model) == 1:
            self.model.model._handle.reshard(True)
        return output


class FSDPCriticWorkerBase(CriticWorkerBase):
    def offload_to_cpu(self, pin_memory=True, non_blocking=True, offload_optimizer=True, offload_model=True):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(
            self.model, self.optimizer, pin_memory, non_blocking, offload_optimizer, offload_model
        )

    def backload_to_gpu(self, non_blocking=True, backload_optimizer=True, backload_model=True):
        self.strategy.backload_to_gpu(self.model, self.optimizer, non_blocking, backload_optimizer, backload_model)

    def init_model(self, model_path, num_training_steps: int = None):
        assert self.cfg.strategy in ("fsdp", "fsdp2")
        strategy = FSDPStrategy(
            fsdp_config=self.cfg.critic.fsdp_config,
            optimizer_config=self.cfg.critic.optimizer_config,
            fsdp_strategy=self.cfg.strategy,
            seed=self.cfg.seed,
            micro_train_batch_size_per_gpu=self.cfg.micro_train_batch_size_per_gpu,
            num_training_steps=num_training_steps,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.strategy.device_mesh
        )
        with init_context():
            critic = get_llm_for_sequence_regression(
                model_path,
                "critic",
                use_flash_attention_2=self.cfg.flash_attn,
                # NOTE (sumanthrh): Model initialization should always be in fp32
                # during training
                bf16=False,
                lora_rank=self.cfg.critic.model.lora.rank,
                lora_alpha=self.cfg.critic.model.lora.alpha,
                lora_dropout=self.cfg.critic.model.lora.dropout,
                target_modules=self.cfg.critic.model.lora.target_modules,
                exclude_modules=self.cfg.critic.model.lora.exclude_modules,
                value_head_prefix=self.cfg.algorithm.value_head_prefix,
                init_value_head=self.cfg.policy.model.path == self.cfg.critic.model.path,
                sequence_parallel_size=self.cfg.critic.sequence_parallel_size,
                use_sample_packing=self.cfg.use_sample_packing,
                model_config_kwargs=self.cfg.critic.model_config_kwargs,
            )
            self._seq_parallel_monkey_patch(model=critic, use_parent_class=True)

            if self.cfg.gradient_checkpointing:
                critic.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": self.cfg.gradient_checkpointing_use_reentrant}
                )

        # prepare models/optimizers...
        self.model, self.optimizer, self.scheduler = strategy.prepare(
            (critic, None, None),
        )
        assert self.optimizer is not None

    def forward(
        self,
        data: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """Run forward pass on data in inference mode.

        Reshard the model after forward pass to redistribute memory and allow for offloading to cpu.
        """
        output = super().forward(data)
        # unshard the root FSDP module (https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes)
        if self._world_size > 1 and fsdp_version(self.model.model) == 1:
            self.model.model._handle.reshard(True)
        return output


class FSDPRefWorkerBase(RefWorkerBase):
    def offload_to_cpu(self, pin_memory=True, non_blocking=True, **kwargs):
        self._set_numa_affinity(torch.distributed.get_rank() % torch.cuda.device_count())
        self.strategy.offload_to_cpu(self.model, None, pin_memory, non_blocking)

    def backload_to_gpu(self, non_blocking=True, **kwargs):
        self.strategy.backload_to_gpu(self.model, None, non_blocking)

    def init_model(self, model_path):
        assert self.cfg.strategy in ("fsdp", "fsdp2")
        strategy = FSDPStrategy(
            fsdp_config=self.cfg.ref.fsdp_config,
            fsdp_strategy=self.cfg.strategy,
            seed=self.cfg.seed,
            micro_train_batch_size_per_gpu=self.cfg.micro_train_batch_size_per_gpu,
        )
        strategy.setup_distributed()
        self.strategy = strategy

        model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not model_config.tie_word_embeddings, mesh=self.strategy.device_mesh
        )

        with init_context():
            wrapped_model = HFModelWrapper(
                model_path,
                use_flash_attention_2=self.cfg.flash_attn,
                bf16=self.cfg.bf16,
                sequence_parallel_size=self.cfg.ref.sequence_parallel_size,
                use_sample_packing=self.cfg.use_sample_packing,
                rope_scaling=get_rope_scaling_config(self.cfg),
                rope_theta=get_rope_theta_config(self.cfg),
                model_config_kwargs=self.cfg.ref.model_config_kwargs,
            )
            self._seq_parallel_monkey_patch(model=wrapped_model.model)

        self.model = strategy.prepare(wrapped_model)
        self.model.eval()

    def forward(
        self,
        data: TrainingInputBatch,
    ) -> TrainingOutputBatch:
        """Run forward pass on data in inference mode.

        Reshard the model after forward pass to redistribute memory and allow for offloading to cpu.
        """
        output = super().forward(data)
        # unshard the root FSDP module (https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes)
        if self._world_size > 1 and fsdp_version(self.model.model) == 1:
            self.model.model._handle.reshard(True)
        return output


# Ray remote actors
PolicyWorker = ray.remote(num_gpus=1)(FSDPPolicyWorkerBase)
CriticWorker = ray.remote(num_gpus=1)(FSDPCriticWorkerBase)
RefWorker = ray.remote(num_gpus=1)(FSDPRefWorkerBase)
