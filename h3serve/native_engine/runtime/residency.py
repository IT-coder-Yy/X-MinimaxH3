"""Component lifecycle and a conservative single-GPU residency budget."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Iterable, Protocol

from .config import RuntimeConfig
from .pinned_pool import pack_pinned_tensors


class ResidencyBudgetError(RuntimeError):
    """Raised before a transition that exceeds the configured GPU budget."""


class ComponentResidency(Protocol):
    """Movement adapter for one lazily or eagerly loaded model component."""

    name: str
    estimated_device_bytes: int

    @property
    def value(self) -> Any:
        """Return the callable component wrapped by this adapter."""

    def move_to(self, device: str, *, non_blocking: bool) -> None:
        """Move component storage to the requested device."""


@dataclass(slots=True)
class TorchModuleResidency:
    """Minimal adapter for objects exposing PyTorch's ``.to(device)`` API."""

    name: str
    module: Any
    estimated_device_bytes: int
    current_device: str = "cpu"

    @property
    def value(self) -> Any:
        return self.module

    def move_to(self, device: str, *, non_blocking: bool) -> None:
        self.module.to(device, non_blocking=non_blocking)
        self.current_device = device


@dataclass(frozen=True, slots=True)
class _TensorSlot:
    """One registered tensor location inside an inference-only module."""

    owner: Any
    name: str
    is_parameter: bool
    requires_grad: bool
    master_index: int
    module_path: str


class ImmutablePinnedModuleResidency:
    """Fast residency for immutable inference weights.

    ``torch.nn.Module.to("cpu")`` copies every device weight back over PCIe.
    That is unnecessary for inference-only checkpoints: their authoritative
    values never change.  This adapter creates one host master per registered
    parameter/buffer, optionally in CUDA pinned memory, and switches the
    module between host masters and fresh device copies.  Eviction therefore
    consists only of rebinding the registered slots and releasing device
    tensors; no D2H transfer is performed.

    The adapter deliberately covers only registered parameters and buffers.
    Native model adapters must register every resident tensor (including LoRA
    pairs) so hidden per-forward H2D copies cannot bypass this lifecycle.
    """

    def __init__(
        self,
        name: str,
        module: Any,
        estimated_device_bytes: int | None = None,
        *,
        pin_host_weights: bool = True,
        copy_host_weights: bool = True,
    ) -> None:
        self.name = name
        self.module = module
        self.pin_host_weights = bool(pin_host_weights)
        self.copy_host_weights = bool(copy_host_weights)
        if self.pin_host_weights and not self.copy_host_weights:
            raise ValueError("pinned host weights require an immutable copied master")
        self.current_device = "cpu"
        self._slots: list[_TensorSlot] = []
        self._host_masters: list[Any] = []
        self._host_slabs: list[Any] = []
        self._host_allocated_bytes = 0
        self._prepared = False
        self._copy_streams: dict[str, Any] = {}
        self.estimated_device_bytes = (
            int(estimated_device_bytes)
            if estimated_device_bytes is not None
            else self._registered_nbytes(module)
        )

    @staticmethod
    def _registered_nbytes(module: Any) -> int:
        seen: set[int] = set()
        total = 0
        for submodule in module.modules():
            for tensor in (*submodule._parameters.values(), *submodule._buffers.values()):
                if tensor is None or id(tensor) in seen:
                    continue
                seen.add(id(tensor))
                total += int(tensor.numel()) * int(tensor.element_size())
        return total

    @property
    def value(self) -> Any:
        return self.module

    @property
    def prepared(self) -> bool:
        return self._prepared

    @property
    def host_bytes(self) -> int:
        return sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in self._host_masters
        )

    @property
    def host_is_pinned(self) -> bool:
        return bool(self._host_masters) and all(
            bool(tensor.is_pinned()) for tensor in self._host_masters
        )

    @property
    def host_allocated_bytes(self) -> int:
        """Pinned slab capacity, including small alignment/tail padding."""

        return self._host_allocated_bytes

    def prepare_host(self) -> None:
        """Materialize immutable host masters once.

        Preparation is intentionally explicit so a service can overlap it
        with other cold-start work before the component is first requested.
        Calling it more than once is a no-op.
        """

        if self._prepared:
            return

        identity_to_master: dict[int, int] = {}
        slots: list[_TensorSlot] = []
        sources: list[Any] = []

        for module_path, submodule in self.module.named_modules(remove_duplicate=True):
            for is_parameter, registry in (
                (True, submodule._parameters),
                (False, submodule._buffers),
            ):
                for slot_name, tensor in tuple(registry.items()):
                    if tensor is None:
                        continue
                    if tensor.device.type != "cpu":
                        raise RuntimeError(
                            f"{self.name}.{slot_name} must be on CPU before host preparation"
                        )

                    source_identity = id(tensor)
                    master_index = identity_to_master.get(source_identity)
                    if master_index is None:
                        source = tensor.detach()
                        master_index = len(sources)
                        identity_to_master[source_identity] = master_index
                        sources.append(source)

                    slots.append(
                        _TensorSlot(
                            owner=submodule,
                            name=slot_name,
                            is_parameter=is_parameter,
                            requires_grad=bool(getattr(tensor, "requires_grad", False)),
                            master_index=master_index,
                            module_path=module_path,
                        )
                    )

        if self.pin_host_weights:
            packed = pack_pinned_tensors(sources)
            masters = list(packed.tensors)
            self._host_slabs = list(packed.slabs)
            self._host_allocated_bytes = packed.allocated_bytes
        elif self.copy_host_weights:
            # Keep distinct immutable masters. This prevents later slot
            # rebinding from losing the authoritative CPU values.
            masters = [
                source.clone(memory_format=import_module("torch").preserve_format)
                for source in sources
            ]
            self._host_allocated_bytes = sum(
                int(tensor.numel()) * int(tensor.element_size())
                for tensor in masters
            )
        else:
            # Low-RAM mode retains the graph's existing immutable CPU storage.
            # Most INT8 checkpoint tensors remain file-backed and reclaimable;
            # H2D copies are synchronous because the pages are not pinned.
            masters = sources
            self._host_allocated_bytes = sum(
                int(tensor.numel()) * int(tensor.element_size())
                for tensor in masters
            )

        self._slots = slots
        self._host_masters = masters
        self._bind(self._host_masters)
        self._prepared = True
        self.current_device = "cpu"

    def _bind(self, tensors: list[Any]) -> None:
        torch = import_module("torch")
        parameter_cache: dict[tuple[int, bool], Any] = {}
        for slot in self._slots:
            tensor = tensors[slot.master_index]
            if slot.is_parameter:
                cache_key = (slot.master_index, slot.requires_grad)
                parameter = parameter_cache.get(cache_key)
                if parameter is None:
                    parameter = torch.nn.Parameter(
                        tensor,
                        requires_grad=slot.requires_grad,
                    )
                    parameter_cache[cache_key] = parameter
                slot.owner._parameters[slot.name] = parameter
            else:
                slot.owner._buffers[slot.name] = tensor

    def _move_to_cuda(self, device: str) -> None:
        torch = import_module("torch")
        target = torch.device(device)
        stream = self._copy_streams.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device=target)
            self._copy_streams[device] = stream

        device_tensors: list[Any] = []
        with torch.cuda.device(target), torch.cuda.stream(stream):
            for master in self._host_masters:
                device_tensors.append(
                    master.to(device=target, non_blocking=self.pin_host_weights)
                )
            ready = torch.cuda.Event()
            ready.record(stream)

        # The caller may execute on a different stream. A stream dependency is
        # sufficient; there is no device-wide synchronization here.
        torch.cuda.current_stream(target).wait_event(ready)
        self._bind(device_tensors)

    @staticmethod
    def _path_is_under(module_path: str, prefixes: tuple[str, ...]) -> bool:
        return any(
            module_path == prefix or module_path.startswith(f"{prefix}.")
            for prefix in prefixes
        )

    def move_partition_to_cuda(
        self,
        device: str,
        *,
        host_module_prefixes: tuple[str, ...],
    ) -> None:
        """Move a module shell to CUDA while selected submodules stay on host.

        This is the weight-residency primitive used by H3 block offload: the
        patch/refiner/final layers are resident, while ``block_stack.blocks``
        remains backed by immutable pinned host masters. Two separately
        allocated device block buffers then stream those source weights.

        Prefixes are qualified paths from ``module.named_modules()``. A shared
        registered tensor cannot straddle the host/device boundary because
        breaking such an alias would silently change the model graph.
        """

        if not device.startswith("cuda:"):
            raise ValueError("partitioned residency requires an explicit CUDA device")
        prefixes = tuple(prefix.strip(".") for prefix in host_module_prefixes)
        if not prefixes or any(not prefix for prefix in prefixes):
            raise ValueError("host_module_prefixes must contain qualified module paths")
        if not self._prepared:
            self.prepare_host()

        usages: dict[int, set[bool]] = {}
        for slot in self._slots:
            usages.setdefault(slot.master_index, set()).add(
                self._path_is_under(slot.module_path, prefixes)
            )
        aliased_across_boundary = [
            index for index, locations in usages.items() if len(locations) > 1
        ]
        if aliased_across_boundary:
            raise RuntimeError(
                "registered tensor aliases cross the requested host/device partition: "
                f"{aliased_across_boundary[:8]}"
            )

        torch = import_module("torch")
        target = torch.device(device)
        stream = self._copy_streams.get(device)
        if stream is None:
            stream = torch.cuda.Stream(device=target)
            self._copy_streams[device] = stream

        host_indices = {
            index for index, locations in usages.items() if locations == {True}
        }
        bound_tensors: list[Any] = list(self._host_masters)
        with torch.cuda.device(target), torch.cuda.stream(stream):
            for index, master in enumerate(self._host_masters):
                if index not in host_indices:
                    bound_tensors[index] = master.to(
                        device=target,
                        non_blocking=self.pin_host_weights,
                    )
            ready = torch.cuda.Event()
            ready.record(stream)

        torch.cuda.current_stream(target).wait_event(ready)
        self._bind(bound_tensors)
        self.current_device = f"partitioned:{device}"

    def move_to(self, device: str, *, non_blocking: bool) -> None:
        del non_blocking  # Transfers are batched on the adapter's copy stream.
        if not self._prepared:
            self.prepare_host()
        if device == self.current_device:
            return
        if device == "cpu":
            self._bind(self._host_masters)
            self.current_device = "cpu"
            return
        if not device.startswith("cuda:"):
            raise ValueError("immutable module residency supports only CPU and CUDA")
        self._move_to_cuda(device)
        self.current_device = device


@dataclass(slots=True)
class HostComponentResidency:
    """Adapter for schedulers, muxers, and other intentionally host-only code."""

    name: str
    component: Any
    estimated_device_bytes: int = 0

    @property
    def value(self) -> Any:
        return self.component

    def move_to(self, device: str, *, non_blocking: bool) -> None:
        if device != "cpu":
            raise RuntimeError(f"host-only component {self.name!r} cannot move to {device}")


class ResidencyManager:
    """Own component transitions for text -> denoise -> decode phases.

    The manager is intentionally policy-only: weight format adapters remain in
    the model layer.  Device budget checks happen before movement, stale
    components leave the GPU before new components enter it, and all decisions
    are exposed through ``active_names`` for profiling and tests.
    """

    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self._components: dict[str, ComponentResidency] = {}
        self._active: set[str] = set()

    def register(self, component: ComponentResidency) -> None:
        if component.name in self._components:
            raise ValueError(f"component already registered: {component.name}")
        if component.estimated_device_bytes < 0:
            raise ValueError("estimated_device_bytes cannot be negative")
        self._components[component.name] = component

    @property
    def active_names(self) -> frozenset[str]:
        return frozenset(self._active)

    def component(self, name: str) -> Any:
        try:
            return self._components[name].value
        except KeyError as exc:
            raise KeyError(f"native pipeline component is not registered: {name}") from exc

    def _validate_budget(self, names: set[str]) -> None:
        unknown = names.difference(self._components)
        if unknown:
            raise KeyError(f"unknown native pipeline components: {sorted(unknown)}")
        required = sum(self._components[name].estimated_device_bytes for name in names)
        if required > self.config.max_device_bytes:
            mib = 1024**2
            raise ResidencyBudgetError(
                f"phase requires {required / mib:.1f} MiB but runtime budget is "
                f"{self.config.max_device_bytes / mib:.1f} MiB"
            )

    def transition(self, active_names: Iterable[str]) -> None:
        desired = set(active_names)
        self._validate_budget(desired)

        # Eviction-first is the central 24 GiB invariant. Text/Qwen and DiT,
        # or DiT and decoded video, must never overlap accidentally.
        for name in sorted(self._active - desired):
            self._components[name].move_to("cpu", non_blocking=False)
            self._active.remove(name)

        for name in sorted(desired - self._active):
            self._components[name].move_to(self.config.device, non_blocking=True)
            self._active.add(name)

        if self.config.clear_cache_on_phase_transition and self.config.device != "cpu":
            torch = import_module("torch")
            torch.cuda.empty_cache()

    def release_all(self) -> None:
        self.transition(())
