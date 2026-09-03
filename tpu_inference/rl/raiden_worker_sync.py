# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the tpu-inference project
"""Binds a TPUWorker's live model weights to the Raiden transport, in-process.

Runs inside the EngineCore subprocess where the live TPU arrays live; only
plain-data metadata (`RaidenWorkerSync.metadata_dict`) crosses the
`collective_rpc` boundary back to `RLVllmSampler`.

Duplicates (rather than imports) tunix's
`raiden_synchronizer.RaidenSynchronizer` binding mechanics, since
`vllm_sampler.py` has no Tunix dependency. Keep the two in sync by hand;
tunix's `weight_sync.dict_to_metadata` defines the metadata shape.
"""

from __future__ import annotations

import inspect
import ipaddress
import os
import socket
import time
from typing import Any, List, Optional, Tuple

import jax
import jax.numpy as jnp

import logging

logger = logging.getLogger(__name__)

_ws_lib: Any = None
_RAIDEN_IMPORT_ERROR: Optional[Exception] = None

_raiden_ffi: Any = None
_FFI_IMPORT_TRIED = False


def _get_ffi() -> Any:
    """Imports the Raiden FFI module on first use.

    Deliberately lazy: `_weight_synchronizer_ffi.so` statically links its own
    ~130MB XLA runtime, and loading it alongside `_tpu_raiden_jax.so` in the
    EngineCore leaves the legacy `WeightSynchronizer` constructor without enough
    address space (`MemoryError: std::bad_alloc` at bind). Only the FFI path
    should pay that cost.
    """
    global _raiden_ffi, _FFI_IMPORT_TRIED
    if not _FFI_IMPORT_TRIED:
        _FFI_IMPORT_TRIED = True
        try:
            from tpu_sync.frameworks.jax import \
                weight_synchronizer_ffi as mod  # pylint: disable=g-import-not-at-top
            _raiden_ffi = mod
        except ImportError:
            _raiden_ffi = None
    return _raiden_ffi


def use_ffi() -> bool:
    """Whether to drive weight sync through Raiden's FFI path.

    A direct-TPU (mcjax) rollout must set RAIDEN_USE_FFI=0: importing
    weight_synchronizer_ffi registers nothing into jaxlib's FFI registry, so
    its `compute_type="device_host"` handlers never resolve here. An FFI
    source paired with a legacy destination also fails, as
    `INVALID_ARGUMENT: Destination out of bounds in batched push`, so the flag
    must match on both sides.
    """
    default = "1" if "proxy" in os.environ.get("JAX_PLATFORMS", "") else "0"
    if os.environ.get("RAIDEN_USE_FFI", default) != "1":
        return False
    return _get_ffi() is not None


def _ensure_ffi_compute_on_compat() -> None:
    """Bridges TPU-sync wheels that call the newer compute_on decorator API."""
    compute_on_mod = getattr(jax, "_src", None)
    if compute_on_mod is None:
        return
    compute_on_mod = getattr(compute_on_mod, "compute_on", None)
    if compute_on_mod is None:
        return
    try:
        params = inspect.signature(compute_on_mod.compute_on).parameters
    except (TypeError, ValueError):
        params = {}
    if "out_memory_spaces" in params:
        return
    compute_on2 = getattr(compute_on_mod, "compute_on2", None)
    if compute_on2 is None:
        raise RuntimeError(
            "Installed JAX lacks compute_on compatibility required by the"
            " TPU-sync FFI wheel.")
    compute_on_mod.compute_on = compute_on2
    # The wheel does `from jax.experimental import compute_on` and calls
    # `compute_on.compute_on(...)`, binding the name at import time, so
    # patching jax._src alone leaves it on the old two-arg version:
    #   TypeError: compute_on() got an unexpected keyword argument
    #   'out_memory_spaces'
    try:
        from jax.experimental import compute_on as _public_compute_on
        _public_compute_on.compute_on = compute_on2
    except ImportError:
        pass


def unpack_ip(row: Any) -> str:
    """Unpacks an IP address from the FFI synchronizer metadata row."""
    raw_bytes = b"".join(
        int(x).to_bytes(4, byteorder="little", signed=True) for x in row[:4])
    if raw_bytes[:10] == b"\x00" * 10 and raw_bytes[10:12] == b"\xff\xff":
        return str(ipaddress.IPv4Address(raw_bytes[12:16]))
    addr_str = str(ipaddress.IPv6Address(raw_bytes))
    return f"[{addr_str}]" if ":" in addr_str else addr_str
try:
    from tpu_sync.api.jax import \
        weight_synchronizer as _ws_lib  # pylint: disable=g-import-not-at-top
except ImportError as exc:
    _RAIDEN_IMPORT_ERROR = exc


def local_ip() -> str:
    for family, probe in (
        (socket.AF_INET, ("8.8.8.8", 80)),
        (socket.AF_INET6, ("2001:4860:4860::8888", 80)),
    ):
        try:
            s = socket.socket(family, socket.SOCK_DGRAM)
            try:
                s.connect(probe)
                ip = s.getsockname()[0]
            finally:
                s.close()
            return f"[{ip}]" if ":" in ip else ip
        except OSError:
            continue
    return "localhost"


def is_maxtext_model(model: Any) -> bool:
    """True if `model` is a MaxTextForCausalLM (checked by class name to
    avoid a maxtext dependency)."""
    return type(model).__name__ == "MaxTextForCausalLM"


def extract_weight_state(state: Any, model: Any) -> Any:
    """Returns the Param state ready for Raiden binding/inspection.

    For MaxText, params nest one level down under the wrapper's `model`
    key (not `base`, unlike the trainer side); unwrapped here either way.
    """
    maxtext = is_maxtext_model(model)
    if state is not None:
        if maxtext:
            try:
                return {"base": state["model"]}
            except (KeyError, TypeError):
                pass
        return state
    if model is not None:
        from flax import nnx
        if maxtext:
            inner = getattr(model, "model", None)
            if inner is not None:
                return {"base": nnx.state(inner, nnx.Param)}
            return None
        return nnx.state(model, nnx.Param)
    return None


def flatten_weights(state: Any) -> Tuple[List[str], List[Any]]:
    """Returns (names, arrays) for every array leaf, in stable tree order."""
    names, arrays = [], []
    for path, leaf in jax.tree_util.tree_leaves_with_path(state):
        arr = getattr(leaf, "value", leaf)
        if hasattr(arr, "shape") and hasattr(arr, "dtype"):
            names.append(jax.tree_util.keystr(path))
            arrays.append(arr)
    return names, arrays


def _bindable(arr: Any) -> bool:
    """True if the native layer can bind this leaf."""
    try:
        if not hasattr(arr, "shape") or not hasattr(arr, "dtype"):
            return False
        if arr.ndim < 1:
            return False
        if not jnp.issubdtype(arr.dtype, jnp.floating):
            return False
        devices = arr.devices()
        if not devices:
            return False
        return all(getattr(d, "platform", "?") == "tpu" for d in devices)
    except Exception:
        return False


# KV-cache leaves live in the same state tree as the weights and are ordinary
# float arrays, so _bindable accepts them -- but the trainer has no counterpart,
# and the controller pairs by name. qwen3.5-35b's attention layers surface
# `attention.cache.cached_prefill_{key,value}`, which showed up as 240
# destination-only entries in the manifest preflight.
_NON_WEIGHT_PATH_PARTS = ("['cache']", )


def _is_weight(name: str) -> bool:
    return not any(part in name for part in _NON_WEIGHT_PATH_PARTS)


def _filter_bindable(names: List[str],
                     arrays: List[Any]) -> Tuple[List[str], List[Any]]:
    """Drops leaves the native layer cannot bind, and non-weight state."""
    keep_names: List[str] = []
    keep_arrays: List[Any] = []
    dropped_non_weight = 0
    for name, arr in zip(names, arrays):
        if not _is_weight(name):
            dropped_non_weight += 1
            continue
        if _bindable(arr):
            if hasattr(arr, "block_until_ready"):
                arr.block_until_ready()
            keep_names.append(name)
            keep_arrays.append(arr)
    if dropped_non_weight:
        logger.info("skipped %d non-weight leaves (KV cache) when binding",
                    dropped_non_weight)
    return keep_names, keep_arrays


def _axis_name(axis: Any) -> str:
    if axis is None:
        return ""
    if isinstance(axis, str):
        return axis
    return ",".join(axis)


def _tensor_metadata_dict(name: str, arr: Any, layer_idx: int) -> dict:
    sharding: Any = getattr(arr, "sharding", None)
    spec = tuple(getattr(sharding, "spec", ()) or ())
    spec = (spec + (None, ) * arr.ndim)[:arr.ndim]
    try:
        local = sharding.shard_shape(tuple(arr.shape))
        mesh_shape = tuple(g // s for g, s in zip(arr.shape, local))
    except Exception:  # pylint: disable=broad-exception-caught
        mesh_shape = (1, ) * arr.ndim
    return {
        "name": name,
        "shape": list(arr.shape),
        "mesh_shape": list(mesh_shape),
        "layout": list(reversed(range(arr.ndim))),
        "item_size": arr.dtype.itemsize,
        "layer_idx": layer_idx,
        "sharding_spec": [_axis_name(a) for a in spec],
    }


def _reduce_mesh(mesh: Any) -> Any:
    """Drops size-1 axes from a mesh for Raiden's FFI shard_map.

    `init_weight_synchronizer` specs its inputs as `P(*mesh.axis_names)`, so a
    mesh carrying six singleton axes yields a spec longer than the operands'
    rank and shard_map rejects it. Only the trivial axes are dropped -- the
    real sharding (e.g. attn_dp x model) is preserved, and the anchor is chosen
    to have at least the mesh's rank; see _ffi_anchor.
    """
    try:
        keep = [a for a in mesh.axis_names if int(mesh.shape[a]) > 1]
    except Exception:  # noqa: BLE001
        return mesh
    if not keep or len(keep) == len(mesh.axis_names):
        return mesh
    shape = tuple(int(mesh.shape[a]) for a in keep)
    reduced = jax.sharding.Mesh(mesh.devices.reshape(shape), tuple(keep))
    logger.info("raiden ffi: reduced mesh %s -> %s", mesh.axis_names,
                reduced.axis_names)
    return reduced


def _ffi_anchor(arrays: List[Any], mesh: Any) -> Any:
    """Picks the array Raiden uses to anchor its shard_map.

    `init_weight_synchronizer` builds
    `P(*axis_names, *[None] * (anchor.ndim - len(axis_names)))`, so the anchor
    must have at least as many dims as the mesh has axes -- otherwise the pad
    count goes negative and shard_map reports an in_specs entry "too long".
    arrays[0] is whatever sorts first, a rank-1 norm here, which is fine for a
    1-D mesh but not once attention DP makes the mesh 2-D.
    """
    rank = len(mesh.axis_names)
    for arr in arrays:
        if getattr(arr, "ndim", 0) >= rank:
            return arr
    return max(arrays, key=lambda a: getattr(a, "ndim", 0))


class RaidenWorkerSync:
    """One TPUWorker's weights on the Raiden transport, plus wire-safe metadata.

    In-process counterpart to tunix's `RaidenSynchronizer`. Construct once
    per `TPUWorker`, rebind on every sync round via `bind()`.
    """

    def __init__(
        self,
        job_name: str,
        *,
        worker_index: int = 0,
        parallelism: int = 4,
        bind_ip: Optional[str] = None,
    ):
        self.job_name = job_name
        self.worker_index = worker_index
        self.names: List[str] = []
        self.arrays: List[Any] = []
        self.ip = bind_ip or local_ip()
        self._parallelism = parallelism
        self._sync: Any = None
        self._ips: List[str] = []
        self._unique_listeners: List[str] = []
        self._use_ffi: bool = use_ffi()
        self._ffi_mesh: Any = None
        self._ffi_shard_idx: Any = None
        # Captured at bind() so the FFI path can rebuild the model's weight
        # tree from the arrays multi_h2d returns. See synced_state().
        self._treedef: Any = None
        self._all_paths: List[str] = []
        self._all_leaves: List[Any] = []

    @property
    def bound(self) -> bool:
        return bool(self.names)

    def _init_ffi_transport(self) -> None:
        """Stands up the FFI destination transport for the bound arrays."""
        if not self.arrays:
            raise RuntimeError(
                f"{self.job_name}: bind() must stage arrays before FFI init")
        import numpy as _np
        from jax.experimental import multihost_utils

        _ensure_ffi_compute_on_compat()
        mesh = getattr(getattr(self.arrays[0], "sharding", None), "mesh", None)
        if mesh is None:
            raise ValueError(
                "Arrays must be sharded on a Mesh for FFI weight sync.")
        mesh = _reduce_mesh(mesh)

        slice_byte_sizes = [
            int(
                _np.prod(arr.sharding.shard_shape(arr.shape)) *
                arr.dtype.itemsize) for arr in self.arrays
        ]
        slice_byte_sizes_sharded = jax.device_put(
            jnp.array(slice_byte_sizes, dtype=jnp.int32),
            jax.sharding.NamedSharding(mesh,
                                       jax.sharding.PartitionSpec(None)))

        task_mesh_shape = tuple(mesh.shape[a] for a in mesh.axis_names)
        global_ids = jnp.array([d.id for d in mesh.devices.flatten()],
                               dtype=jnp.int32).reshape(task_mesh_shape)
        shard_idx = jax.device_put(
            global_ids,
            jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec(*mesh.axis_names)))

        devices = mesh.devices.flatten()
        num_processes = len(
            set(getattr(d, "process_index", 0) for d in devices))
        devices_per_host = len(devices) // max(1, num_processes)

        ws_info = _get_ffi().init_weight_synchronizer(
            device_array=_ffi_anchor(self.arrays, mesh),
            shard_idx=shard_idx,
            mesh=mesh,
            slice_byte_sizes=slice_byte_sizes_sharded,
            parallelism=self._parallelism,
            num_layers=len(self.arrays),
            listener_port=0,
            num_shards=devices_per_host,
        )

        local_ws_info = multihost_utils.global_array_to_host_local_array(
            ws_info, mesh,
            jax.sharding.PartitionSpec(*mesh.axis_names, None))
        gathered = multihost_utils.process_allgather(local_ws_info).reshape(
            -1, 6)

        self._ips, listeners = [], []
        for row in gathered:
            ip = unpack_ip(row)
            self._ips.append(f"{ip}:{int(row[4])}")
            listeners.append(f"{ip}:{int(row[5])}")
        self._unique_listeners = []
        for listener in listeners:
            if listener not in self._unique_listeners:
                self._unique_listeners.append(listener)
        self._ffi_mesh = mesh
        self._ffi_shard_idx = shard_idx

    def _ffi_h2d(self) -> None:
        if self._ffi_mesh is None or self._ffi_shard_idx is None:
            raise RuntimeError(f"{self.job_name}: bind() must run before h2d()")
        # Returns *new* arrays, so the model must be re-pointed afterwards:
        # synced_state() + TPUWorker.refresh_model_state_leaves.
        self.arrays = list(
            _get_ffi().multi_h2d(self.arrays, self._ffi_shard_idx,
                                 self._ffi_mesh))
        jax.block_until_ready(self.arrays)

    def bind(self, state: Any) -> None:
        """Binds (or rebinds after a weight update) this worker's weights."""
        self.names, self.arrays = _filter_bindable(*flatten_weights(state))
        self._treedef, self._all_paths, self._all_leaves = None, [], []
        if self._use_ffi:
            # Snapshot the tree so synced_state() can rebuild it from the
            # arrays multi_h2d returns.
            pairs, self._treedef = jax.tree_util.tree_flatten_with_path(state)
            self._all_paths = [jax.tree_util.keystr(p) for p, _ in pairs]
            self._all_leaves = [leaf for _, leaf in pairs]
            self._init_ffi_transport()
            return
        if _ws_lib is None:
            raise RuntimeError(
                f"{self.job_name}: tpu_sync is not importable, cannot bind "
                f"weight_synchronizer. Original error: {_RAIDEN_IMPORT_ERROR}")
        if self._sync is None:
            self._sync = _ws_lib.WeightSynchronizer(
                self.arrays,
                local_port=0,
                parallelism=self._parallelism,
                unsafe_skip_buffer_lock=True,
                listener_port=0,
                bind_ip=None,
                # Ingest each slice as it lands. At the default (False), h2d()
                # unpacks whatever is staged when it is called, installing
                # in-flight tensors torn -- silently, as checksums that are
                # partway between the initial and synced values. Matches how
                # tunix's in-process destination already binds.
                auto_h2d=True,
            )
        else:
            self._sync.bind_weights(self.arrays)

    def synced_state(self) -> Any:
        """The bound weight tree, rebuilt from whatever h2d produced.

        FFI only. `multi_h2d` returns *new* arrays instead of writing into the
        bound buffers, so `runner.state` still holds the pre-sync ones and
        `refresh_model_state_leaves`, which re-derives leaves from it, would
        republish stale weights -- silently, since checksums are taken over
        the new arrays. Returns None on the legacy path, which DMAs in place.
        """
        if self._treedef is None or not self.arrays:
            return None
        if len(self.names) != len(self.arrays):
            raise RuntimeError(
                f"{self.job_name}: {len(self.names)} names vs "
                f"{len(self.arrays)} arrays; refusing to publish a partial "
                "weight tree")
        by_path = dict(zip(self.names, self.arrays))
        new_leaves = [
            by_path.get(path, leaf)
            for path, leaf in zip(self._all_paths, self._all_leaves)
        ]
        # Release the pre-sync leaves; keeping them pinned a second full copy
        # of the model in HBM for the worker's lifetime.
        self._all_leaves = new_leaves
        return jax.tree_util.tree_unflatten(self._treedef, new_leaves)

    def _require_sync(self, op: str) -> Any:
        if self._sync is None:
            raise RuntimeError(f"{self.job_name}: bind() must run before {op}")
        return self._sync

    def h2d(self) -> None:
        # h2d() is async; block so a checksum/read right after sees the
        # transferred data.
        if self._use_ffi:
            self._ffi_h2d()
            return
        self._require_sync("h2d()").h2d()
        jax.block_until_ready(self.arrays)
        # `block_until_ready` only orders JAX computations. Raiden's H2D is
        # documented as asynchronous and writes these buffers via DMA outside
        # the JAX graph, so it is not a completion barrier -- without the wait
        # below the rollout can resume generating from half-written weights.
        # Measured settle time is 2-4s for Qwen3-0.6B.
        if os.environ.get("RAIDEN_H2D_SETTLE", "1") == "1":
            self._wait_until_settled()

    def _wait_until_settled(self,
                            timeout_s: float = 180.0,
                            interval_s: float = 0.5,
                            stable_reads: int = 3) -> None:
        """Blocks until a digest over all bound arrays stops changing.

        Interim stand-in for a real completion signal; drop it once
        `WeightSynchronizer` exposes one (its API has no wait/join today).
        """
        def digest() -> float:
            # Sum in-device and sync once; a float() per tensor is 310 blocking
            # round trips per poll.
            return float(
                sum(jnp.sum(jnp.abs(a), dtype=jnp.float32)
                    for a in self.arrays))

        prev = None
        stable = 0
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            cur = digest()
            if prev is not None and cur == prev:
                stable += 1
                if stable >= stable_reads:
                    return
            else:
                stable = 0
            prev = cur
            time.sleep(interval_s)
        logger.warning("raiden h2d did not settle within %.0fs", timeout_s)

    def metrics(self) -> dict:
        if self._sync is None:
            return {}
        getter = getattr(self._sync, "get_metrics",
                         getattr(self._sync, "metrics", None))
        return getter() if getter is not None else {}

    def checksums(self, sample: int = 3) -> dict:
        """Per-tensor float32 abs-sums for cross-process verification.

        `__grand_total__` covers every bound tensor: a three-tensor sample
        says nothing about how much of the model actually arrived.
        """

        def total(arr):
            return float(jnp.sum(jnp.abs(arr).astype(jnp.float32)))

        out = {
            name: total(arr)
            for name, arr in list(zip(self.names, self.arrays))[:sample]
        }
        out["__grand_total__"] = float(
            sum(jnp.sum(jnp.abs(a), dtype=jnp.float32) for a in self.arrays))
        # Totals only compare if both sides bound the same tensors.
        out["__tensor_count__"] = len(self.arrays)
        out["__element_count__"] = int(sum(a.size for a in self.arrays))
        return out

    def metadata_dict(self) -> dict:
        """Wire-safe registration metadata, shaped for tunix's
        `weight_sync.dict_to_metadata`."""
        # Positional index, not name-derived -- must match tunix's side.
        variables = [
            _tensor_metadata_dict(name, arr, idx)
            for idx, (name, arr) in enumerate(zip(self.names, self.arrays))
        ]
        mesh = None
        for arr in self.arrays:
            mesh = getattr(getattr(arr, "sharding", None), "mesh", None)
            if mesh is not None:
                break
        if mesh is None:
            raise RuntimeError(
                f"{self.job_name}: no bound array carries a sharding mesh; "
                "cannot determine mesh_shape/mesh_axes for registration "
                "metadata.")
        mesh_axes = tuple(mesh.axis_names)
        mesh_shape = tuple(int(mesh.shape[a]) for a in mesh.axis_names)
        if self._use_ffi:
            # FFI advertises one real endpoint per device rather than a single
            # address repeated num_shards times.
            shards = list(self._ips)
            control_addr = (self._unique_listeners[0]
                            if self._unique_listeners else "")
        else:
            data_addr = f"{self.ip}:{self._sync.local_port}" if self._sync else ""
            control_addr = (f"{self.ip}:{self._sync.listener_port}"
                            if self._sync and self._sync.listener_port else "")
            num_shards = self._sync.num_shards if self._sync else 1
            shards = [data_addr] * num_shards if data_addr else []
        return {
            "unit": {
                "job_name":
                self.job_name,
                "job_replica_id":
                str(self.worker_index) if self.worker_index else "",
            },
            "shards": shards,
            "control_plane_rpc_address": control_addr,
            "mesh_shape": list(mesh_shape),
            "variables": variables,
            "mesh_axes": list(mesh_axes) if mesh_axes else None,
        }
