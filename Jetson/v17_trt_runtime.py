"""TensorRT runtime wrapper for the V17 actor engine.

This module avoids PyCUDA/cuda-python by using ctypes against libcudart.
It is intentionally small: one fixed-shape actor engine, batch size 1.
"""

import ctypes
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import tensorrt as trt


class _CudaRuntime:
    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2

    def __init__(self):
        last_error = None
        for name in ("libcudart.so", "libcudart.so.10.2", "/usr/local/cuda/lib64/libcudart.so"):
            try:
                self.lib = ctypes.CDLL(name)
                break
            except OSError as exc:
                last_error = exc
        else:
            raise OSError("could not load libcudart") from last_error

        self.lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.lib.cudaMalloc.restype = ctypes.c_int
        self.lib.cudaFree.argtypes = [ctypes.c_void_p]
        self.lib.cudaFree.restype = ctypes.c_int
        self.lib.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.lib.cudaMemcpyAsync.restype = ctypes.c_int
        self.lib.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.lib.cudaStreamCreate.restype = ctypes.c_int
        self.lib.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamSynchronize.restype = ctypes.c_int
        self.lib.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cudaStreamDestroy.restype = ctypes.c_int

    def _check(self, code: int, name: str) -> None:
        if int(code) != 0:
            raise RuntimeError("%s failed with cudaError %d" % (name, int(code)))

    def malloc(self, nbytes: int) -> ctypes.c_void_p:
        ptr = ctypes.c_void_p()
        self._check(self.lib.cudaMalloc(ctypes.byref(ptr), int(nbytes)), "cudaMalloc")
        return ptr

    def free(self, ptr: ctypes.c_void_p) -> None:
        if ptr and ptr.value:
            self._check(self.lib.cudaFree(ptr), "cudaFree")

    def stream_create(self) -> ctypes.c_void_p:
        stream = ctypes.c_void_p()
        self._check(self.lib.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
        return stream

    def stream_destroy(self, stream: ctypes.c_void_p) -> None:
        if stream and stream.value:
            self._check(self.lib.cudaStreamDestroy(stream), "cudaStreamDestroy")

    def memcpy_async(self, dst, src, nbytes: int, kind: int, stream: ctypes.c_void_p) -> None:
        self._check(
            self.lib.cudaMemcpyAsync(
                ctypes.c_void_p(int(dst)),
                ctypes.c_void_p(int(src)),
                int(nbytes),
                int(kind),
                stream,
            ),
            "cudaMemcpyAsync",
        )

    def stream_sync(self, stream: ctypes.c_void_p) -> None:
        self._check(self.lib.cudaStreamSynchronize(stream), "cudaStreamSynchronize")


class V17TensorRTActor:
    """Batch-1 TensorRT actor with persistent LSTM state."""

    def __init__(self, engine_path: str, metadata_path: str = None):
        self.engine_path = os.path.abspath(os.path.expanduser(engine_path))
        self.metadata_path = metadata_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.cuda = _CudaRuntime()
        self.stream = self.cuda.stream_create()
        self.bindings: List[int] = []
        self.host: Dict[str, np.ndarray] = {}
        self.device: Dict[str, ctypes.c_void_p] = {}

        self.metadata = self._load_metadata(metadata_path)
        self.engine = self._load_engine(self.engine_path)
        self.context = self.engine.create_execution_context()
        self._allocate_bindings()
        self.reset()

    def _load_metadata(self, metadata_path: str) -> Dict:
        if not metadata_path:
            return {}
        metadata_path = os.path.abspath(os.path.expanduser(metadata_path))
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_engine(self, engine_path: str):
        with open(engine_path, "rb") as f:
            payload = f.read()
        runtime = trt.Runtime(self.logger)
        engine = runtime.deserialize_cuda_engine(payload)
        if engine is None:
            raise RuntimeError("failed to deserialize TensorRT engine: " + engine_path)
        self.runtime = runtime
        return engine

    def _expected_shape(self, name: str) -> Tuple[int, ...]:
        shape = self.metadata.get("shape", {})
        mapping = {
            "image": (1, shape.get("image_channels", 6), shape.get("obs_size", 128), shape.get("obs_size", 128)),
            "state": (1, shape.get("state_dim", 7)),
            "lidar": (1, shape.get("lidar_dim", 144)),
            "lidar_meta": (1, shape.get("lidar_meta_dim", 2)),
            "h": (shape.get("lstm_layers", 2), 1, shape.get("lstm_hidden_size", 256)),
            "c": (shape.get("lstm_layers", 2), 1, shape.get("lstm_hidden_size", 256)),
            "action": (1, 3),
            "next_h": (shape.get("lstm_layers", 2), 1, shape.get("lstm_hidden_size", 256)),
            "next_c": (shape.get("lstm_layers", 2), 1, shape.get("lstm_hidden_size", 256)),
        }
        return tuple(int(x) for x in mapping[name])

    def _binding_shape(self, index: int, name: str) -> Tuple[int, ...]:
        try:
            dims = tuple(int(x) for x in self.engine.get_binding_shape(index))
        except Exception:
            dims = ()
        if not dims or any(x < 0 for x in dims):
            return self._expected_shape(name)
        return dims

    def _allocate_bindings(self) -> None:
        n = int(self.engine.num_bindings)
        self.bindings = [0] * n
        for index in range(n):
            name = self.engine.get_binding_name(index)
            dtype = trt.nptype(self.engine.get_binding_dtype(index))
            shape = self._binding_shape(index, name)
            arr = np.empty(shape, dtype=dtype)
            ptr = self.cuda.malloc(arr.nbytes)
            self.host[name] = arr
            self.device[name] = ptr
            self.bindings[index] = int(ptr.value)

    def binding_summary(self) -> Dict[str, Tuple[int, ...]]:
        return {name: tuple(arr.shape) for name, arr in self.host.items()}

    def reset(self) -> None:
        self.host["h"].fill(0.0)
        self.host["c"].fill(0.0)

    @staticmethod
    def _copy_obs(dst: np.ndarray, value: np.ndarray) -> None:
        arr = np.asarray(value, dtype=dst.dtype)
        if arr.shape == dst.shape[1:]:
            arr = arr.reshape(dst.shape)
        if arr.shape != dst.shape:
            raise ValueError("shape mismatch for input: expected %s got %s" % (dst.shape, arr.shape))
        np.copyto(dst, arr)

    def predict_np(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        self._copy_obs(self.host["image"], obs["image"])
        self._copy_obs(self.host["state"], obs["state"])
        self._copy_obs(self.host["lidar"], obs["lidar"])
        self._copy_obs(self.host["lidar_meta"], obs["lidar_meta"])

        for name in ("image", "state", "lidar", "lidar_meta", "h", "c"):
            arr = self.host[name]
            self.cuda.memcpy_async(
                self.device[name].value,
                arr.ctypes.data,
                arr.nbytes,
                _CudaRuntime.cudaMemcpyHostToDevice,
                self.stream,
            )

        ok = self.context.execute_async_v2(bindings=self.bindings, stream_handle=int(self.stream.value))
        if not ok:
            raise RuntimeError("TensorRT execute_async_v2 returned false")

        for name in ("action", "next_h", "next_c"):
            arr = self.host[name]
            self.cuda.memcpy_async(
                arr.ctypes.data,
                self.device[name].value,
                arr.nbytes,
                _CudaRuntime.cudaMemcpyDeviceToHost,
                self.stream,
            )
        self.cuda.stream_sync(self.stream)

        np.copyto(self.host["h"], self.host["next_h"])
        np.copyto(self.host["c"], self.host["next_c"])
        return np.asarray(self.host["action"], dtype=np.float32).reshape(-1)

    def close(self) -> None:
        for ptr in list(self.device.values()):
            self.cuda.free(ptr)
        self.device.clear()
        self.bindings = []
        if getattr(self, "stream", None):
            self.cuda.stream_destroy(self.stream)
            self.stream = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
