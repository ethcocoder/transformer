"""Python ctypes wrapper for the Aurelis LENS trainer with BPE."""

import ctypes
import json
import os
from pathlib import Path
from typing import List, Optional, Union

_LIB_NAMES = [
    "libaurelis_capi.so",
    "libaurelis_capi.dylib",
    "aurelis_capi.dll",
]

_lib = None
for name in _LIB_NAMES:
    p = Path(__file__).parent.parent.parent / "build" / name
    if p.exists():
        _lib = ctypes.CDLL(str(p))
        break


class AurelisError(Exception):
    """Raised when an Aurelis C API call fails."""


class Trainer:
    """High-level Python wrapper for the Aurelis LENS trainer with BPE."""

    def __init__(self, config: Optional[dict] = None):
        if _lib is None:
            raise RuntimeError(
                "Cannot find libaurelis_capi. Build first:\n"
                "  cmake -B build && cmake --build build"
            )
        self._setup_ctypes()
        config_json = json.dumps(config or {})
        self._handle = _lib.aurelis_trainer_create(
            config_json.encode("utf-8")
        )
        if not self._handle:
            raise AurelisError("Failed to create trainer")

    def _setup_ctypes(self):
        _lib.aurelis_trainer_create.argtypes = [ctypes.c_char_p]
        _lib.aurelis_trainer_create.restype = ctypes.c_void_p

        _lib.aurelis_trainer_destroy.argtypes = [ctypes.c_void_p]
        _lib.aurelis_trainer_destroy.restype = None

        _lib.aurelis_trainer_load_data.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p
        ]
        _lib.aurelis_trainer_load_data.restype = ctypes.c_int

        _lib.aurelis_trainer_train_epoch.argtypes = [ctypes.c_void_p]
        _lib.aurelis_trainer_train_epoch.restype = ctypes.c_int

        _lib.aurelis_trainer_get_loss.argtypes = [ctypes.c_void_p]
        _lib.aurelis_trainer_get_loss.restype = ctypes.c_float

        _lib.aurelis_trainer_get_accuracy.argtypes = [ctypes.c_void_p]
        _lib.aurelis_trainer_get_accuracy.restype = ctypes.c_float

        _lib.aurelis_trainer_vocab_size.argtypes = [ctypes.c_void_p]
        _lib.aurelis_trainer_vocab_size.restype = ctypes.c_int

        _lib.aurelis_trainer_num_params.argtypes = [ctypes.c_void_p]
        _lib.aurelis_trainer_num_params.restype = ctypes.c_size_t

        _lib.aurelis_trainer_last_error.argtypes = [ctypes.c_void_p]
        _lib.aurelis_trainer_last_error.restype = ctypes.c_char_p

    def load_data(self, path: str) -> None:
        """Load text data, train BPE tokenizer, prepare corpus."""
        status = _lib.aurelis_trainer_load_data(
            self._handle, path.encode("utf-8")
        )
        if status != 0:
            err = _lib.aurelis_trainer_last_error(self._handle)
            raise AurelisError(
                f"load_data failed (code={status}): {err.decode()}"
            )

    def train_epoch(self) -> float:
        """Run one epoch of training. Returns average loss."""
        status = _lib.aurelis_trainer_train_epoch(self._handle)
        if status != 0:
            err = _lib.aurelis_trainer_last_error(self._handle)
            raise AurelisError(
                f"train_epoch failed (code={status}): {err.decode()}"
            )
        return self.loss

    @property
    def loss(self) -> float:
        return _lib.aurelis_trainer_get_loss(self._handle)

    @property
    def accuracy(self) -> float:
        return _lib.aurelis_trainer_get_accuracy(self._handle)

    @property
    def vocab_size(self) -> int:
        return _lib.aurelis_trainer_vocab_size(self._handle)

    @property
    def num_params(self) -> int:
        return _lib.aurelis_trainer_num_params(self._handle)

    def close(self):
        if self._handle:
            _lib.aurelis_trainer_destroy(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
