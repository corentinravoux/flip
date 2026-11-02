"""Write and read CovMatrix objects.

A file holds the model attributes, the covariance blocks (flat or matrix form)
and `covariance_prefactor_dict`, all as plain arrays. The coefficients module is
not saved: `read` re-imports it from `model_name`, and `CovMatrix.__init__`
rebuilds the covariance-sum functions.

Formats:
    pickle   one versioned dictionary.
    npz      one array per block (`block__vv`, ...) and per prefactor
             (`prefactor__...`), a JSON header; read with `allow_pickle=False`.
    parquet  one row per array (`name`, `shape`, `values`), JSON header in the
             schema metadata; blocks of different sizes are allowed.

Files are written through a temporary file: a failed write leaves no file.
"""

import importlib
import json
import os
import pickle

import numpy as np

FILE_FORMATS = ("pickle", "npz", "parquet")
FILE_FORMAT_VERSION = 1
SAVED_ATTRIBUTES = (
    "model_name",
    "model_kind",
    "free_par",
    "los_definition",
    "variant",
    "matrix_form",
    "number_densities",
    "number_velocities",
)
_HEADER_KEY = "flip_covariance_format"


def _check_format(file_format):
    """Raise a ValueError if `file_format` is not one of `FILE_FORMATS`."""
    if file_format not in FILE_FORMATS:
        raise ValueError(f"Unknown file format {file_format}, choose in {FILE_FORMATS}")


def _to_serializable(value):
    """Convert numpy scalars to Python types for JSON."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def _atomic_write(path, write_function):
    """Write through a temporary file, so a failed write leaves no file."""
    temporary_path = f"{path}.tmp-{os.getpid()}"
    try:
        with open(temporary_path, "wb") as file:
            write_function(file)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _as_arrays(dictionary):
    """Convert the values of a dictionary to numpy arrays, keeping None."""
    if dictionary is None:
        return None
    return {key: np.asarray(value) for key, value in dictionary.items()}


def _named_arrays(covariance_dict, covariance_prefactor_dict):
    """Merge blocks and prefactors into one dictionary with prefixed keys."""
    arrays = {f"block__{k}": v for k, v in covariance_dict.items()}
    if covariance_prefactor_dict is not None:
        arrays.update({f"prefactor__{k}": v for k, v in covariance_prefactor_dict.items()})
    return arrays


def _split_arrays(arrays, header):
    """Invert `_named_arrays`, using the key lists stored in the header."""
    covariance_dict = {key: arrays[f"block__{key}"] for key in header["blocks"]}
    covariance_prefactor_dict = (
        None
        if header["prefactors"] is None
        else {key: arrays[f"prefactor__{key}"] for key in header["prefactors"]}
    )
    return covariance_dict, covariance_prefactor_dict


# ----------------------------------------------------------------------------- write
def _write_pickle(path, header, covariance_dict, covariance_prefactor_dict):
    payload = {
        _HEADER_KEY: FILE_FORMAT_VERSION,
        "attributes": header["attributes"],
        "covariance_dict": covariance_dict,
        "covariance_prefactor_dict": covariance_prefactor_dict,
    }
    _atomic_write(path, lambda file: pickle.dump(payload, file))


def _write_npz(path, header, covariance_dict, covariance_prefactor_dict):
    arrays = {_HEADER_KEY: np.array(json.dumps(header))}
    arrays.update(_named_arrays(covariance_dict, covariance_prefactor_dict))
    _atomic_write(path, lambda file: np.savez(file, **arrays))


def _write_parquet(path, header, covariance_dict, covariance_prefactor_dict):
    import pyarrow
    import pyarrow.parquet

    arrays = _named_arrays(covariance_dict, covariance_prefactor_dict)
    # All arrays are flattened into one large list column, sliced by offsets.
    offsets = np.cumsum([0] + [np.size(v) for v in arrays.values()])
    flat_values = np.concatenate(
        [np.ravel(v).astype(np.float64) for v in arrays.values()]
    )
    table = pyarrow.table(
        {
            "name": pyarrow.array(list(arrays), type=pyarrow.string()),
            "shape": pyarrow.array(
                [list(np.shape(v)) for v in arrays.values()],
                type=pyarrow.list_(pyarrow.int64()),
            ),
            "values": pyarrow.LargeListArray.from_arrays(
                offsets, pyarrow.array(flat_values)
            ),
        }
    ).replace_schema_metadata({_HEADER_KEY: json.dumps(header)})
    _atomic_write(path, lambda file: pyarrow.parquet.write_table(table, file))


_WRITERS = {"pickle": _write_pickle, "npz": _write_npz, "parquet": _write_parquet}


def write(covariance, filename, file_format):
    """Write a CovMatrix to `{filename}.{file_format}`.

    Args:
        covariance (CovMatrix): Covariance to save.
        filename (str): Output path without extension.
        file_format (str): One of `pickle`, `npz`, `parquet`.

    Raises:
        ValueError: For an unknown format.
        NotImplementedError: For emulator covariances (callable blocks).
        TypeError: If a block or prefactor is not a numeric array.
    """
    _check_format(file_format)
    if covariance.emulator_flag:
        raise NotImplementedError(
            "Emulator covariances hold callables and cannot be written"
        )

    covariance_dict = _as_arrays(covariance.covariance_dict)
    covariance_prefactor_dict = _as_arrays(covariance.covariance_prefactor_dict)
    for name, value in _named_arrays(covariance_dict, covariance_prefactor_dict).items():
        if value.dtype == object:
            raise TypeError(f"{name} is not a numeric array and cannot be written")

    header = {
        "version": FILE_FORMAT_VERSION,
        "attributes": {
            key: _to_serializable(getattr(covariance, key)) for key in SAVED_ATTRIBUTES
        },
        "blocks": list(covariance_dict),
        "prefactors": (
            None if covariance_prefactor_dict is None else list(covariance_prefactor_dict)
        ),
    }
    _WRITERS[file_format](
        f"{filename}.{file_format}", header, covariance_dict, covariance_prefactor_dict
    )


# ----------------------------------------------------------------------------- read
def _read_pickle(path):
    with open(path, "rb") as file:
        payload = pickle.load(file)
    if _HEADER_KEY in payload:
        attributes = payload["attributes"]
    else:
        # Older files: the whole vars(self) dictionary.
        attributes = {k: payload[k] for k in SAVED_ATTRIBUTES if k in payload}
    return (
        attributes,
        payload["covariance_dict"],
        payload.get("covariance_prefactor_dict"),
    )


def _read_npz(path):
    with np.load(path, allow_pickle=False) as file:
        if _HEADER_KEY not in file:
            raise ValueError(f"{path} was not written by CovMatrix.write")
        header = json.loads(str(file[_HEADER_KEY]))
        arrays = {key: file[key] for key in file.files if key != _HEADER_KEY}
    return (header["attributes"], *_split_arrays(arrays, header))


def _read_parquet(path):
    import pyarrow.parquet

    table = pyarrow.parquet.read_table(path)
    metadata = table.schema.metadata or {}
    if _HEADER_KEY.encode() not in metadata:
        raise ValueError(f"{path} was not written by CovMatrix.write")
    header = json.loads(metadata[_HEADER_KEY.encode()])
    values = table.column("values").combine_chunks()
    flat_values = values.values.to_numpy()
    offsets = values.offsets.to_numpy()
    arrays = {
        name: flat_values[offsets[i] : offsets[i + 1]].reshape(shape)
        for i, (name, shape) in enumerate(
            zip(table.column("name").to_pylist(), table.column("shape").to_pylist())
        )
    }
    return (header["attributes"], *_split_arrays(arrays, header))


_READERS = {"pickle": _read_pickle, "npz": _read_npz, "parquet": _read_parquet}


def read(filename, file_format):
    """Read `{filename}.{file_format}` written by `write`.

    Args:
        filename (str): Path without extension.
        file_format (str): One of `pickle`, `npz`, `parquet`.

    Returns:
        dict: Keyword arguments of `CovMatrix.__init__` (blocks as numpy arrays,
        coefficients module re-imported from `model_name`).

    Raises:
        ValueError: For an unknown format or a file not written by flip.
    """
    _check_format(file_format)
    attributes, covariance_dict, covariance_prefactor_dict = _READERS[file_format](
        f"{filename}.{file_format}"
    )
    coefficients = importlib.import_module(
        f"flip.covariance.analytical.{attributes['model_name']}.coefficients"
    )
    return {
        **attributes,
        "covariance_dict": covariance_dict,
        "coefficients": coefficients,
        "covariance_prefactor_dict": covariance_prefactor_dict,
    }
