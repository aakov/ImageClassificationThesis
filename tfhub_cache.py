import hashlib
import os
import tempfile
from pathlib import Path


def get_tfhub_cache_dir():
    configured = os.environ.get("TFHUB_CACHE_DIR")
    if configured:
        return Path(configured)
    return Path(tempfile.gettempdir()) / "tfhub_modules"


def get_cached_module_path(handle):
    digest = hashlib.sha1(handle.encode("utf-8")).hexdigest()
    return get_tfhub_cache_dir() / digest


def validate_cached_tfhub_module(handle):
    module_path = get_cached_module_path(handle)
    if not module_path.exists():
        return

    variables_dir = module_path / "variables"
    has_saved_model = (module_path / "saved_model.pb").is_file()
    has_variables_index = (variables_dir / "variables.index").is_file()
    has_variables_data = any(variables_dir.glob("variables.data-*"))
    if has_saved_model and has_variables_index and has_variables_data:
        return

    raise RuntimeError(
        "Incomplete TensorFlow Hub cache for "
        f"{handle}: {module_path}. Remove only this cached module directory "
        "and run the model again to download a complete copy."
    )
