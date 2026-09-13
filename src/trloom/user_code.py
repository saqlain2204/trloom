"""Bundle local user modules so remote jobs can import YAML-referenced callables."""

from __future__ import annotations

import importlib
import logging
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import Any

from trloom.config.schema import FineTuneConfig
from trloom.imports import parse_import_spec

logger = logging.getLogger(__name__)

_SITE_MARKERS = ("site-packages", "dist-packages")


def collect_callable_specs(config: FineTuneConfig) -> list[str]:
    """Return import-path strings referenced by the config (order-preserving, unique)."""
    specs: list[str] = []

    def _add(value: str | None) -> None:
        if value and value.strip():
            specs.append(value.strip())

    _add(config.formatting_func)
    _add(config.dataset.map_fn)
    _add(config.dataset.formatting_func)

    if config.reward_funcs:
        for name in config.reward_funcs:
            if isinstance(name, str) and _looks_like_import_path(name):
                _add(name)

    seen: set[str] = set()
    unique: list[str] = []
    for spec in specs:
        if spec not in seen:
            seen.add(spec)
            unique.append(spec)
    return unique


def _looks_like_import_path(name: str) -> bool:
    """True when ``name`` looks like ``pkg.mod:attr`` / ``pkg.mod.attr``."""
    if ":" in name:
        return True
    parts = name.split(".")
    return len(parts) >= 2 and all(part.isidentifier() for part in parts)


def _stdlib_roots() -> list[Path]:
    roots: list[Path] = []
    for key in ("stdlib", "platstdlib"):
        raw = sysconfig.get_paths().get(key)
        if raw:
            roots.append(Path(raw).resolve())
    return roots


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _is_bundlable_file(path: Path) -> bool:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.suffix != ".py":
        return False

    text = str(resolved).replace("\\", "/")
    if any(marker in text for marker in _SITE_MARKERS):
        return False

    for stdlib_root in _stdlib_roots():
        if _is_under(resolved, stdlib_root):
            if "site-packages" not in text and "dist-packages" not in text:
                return False

    try:
        import trloom as _trloom

        trloom_root = Path(_trloom.__file__).resolve().parent
        if trloom_root == resolved.parent or trloom_root in resolved.parents:
            return False
    except Exception:
        pass

    return True


def _top_level_package_dir(module_name: str) -> tuple[str, Path] | None:
    """Return ``(top_level_name, package_dir_or_module_file)`` for bundling."""
    top_name = module_name.split(".", 1)[0]
    try:
        top = importlib.import_module(top_name)
    except ImportError:
        return None

    file = getattr(top, "__file__", None)
    if not file:
        return None

    path = Path(file).resolve()
    probe = path
    candidate = path.parent if path.name == "__init__.py" else path
    if not _is_bundlable_file(probe):
        return None
    return top_name, candidate


def _bundle_tree(root: Path, *, prefix: str) -> dict[str, str]:
    """Bundle a package directory or single module file under ``prefix``."""
    bundle: dict[str, str] = {}
    if root.is_file():
        bundle[f"{prefix}.py"] = root.read_text(encoding="utf-8")
        return bundle

    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        key = f"{prefix}/{rel}"
        bundle[key] = path.read_text(encoding="utf-8")
    return bundle


def bundle_module(module_name: str) -> dict[str, str]:
    """Bundle a top-level package/module needed to import ``module_name``."""
    located = _top_level_package_dir(module_name)
    if located is None:
        logger.debug("Skipping non-bundlable module '%s'", module_name)
        return {}
    top_name, root = located
    bundled = _bundle_tree(root, prefix=top_name)
    logger.info(
        "Bundled user module '%s' (%d file%s)",
        top_name,
        len(bundled),
        "" if len(bundled) == 1 else "s",
    )
    return bundled


def bundle_path(path: str | Path) -> dict[str, str]:
    """Bundle an explicit file or package directory from ``user_code``."""
    candidate = Path(path).expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"user_code path not found: {candidate}")

    if candidate.is_file():
        if candidate.suffix != ".py":
            raise ValueError(f"user_code file must be a .py module: {candidate}")
        key = candidate.name
        logger.info("Bundled user_code file '%s'", key)
        return {key: candidate.read_text(encoding="utf-8")}

    # Directory → package named after the directory
    prefix = candidate.name
    bundled = _bundle_tree(candidate, prefix=prefix)
    if not bundled:
        raise ValueError(f"user_code directory has no .py files: {candidate}")
    logger.info("Bundled user_code package '%s' (%d files)", prefix, len(bundled))
    return bundled


def prepare_local_user_code(config: FineTuneConfig) -> list[Path]:
    """Ensure ``user_code`` paths are importable for local runs.

    Single ``.py`` files have their parent directory prepended to ``sys.path``.
    Package directories have their parent prepended so ``package_name`` imports work.
    """
    added: list[Path] = []
    for raw in config.user_code:
        candidate = Path(raw).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"user_code path not found: {candidate}")
        parent = candidate.parent
        parent_str = str(parent)
        if parent_str not in sys.path:
            sys.path.insert(0, parent_str)
            added.append(parent)
            logger.info("Added user_code path to sys.path: %s", parent)
    return added


def build_user_code_bundle(config: FineTuneConfig) -> dict[str, str]:
    """Collect local Python sources referenced by the config for remote execution."""
    prepare_local_user_code(config)

    bundle: dict[str, str] = {}

    for path in config.user_code:
        bundle.update(bundle_path(path))

    for spec in collect_callable_specs(config):
        try:
            module_name, _attr = parse_import_spec(spec)
        except ValueError:
            continue
        try:
            bundle.update(bundle_module(module_name))
        except Exception as exc:
            logger.warning("Could not bundle module for '%s': %s", spec, exc)

    return bundle


def install_user_code_bundle(
    bundle: dict[str, str] | None,
    *,
    root: str | Path | None = None,
) -> Path | None:
    """Write a source bundle to disk and prepend it to ``sys.path``.

    Returns the install directory, or ``None`` when ``bundle`` is empty.
    """
    if not bundle:
        return None

    if root is None:
        dest = Path(tempfile.mkdtemp(prefix="trloom_user_code_"))
    else:
        dest = Path(root).expanduser().resolve()
        dest.mkdir(parents=True, exist_ok=True)

    for rel, source in bundle.items():
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")

    dest_str = str(dest)
    if dest_str not in sys.path:
        sys.path.insert(0, dest_str)

    # Drop cached imports so freshly written modules are picked up.
    to_drop = []
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if file and str(Path(file).resolve()).startswith(dest_str):
            to_drop.append(name)
    for name in to_drop:
        sys.modules.pop(name, None)

    logger.info("Installed user_code bundle at %s (%d files)", dest, len(bundle))
    return dest


def ensure_user_code_available(
    config: FineTuneConfig,
    bundle: dict[str, str] | None = None,
) -> None:
    """Install a bundle when provided; otherwise no-op for local runs."""
    if bundle:
        install_user_code_bundle(bundle)


def bundle_payload_for_transport(config: FineTuneConfig) -> dict[str, Any]:
    """Build a JSON-serializable payload for remote providers (e.g. Modal)."""
    return dict(build_user_code_bundle(config))
