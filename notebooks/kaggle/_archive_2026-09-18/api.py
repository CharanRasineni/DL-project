"""Kaggle source resolution and kernel utilities for the FathomNet CLEF project.

Real data flow (on the Kaggle VM):
 * Split 02 writes /kaggle/input/fathomnet-02-split/split.json
 * Each training kernel 03..09 has kernel-sources.yaml listing the input
   source slugs it needs, keyed by a local alias like "split", "data".
 * A small resolver reads kernel-sources.yaml + kernel-metadata.json and
 *disambiguates* mounted-from-set vs mounted-from-competition vs
 *mounted-from-current, returning the on-VM paths the notebook can actually
 *use.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Tuple

HERE = Path(__file__).resolve().parent
WORK = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()


def resolve_sources(
    notebook_dir: Path,
    *,
    dry_run: bool = False,
) -> Dict[str, Path]:
    """Return {alias: on-VM Path} for a kernel's declared sources.

    Reads `kernel-sources.yaml` (if present) and falls back to the
    competition / current / set heuristics used by the local runner.
    """
    meta = read_kernel_metadata(notebook_dir)
    sources_yaml = notebook_dir / "kernel-sources.yaml"
    result: Dict[str, Path] = {}

    if sources_yaml.exists():
        spec = yaml_read(sources_yaml)
        for alias, info in spec.items():
            result[alias] = _resolve_one(info, meta, dry_run=dry_run)
    else:
        # Heuristic mode — one input directory per alias
        for alias in (meta.get("kernel_sources", []) or []):
            slug = alias.get("slug") if isinstance(alias, dict) else ""
            key = alias.get("source") if isinstance(alias, dict) else alias
            if key is None:
                key = slug or alias
            result[alias if isinstance(alias, str) else slug or alias] = _resolve_one(
                {"source": key, "slug": slug or alias}, meta, dry_run=dry_run
            )
    return result


def _resolve_one(info: Dict[str, Any], meta: Dict[str, Any], dry_run: bool = False) -> Path:
    source = info.get("source", "")
    slug = info.get("slug", "")

    # 1) competition input mount
    if source in ("competition", None):
        comp = meta.get("competition") or os.environ.get("KAGGLE_COMPETITION", "")
        if comp:
            p = WORK.parent / "input" / comp
        else:
            p = WORK.parent / "input" / slug if slug else WORK.parent / "input"
    # 2) current kernel's own outputs (re-run artefacts)
    elif source == "current":
        p = WORK / slug if slug else WORK
    # 3) a specific set / public dataset slug
    else:
        p = WORK.parent / "input" / slug if slug else WORK.parent / "input" / source

    if dry_run:
        return p
    return p


def read_kernel_metadata(notebook_dir: Path) -> Dict[str, Any]:
    mf = notebook_dir / "kernel-metadata.json"
    if mf.exists():
        return json.loads(mf.read_text(encoding="utf-8"))
    return {}


def yaml_read(path: Path) -> Dict[str, Any]:
    """Tiny YAML subset reader for kernel-sources.yaml style maps.

    Only handles:
        alias:
            source: <kind>
            slug: <slug>
        alias: <kind>
    """
    out: Dict[str, Any] = {}
    current: Any = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith(" ") and line.endswith(":"):
            key = line[:-1].strip()
            out[key] = {}
            current = out[key]
        elif line.startswith(" ") and current is not None and ": " in line:
            k, v = line.strip().split(": ", 1)
            current[k] = v
        elif line.startswith(" ") and current is not None and line.endswith(":"):
            k = line.strip()[:-1]
            current[k] = {}
            current = current[k]
    return out


def list_input_datasets() -> Dict[str, Path]:
    """Enumerate /kaggle/input/* directories and their slugs.

    Returns {slug_or_dirname: path}.
    """
    inp = WORK.parent / "input"
    if not inp.is_dir():
        return {}
    out: Dict[str, Path] = {}
    for child in sorted(inp.iterdir()):
        if child.is_dir():
            out[child.name] = child
    return out


def copy_to_working(src: Path, dst: Path, *, follow_links: bool = True) -> Path:
    """Copy or symlink `src` tree into /kaggle/working, preserving structure.

    Returns the destination root.
    """
    dst = WORK / dst.relative_to(src.parent) if src.parent in dst.parents else WORK / dst.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_symlink() and follow_links:
        src = src.resolve()
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, symlinks=False)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return dst


def make_symlinks(source_dir: Path, target_dir: Path, *, link_type: str = "symlink") -> Path:
    """Create links from source_dir into target_dir, mirroring structure."""
    target_dir.mkdir(parents=True, exist_ok=True)
    for child in source_dir.iterdir():
        tgt = target_dir / child.name
        if tgt.exists() or tgt.is_symlink():
            tgt.unlink()
        if link_type == "symlink":
            tgt.symlink_to(child.resolve() if child.is_symlink() else child)
        else:
            if child.is_dir():
                shutil.copytree(child, tgt, symlinks=False)
            else:
                shutil.copy2(child, tgt)
    return target_dir
