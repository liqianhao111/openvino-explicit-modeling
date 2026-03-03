from __future__ import annotations

import argparse
import filecmp
import shutil
from dataclasses import dataclass
from pathlib import Path

PREFERRED_CONFIGS = ("RelWithDebInfo", "Release")
TARGET_SUFFIXES = {".dll", ".exe"}


@dataclass(frozen=True)
class CopySource:
    name: str
    relative_path_template: str
    source_kind: str
    pattern: str | None = None
    config_sensitive: bool = False

    def resolve(self, repo_root: Path, config: str) -> Path:
        return repo_root / Path(self.relative_path_template.format(config=config))


@dataclass(frozen=True)
class ConfigProbe:
    config: str
    available_sources: int
    total_sources: int
    issues: tuple[str, ...]


COPY_SOURCES = (
    CopySource(
        name="OpenVINO runtime DLL directory",
        relative_path_template="openvino/bin/intel64/{config}",
        source_kind="directory",
        pattern="*.dll",
        config_sensitive=True,
    ),
    CopySource(
        name="OpenVINO GenAI DLL directory",
        relative_path_template="openvino.genai/build/openvino_genai",
        source_kind="directory",
        pattern="*.dll",
    ),
    CopySource(
        name="Modeling API test executable",
        relative_path_template="openvino.genai/build/src/cpp/src/modeling/{config}/test_modeling_api.exe",
        source_kind="file",
        config_sensitive=True,
    ),
    CopySource(
        name="Modeling samples executable directory",
        relative_path_template="openvino.genai/build/src/cpp/src/modeling/samples/{config}",
        source_kind="directory",
        pattern="*.exe",
        config_sensitive=True,
    ),
    CopySource(
        name="Text generation sample executable",
        relative_path_template="openvino.genai/build/samples/cpp/text_generation/{config}/greedy_causal_lm.exe",
        source_kind="file",
        config_sensitive=True,
    ),
    CopySource(
        name="TBB runtime DLL",
        relative_path_template="openvino/temp/Windows_AMD64/tbb/bin/tbb12.dll",
        source_kind="file",
    ),
)


def log(level: str, message: str) -> None:
    print(f"[{level}] {message}")


def format_bytes(size_in_bytes: int) -> str:
    size = float(size_in_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size_in_bytes} B"


def collect_source_files(source: CopySource, repo_root: Path, config: str) -> tuple[list[Path], list[str]]:
    resolved_path = source.resolve(repo_root, config)

    if source.source_kind == "file":
        if not resolved_path.exists():
            return [], [
                f"{source.name}: expected file was not found: {resolved_path}"
            ]
        if not resolved_path.is_file():
            return [], [
                f"{source.name}: expected a file, but found a non-file path: {resolved_path}"
            ]
        if resolved_path.suffix.lower() not in TARGET_SUFFIXES:
            return [], [
                f"{source.name}: file does not match expected dll/exe type: {resolved_path}"
            ]
        return [resolved_path], []

    if source.source_kind != "directory":
        return [], [f"{source.name}: unsupported source kind '{source.source_kind}'."]

    if not resolved_path.exists():
        return [], [
            f"{source.name}: expected directory was not found: {resolved_path}"
        ]
    if not resolved_path.is_dir():
        return [], [
            f"{source.name}: expected a directory, but found a non-directory path: {resolved_path}"
        ]

    pattern = source.pattern or "*"
    matched_files = sorted(
        file
        for file in resolved_path.glob(pattern)
        if file.is_file() and file.suffix.lower() in TARGET_SUFFIXES
    )
    if not matched_files:
        return [], [
            f"{source.name}: directory exists but no files matched '{pattern}': {resolved_path}"
        ]

    return matched_files, []


def probe_config(repo_root: Path, config: str) -> ConfigProbe:
    issues: list[str] = []
    available_sources = 0
    total_sources = 0

    for source in COPY_SOURCES:
        if not source.config_sensitive:
            continue
        total_sources += 1
        files, source_issues = collect_source_files(source, repo_root, config)
        if source_issues:
            issues.extend(source_issues)
            continue
        if files:
            available_sources += 1

    return ConfigProbe(
        config=config,
        available_sources=available_sources,
        total_sources=total_sources,
        issues=tuple(issues),
    )


def choose_config(repo_root: Path) -> tuple[str, tuple[ConfigProbe, ...], str]:
    probes = tuple(probe_config(repo_root, config) for config in PREFERRED_CONFIGS)
    rel_probe = probes[0]
    release_probe = probes[1]

    if rel_probe.available_sources == rel_probe.total_sources:
        return rel_probe.config, probes, "RelWithDebInfo is fully available."

    if release_probe.available_sources == release_probe.total_sources:
        return (
            release_probe.config,
            probes,
            "RelWithDebInfo is incomplete, falling back to Release because all config-sensitive sources exist there.",
        )

    if rel_probe.available_sources == 0 and release_probe.available_sources > 0:
        return (
            release_probe.config,
            probes,
            "RelWithDebInfo was not found, falling back to Release.",
        )

    if release_probe.available_sources > rel_probe.available_sources:
        return (
            release_probe.config,
            probes,
            "Neither configuration is complete, but Release has more available sources.",
        )

    return (
        rel_probe.config,
        probes,
        "Neither configuration is complete, keeping preferred RelWithDebInfo and reporting missing paths.",
    )


def copy_one_file(source_file: Path, destination_dir: Path) -> tuple[str, int]:
    destination_file = destination_dir / source_file.name
    file_size = source_file.stat().st_size

    if destination_file.exists():
        if not destination_file.is_file():
            raise RuntimeError(
                f"Destination exists but is not a file: {destination_file}"
            )

        if filecmp.cmp(source_file, destination_file, shallow=False):
            log(
                "SKIP",
                f"Identical file already exists, skipping: {source_file} -> {destination_file} ({format_bytes(file_size)})",
            )
            return "skipped", 0

        log(
            "COPY",
            f"Overwriting existing file: {source_file} -> {destination_file} ({format_bytes(file_size)})",
        )
        shutil.copy2(source_file, destination_file)
        return "overwritten", file_size

    log(
        "COPY",
        f"Copying file: {source_file} -> {destination_file} ({format_bytes(file_size)})",
    )
    shutil.copy2(source_file, destination_file)
    return "copied", file_size


def collect_package_files(destination_dir: Path) -> list[Path]:
    if not destination_dir.exists():
        return []

    return sorted(
        file
        for file in destination_dir.iterdir()
        if file.is_file() and file.suffix.lower() in TARGET_SUFFIXES
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect built DLL/EXE artifacts from the OpenVINO and OpenVINO GenAI "
            "workspace into a single package directory."
        ),
        epilog=(
            "Default behavior:\n"
            "  - Prefer RelWithDebInfo, fall back to Release when RelWithDebInfo is incomplete.\n"
            "  - Use the workspace-level 'package' directory as the output root.\n"
            "  - Store files in '<output_root>\\\\<Config>'.\n"
            "  - Keep existing files unless --clean is specified.\n\n"
            "Examples:\n"
            "  python scripts\\package.py\n"
            "  python scripts\\package.py --clean\n"
            "  python scripts\\package.py --output D:\\artifacts\\package_bundle\n"
            "  python scripts\\package.py --output custom_package --clean"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Delete existing DLL/EXE files in the final destination directory before copying. "
            "Only files directly inside '<output_root>\\<Config>' are removed."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Override the package output root directory. If omitted, the default is "
            "the workspace-level 'package' directory. Relative paths are resolved "
            "relative to the workspace root (two levels above this script)."
        ),
    )
    return parser


def resolve_output_root(output_arg: str | None, workspace_root: Path) -> Path:
    if not output_arg:
        return workspace_root / "package"

    output_path = Path(output_arg)
    if not output_path.is_absolute():
        output_path = workspace_root / output_path
    return output_path.resolve()


def clean_destination_dir(destination_dir: Path) -> tuple[int, int]:
    if not destination_dir.exists():
        log("INFO", f"Clean requested but destination does not exist yet: {destination_dir}")
        return 0, 0

    removed_files = 0
    removed_bytes = 0

    for file in collect_package_files(destination_dir):
        file_size = file.stat().st_size
        log("CLEAN", f"Removing existing package file: {file} ({format_bytes(file_size)})")
        file.unlink()
        removed_files += 1
        removed_bytes += file_size

    return removed_files, removed_bytes


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    workspace_root = repo_root.parent
    package_root = resolve_output_root(args.output, workspace_root)

    log("INFO", f"Script directory: {script_dir}")
    log("INFO", f"Repository root: {repo_root}")
    log("INFO", f"Workspace root: {workspace_root}")
    log("INFO", f"Configured package root: {package_root}")

    chosen_config, probes, reason = choose_config(repo_root)
    for probe in probes:
        log(
            "INFO",
            f"Config probe {probe.config}: {probe.available_sources}/{probe.total_sources} config-sensitive sources available.",
        )
        for issue in probe.issues:
            log("WARN", issue)

    log("INFO", f"Selected configuration: {chosen_config}")
    log("INFO", reason)

    destination_dir = package_root / chosen_config
    destination_dir.mkdir(parents=True, exist_ok=True)
    log("INFO", f"Package output directory: {destination_dir}")

    summary = {
        "matched_files": 0,
        "copied_files": 0,
        "overwritten_files": 0,
        "skipped_files": 0,
        "error_count": 0,
        "matched_bytes": 0,
        "written_bytes": 0,
        "cleaned_files": 0,
        "cleaned_bytes": 0,
    }

    if args.clean:
        cleaned_files, cleaned_bytes = clean_destination_dir(destination_dir)
        summary["cleaned_files"] = cleaned_files
        summary["cleaned_bytes"] = cleaned_bytes
        log(
            "INFO",
            f"Clean completed. Removed {cleaned_files} file(s), reclaimed {format_bytes(cleaned_bytes)}.",
        )

    for source in COPY_SOURCES:
        resolved_path = source.resolve(repo_root, chosen_config)
        log("INFO", f"Scanning source '{source.name}': {resolved_path}")
        source_files, source_issues = collect_source_files(source, repo_root, chosen_config)

        if source_issues:
            summary["error_count"] += len(source_issues)
            for issue in source_issues:
                log("ERROR", issue)
            continue

        log(
            "INFO",
            f"Found {len(source_files)} file(s) from '{source.name}'.",
        )

        for source_file in source_files:
            summary["matched_files"] += 1
            file_size = source_file.stat().st_size
            summary["matched_bytes"] += file_size

            try:
                action, written_bytes = copy_one_file(source_file, destination_dir)
            except Exception as error:  # pragma: no cover - defensive logging
                summary["error_count"] += 1
                log(
                    "ERROR",
                    f"Failed to copy file '{source_file}' to '{destination_dir}': {error}",
                )
                continue

            if action == "copied":
                summary["copied_files"] += 1
                summary["written_bytes"] += written_bytes
            elif action == "overwritten":
                summary["overwritten_files"] += 1
                summary["written_bytes"] += written_bytes
            else:
                summary["skipped_files"] += 1

    packaged_files = collect_package_files(destination_dir)
    packaged_bytes = sum(file.stat().st_size for file in packaged_files)

    log("SUMMARY", "Packaging finished.")
    log("SUMMARY", f"Configuration: {chosen_config}")
    log("SUMMARY", f"Destination: {destination_dir}")
    log(
        "SUMMARY",
        f"Clean option: {'enabled' if args.clean else 'disabled'}; cleaned files: {summary['cleaned_files']}; cleaned size: {format_bytes(summary['cleaned_bytes'])}",
    )
    log("SUMMARY", f"Matched source files: {summary['matched_files']}")
    log(
        "SUMMARY",
        f"Copied files: {summary['copied_files']}, overwritten files: {summary['overwritten_files']}, skipped identical files: {summary['skipped_files']}",
    )
    log("SUMMARY", f"Encountered errors: {summary['error_count']}")
    log(
        "SUMMARY",
        f"Total matched source size: {format_bytes(summary['matched_bytes'])}",
    )
    log(
        "SUMMARY",
        f"Total bytes written this run: {format_bytes(summary['written_bytes'])}",
    )
    log(
        "SUMMARY",
        f"Current package folder contains {len(packaged_files)} dll/exe file(s), total size {format_bytes(packaged_bytes)}.",
    )

    return 1 if summary["error_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
