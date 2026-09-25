#!/usr/bin/env python3
"""
Render a ParaView animation from VTU data without opening the GUI.

Usage examples:
  python scripts/render_paraview_animation.py \
    --input "examples/output/advection_torus/solution_*.vtu" \
    --output examples/output/advection_torus/animation.mp4 \
    --field h

  python scripts/render_paraview_animation.py \
    --input "examples/output/advection_cubed_sphere/solution_*.vtu" \
    --output "examples/output/advection_cubed_sphere/plot_*.png" \
    --field h

  python scripts/render_paraview_animation.py \
    --input "examples/output/advection_meshed_sphere/solution_*.vtu" \
    --output examples/output/advection_meshed_sphere/animation_plot.mp4 \
    --field u1 \
    --legend-variable '$p$' \
    --hide-orientation-axes \
    --color-min -0.4 \
    --color-max 0.4
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List


MOVIE_EXTS = {".mp4", ".avi", ".ogv", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def normalize_colormap_name(name: str) -> str:
    aliases = {
        "cool-to-warm (extended)": "Cool to Warm (Extended)",
    }
    return aliases.get(name.strip().lower(), name)


def find_pvbatch() -> str | None:
    env_override = os.environ.get("PVBATCH")
    if env_override:
        return env_override

    on_path = shutil.which("pvbatch")
    if on_path:
        return on_path

    # Common macOS app bundle install locations.
    app_globs = [
        "/Applications/ParaView*.app/Contents/bin/pvbatch",
        "/Applications/paraview*.app/Contents/bin/pvbatch",
        str(Path.home() / "Applications" / "ParaView*.app" / "Contents/bin/pvbatch"),
        str(Path.home() / "Applications" / "paraview*.app" / "Contents/bin/pvbatch"),
    ]
    candidates: List[str] = []
    for pattern in app_globs:
        candidates.extend(glob.glob(pattern))
    for candidate in sorted(candidates):
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate

    return None


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headless ParaView animation renderer for VTU time series."
    )
    parser.add_argument(
        "--input",
        required=True,
        help=(
            "Input source: VTU glob or directory containing VTU files. "
            "Files ending with '_celldata.vtu' are ignored."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Output animation path, e.g. animation.mp4, animation.avi, or frames.png. "
            "For image output, ParaView writes a frame sequence. You can pass "
            "a quoted pattern like 'plot_*.png'."
        ),
    )
    parser.add_argument(
        "--field",
        default="h",
        help="Scalar field for coloring (default: h).",
    )
    parser.add_argument(
        "--association",
        default="POINTS",
        choices=["POINTS", "CELLS"],
        help="Data association for --field (default: POINTS).",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Animation framerate (default: 20).",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Render width in pixels (default: 1280).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Render height in pixels (default: 1024).",
    )
    parser.add_argument(
        "--colormap",
        default="Cool-to-warm (extended)",
        help="ParaView colormap preset name (default: 'Cool-to-warm (extended)').",
    )
    parser.add_argument(
        "--representation",
        default="Surface With Edges",
        choices=["Surface", "Surface With Edges", "Wireframe", "Points"],
        help="Geometry representation (default: Surface With Edges).",
    )
    parser.add_argument(
        "--show-time",
        action="store_true",
        help="Overlay simulation time text.",
    )
    parser.add_argument(
        "--time-format",
        default="t = %.5f",
        help="Annotate-time format string (default: 't = %%.5f').",
    )
    parser.add_argument(
        "--parallel-projection",
        action="store_true",
        help="Enable parallel projection camera.",
    )
    parser.add_argument(
        "--camera",
        default="isometric",
        choices=["reset", "xy", "isometric"],
        help="Camera preset (default: isometric).",
    )
    parser.add_argument(
        "--hide-orientation-axes",
        action="store_true",
        help="Hide the XYZ orientation axes symbol (default: shown).",
    )
    parser.add_argument(
        "--color-min",
        type=float,
        default=0.0,
        help="Lower bound for color range (default: 0.0).",
    )
    parser.add_argument(
        "--color-max",
        type=float,
        default=1.0,
        help="Upper bound for color range (default: 1.0).",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=11,
        help="Use a discretized colormap with this many bins (default: 11).",
    )
    parser.add_argument(
        "--font-size",
        type=int,
        default=32,
        help="Font size for scalar bar and time annotation text (default: 32).",
    )
    parser.add_argument(
        "--scalar-bar-title",
        default="",
        help="Override color-bar title text (default: no title).",
    )
    parser.add_argument(
        "--legend-variable",
        default=None,
        help=(
            "Override legend variable text (supports math mode, e.g. '$p$'). "
            "Takes precedence over --scalar-bar-title."
        ),
    )
    parser.add_argument(
        "--scalar-bar-component-title",
        default=None,
        help=(
            "Override color-bar component title text. If omitted and "
            "--scalar-bar-title is set, the component title is hidden."
        ),
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="First frame index to write (default: first available).",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Last frame index to write (default: last available).",
    )
    args = parser.parse_args(argv)
    if args.bins is not None and args.bins < 2:
        parser.error("--bins must be >= 2.")
    return args


def reexec_with_pvbatch_if_needed(argv: List[str]) -> None:
    try:
        import paraview.simple  # noqa: F401

        return
    except Exception:
        pass

    pvbatch = find_pvbatch()
    if pvbatch is None:
        raise SystemExit(
            "Could not find 'pvbatch'. Install ParaView and ensure pvbatch is on PATH, "
            "or set PVBATCH=/path/to/pvbatch."
        )

    cmd = [pvbatch, __file__, "--"] + argv
    print("Re-executing via pvbatch:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)
    raise SystemExit(0)


def resolve_input(input_arg: str) -> List[str]:
    path = Path(input_arg)

    if path.is_dir():
        return collect_vtu_files(str(path / "*.vtu"))

    return collect_vtu_files(input_arg)


def collect_vtu_files(pattern: str) -> List[str]:
    files = sorted(glob.glob(pattern))
    files = [f for f in files if f.endswith(".vtu") and not f.endswith("_celldata.vtu")]
    if not files:
        raise SystemExit(f"No VTU files found for pattern: {pattern}")
    return files


def normalize_output_path(output_path: Path) -> Path:
    if output_path.suffix.lower() in IMAGE_EXTS and "*" in output_path.name:
        return output_path.with_name(output_path.name.replace("*", ""))
    return output_path


def maybe_setattr(obj, attr: str, value) -> None:
    try:
        setattr(obj, attr, value)
    except Exception:
        pass


def configure_camera(view, camera_mode: str, parallel_projection: bool) -> None:
    view.ResetCamera()

    if camera_mode == "xy":
        view.InteractionMode = "2D"
        view.CameraPosition = [0.0, 0.0, 1.0]
        view.CameraFocalPoint = [0.0, 0.0, 0.0]
        view.CameraViewUp = [0.0, 1.0, 0.0]
    elif camera_mode == "isometric":
        # ParaView default-ish isometric view direction.
        view.CameraPosition = [1.0, 1.0, 1.0]
        view.CameraFocalPoint = [0.0, 0.0, 0.0]
        view.CameraViewUp = [0.0, 0.0, 1.0]
        view.ResetCamera()

    if parallel_projection:
        view.CameraParallelProjection = 1


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    reexec_with_pvbatch_if_needed(argv)

    from paraview.simple import (  # type: ignore
        AssignViewToLayout,
        AnnotateTimeFilter,
        ColorBy,
        CreateLayout,
        CreateView,
        GetAnimationScene,
        GetColorTransferFunction,
        GetScalarBar,
        GetOpacityTransferFunction,
        SaveAnimation,
        SetActiveView,
        Show,
        XMLUnstructuredGridReader,
    )

    source_data = resolve_input(args.input)
    reader = XMLUnstructuredGridReader(FileName=source_data)

    view = CreateView("RenderView")
    view.ViewSize = [args.width, args.height]
    view.Background = [1.0, 1.0, 1.0]
    maybe_setattr(view, "OrientationAxesVisibility", 0 if args.hide_orientation_axes else 1)
    SetActiveView(view)
    layout = CreateLayout("Layout")
    AssignViewToLayout(view=view, layout=layout, hint=0)
    layout.SetSize(args.width, args.height)

    display = Show(reader, view)
    display.Representation = args.representation

    ColorBy(display, (args.association, args.field))
    display.SetScalarBarVisibility(view, True)

    lut = GetColorTransferFunction(args.field)
    opacity = GetOpacityTransferFunction(args.field)
    preset_name = normalize_colormap_name(args.colormap)
    try:
        lut.ApplyPreset(preset_name, True)
    except Exception:
        print(f"Warning: colormap preset '{args.colormap}' not found; using default.")

    lut.RescaleTransferFunction(args.color_min, args.color_max)
    opacity.RescaleTransferFunction(args.color_min, args.color_max)

    if args.bins is not None:
        maybe_setattr(lut, "Discretize", 1)
        maybe_setattr(lut, "NumberOfTableValues", args.bins)

    scalar_bar = GetScalarBar(lut, view)
    scalar_bar_title = args.legend_variable
    if scalar_bar_title is None:
        scalar_bar_title = args.scalar_bar_title
    if scalar_bar_title is not None:
        scalar_bar.Title = scalar_bar_title
    if args.scalar_bar_component_title is not None:
        scalar_bar.ComponentTitle = args.scalar_bar_component_title
    elif scalar_bar_title is not None:
        scalar_bar.ComponentTitle = ""
    # Default to one-decimal labels and show a midpoint marker on the scalar bar.
    for attr, value in (
        ("AddRangeLabels", 0),
        ("DrawTickLabels", 1),
        ("DrawSubTickMarks", 0),
        ("AutomaticLabelFormat", 0),
    ):
        maybe_setattr(scalar_bar, attr, value)
    midpoint = 0.5 * (args.color_min + args.color_max)
    maybe_setattr(scalar_bar, "UseCustomLabels", 1)
    maybe_setattr(scalar_bar, "CustomLabels", [args.color_min, midpoint, args.color_max])
    for attr in ("RangeLabelFormat", "LabelFormat"):
        maybe_setattr(scalar_bar, attr, "%.1f")
    for attr in ("TitleFontSize", "LabelFontSize"):
        maybe_setattr(scalar_bar, attr, args.font_size)

    # ParaView property names vary by version; keep transfer functions fixed
    # after explicit rescaling when those properties are available.
    for tf in (lut, opacity):
        maybe_setattr(tf, "AutomaticRescaleRangeMode", "Never")

    if args.show_time:
        time_text = AnnotateTimeFilter(Input=reader)
        time_text.Format = args.time_format
        time_display = Show(time_text, view)
        maybe_setattr(time_display, "FontSize", args.font_size)

    configure_camera(view, args.camera, args.parallel_projection)

    animation_scene = GetAnimationScene()
    animation_scene.UpdateAnimationUsingDataTimeSteps()

    save_kwargs = {
        "ImageResolution": [args.width, args.height],
        "FrameRate": args.fps,
    }

    if args.start_frame is not None or args.end_frame is not None:
        start = 0 if args.start_frame is None else args.start_frame
        end = -1 if args.end_frame is None else args.end_frame
        save_kwargs["FrameWindow"] = [start, end]

    output_path = normalize_output_path(Path(args.output))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    is_movie = output_path.suffix.lower() in MOVIE_EXTS
    save_kwargs["SaveAllViews"] = 1

    try:
        SaveAnimation(str(output_path), layout, **save_kwargs)
    except Exception as exc:
        if not is_movie:
            raise
        print(f"Direct movie export failed ({exc}). Falling back to PNG frames + ffmpeg.")
        render_with_ffmpeg_fallback(SaveAnimation, layout, output_path, save_kwargs, args.fps)
    if is_movie and not output_path.exists():
        print("Direct movie export did not create a file. Falling back to PNG frames + ffmpeg.")
        render_with_ffmpeg_fallback(SaveAnimation, layout, output_path, save_kwargs, args.fps)

    print(f"Saved animation: {output_path}")


def render_with_ffmpeg_fallback(save_animation, layout, output_path: Path, save_kwargs, fps: int):
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "Direct movie export is unavailable and ffmpeg was not found on PATH."
        )

    with tempfile.TemporaryDirectory(prefix="paraview_frames_") as tmpdir:
        frame_seed = Path(tmpdir) / "frame.png"
        frame_kwargs = dict(save_kwargs)
        frame_kwargs.pop("FrameRate", None)
        save_animation(str(frame_seed), layout, **frame_kwargs)

        frames = sorted(Path(tmpdir).glob("frame*.png"))
        if not frames:
            frames = sorted(Path(tmpdir).glob("*.png"))
        if not frames:
            raise RuntimeError("ParaView did not produce PNG frames for ffmpeg fallback.")

        pattern = str(Path(tmpdir) / "frame*.png")
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-pattern_type",
            "glob",
            "-i",
            pattern,
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    forwarded_argv = sys.argv[1:]
    if forwarded_argv and forwarded_argv[0] == "--":
        forwarded_argv = forwarded_argv[1:]
    main(forwarded_argv)
