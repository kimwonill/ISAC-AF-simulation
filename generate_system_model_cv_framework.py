"""Recreate the paper's system-model diagram with Matplotlib.

The source SVG is deliberately treated as a compact geometry specification: every
visible SVG primitive is converted to a Matplotlib artist and Matplotlib writes the
new SVG/PDF/PNG files.  This keeps the diagram editable as vector graphics while
preserving the coordinates, colours, labels, and layer order of the approved figure.
"""

from __future__ import annotations

import argparse
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib import patches
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.transforms import Affine2D


SVG_NS = "{http://www.w3.org/2000/svg}"
VIEWBOX = (51.0, 0.0, 1257.0, 454.0)
SVG_DPI = 96.0
PX_TO_PT = 72.0 / SVG_DPI

CLASS_STYLE = {
    "base": {"color": "#27313a", "fontfamily": "Times New Roman"},
    "panel-title": {"fontsize": 31, "fontweight": "medium"},
    "section-title": {"fontsize": 25, "fontweight": "medium"},
    "lane-title": {"fontsize": 23, "fontweight": "medium"},
    "label": {"fontsize": 23},
    "small": {"fontsize": 20},
    "tiny": {"fontsize": 18},
    "micro": {"fontsize": 11},
    "beam-title": {"fontsize": 17},
    "rx-title": {"fontsize": 19},
    "rx-detail": {"fontsize": 15},
    "rf-label": {"fontsize": 16},
    "sig-detail": {"fontsize": 14},
    "block-text": {"fontsize": 18},
    "axis-label": {"fontsize": 16, "fontweight": "normal"},
    "halo-text": {"halo_width": 5},
    "matrix": {"fontweight": "bold"},
    "vector": {"fontweight": "bold"},
    "box": {"edgecolor": "#6f7b84", "linewidth": 2, "facecolor": "#ffffff"},
    "lane": {"edgecolor": "#98a3ab", "linewidth": 2, "linestyle": (0, (8, 6)), "facecolor": "#f5f7f8"},
    "teal-box": {"edgecolor": "#0f8f7c", "linewidth": 2.5, "facecolor": "#eefaf7"},
    "blue-box": {"edgecolor": "#2b78bb", "linewidth": 2, "facecolor": "#eef6fd"},
    "red-box": {"edgecolor": "#d83b30", "linewidth": 2, "facecolor": "#fff1ef"},
    "gold-box": {"edgecolor": "#c78918", "linewidth": 2, "facecolor": "#fff8df"},
    "purple-box": {"edgecolor": "#785298", "linewidth": 2, "facecolor": "#f6effa"},
    "gray-box": {"edgecolor": "#747d85", "linewidth": 2.25, "facecolor": "#f1f3f4"},
    "blue-line": {"edgecolor": "#2b78bb", "linewidth": 4, "fill": False, "arrow": True},
    "red-line": {"edgecolor": "#d83b30", "linewidth": 4, "fill": False, "arrow": True},
    "echo-line": {
        "edgecolor": "#d83b30", "linewidth": 3.5,
        "linestyle": (0, (3, 2)), "fill": False, "arrow": True,
        "separate_arrowhead": True,
    },
    "teal-line": {"edgecolor": "#0f8f7c", "linewidth": 4, "fill": False, "arrow": True},
    "gray-line": {"edgecolor": "#000000", "linewidth": 3, "fill": False, "arrow": True},
    "thin": {"linewidth": 2},
}

COLOUR_KEYS = {"stroke": "edgecolor", "fill": "facecolor"}
NUMBER = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
PATH_TOKEN = re.compile(r"[MmLlHhVvCcZz]|[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?")
TRANSFORM = re.compile(r"([A-Za-z]+)\s*\(([^)]*)\)")


def floats(value: str) -> list[float]:
    return [float(item) for item in NUMBER.findall(value)]


def svg_transform(value: str | None) -> Affine2D:
    result = Affine2D()
    if not value:
        return result
    # SVG applies transform functions from right to left. Affine2D mutator
    # methods post-compose in call order, so process the SVG list in reverse.
    for name, raw_args in reversed(TRANSFORM.findall(value)):
        args = floats(raw_args)
        if name == "translate":
            result.translate(args[0], args[1] if len(args) > 1 else 0)
        elif name == "scale":
            result.scale(args[0], args[1] if len(args) > 1 else args[0])
        elif name == "rotate":
            angle = args[0]
            if len(args) == 3:
                result.translate(-args[1], -args[2]).rotate_deg(angle).translate(args[1], args[2])
            else:
                result.rotate_deg(angle)
        elif name == "matrix":
            result += Affine2D.from_values(*args)
        else:
            raise ValueError(f"Unsupported SVG transform: {name}")
    return result


def merged_style(element: ET.Element) -> dict:
    style: dict = {}
    for class_name in element.get("class", "").split():
        style.update(CLASS_STYLE.get(class_name, {}))
    for svg_key, mpl_key in COLOUR_KEYS.items():
        if svg_key in element.attrib:
            style[mpl_key] = element.attrib[svg_key]
    if "stroke-width" in element.attrib:
        style["linewidth"] = float(element.get("stroke-width", "1"))
    if "stroke-dasharray" in element.attrib:
        vals = floats(element.get("stroke-dasharray", ""))
        style["linestyle"] = (0, tuple(vals))
    if element.get("fill") == "none":
        style["fill"] = False
        style.pop("facecolor", None)
    return style


def mpl_patch_style(style: dict) -> dict:
    allowed = {"edgecolor", "facecolor", "linewidth", "linestyle", "fill"}
    result = {key: value for key, value in style.items() if key in allowed}
    if "linewidth" in result:
        result["linewidth"] *= PX_TO_PT
    if result.get("facecolor") == "none":
        result["facecolor"] = "none"
    return result


def parse_path(data: str) -> MplPath:
    tokens = PATH_TOKEN.findall(data.replace(",", " "))
    vertices: list[tuple[float, float]] = []
    codes: list[int] = []
    index = 0
    command = ""
    current = (0.0, 0.0)
    start = current

    def number() -> float:
        nonlocal index
        value = float(tokens[index])
        index += 1
        return value

    while index < len(tokens):
        if tokens[index].isalpha():
            command = tokens[index]
            index += 1
        relative = command.islower()
        op = command.upper()
        if op == "Z":
            vertices.append(start)
            codes.append(MplPath.CLOSEPOLY)
            current = start
            command = ""
            continue
        if op == "M" or op == "L":
            x, y = number(), number()
            if relative:
                x, y = x + current[0], y + current[1]
            current = (x, y)
            vertices.append(current)
            codes.append(MplPath.MOVETO if op == "M" else MplPath.LINETO)
            if op == "M":
                start = current
                command = "l" if relative else "L"
        elif op == "H":
            x = number() + (current[0] if relative else 0)
            current = (x, current[1])
            vertices.append(current)
            codes.append(MplPath.LINETO)
        elif op == "V":
            y = number() + (current[1] if relative else 0)
            current = (current[0], y)
            vertices.append(current)
            codes.append(MplPath.LINETO)
        elif op == "C":
            values = [number() for _ in range(6)]
            if relative:
                values = [value + current[i % 2] for i, value in enumerate(values)]
            vertices.extend([(values[0], values[1]), (values[2], values[3]), (values[4], values[5])])
            codes.extend([MplPath.CURVE4] * 3)
            current = (values[4], values[5])
        else:
            raise ValueError(f"Unsupported SVG path command: {command}")
    return MplPath(vertices, codes)


def rich_text(element: ET.Element) -> str:
    """Translate SVG tspans used in the figure to Matplotlib mathtext."""
    pieces: list[str] = []
    if element.text:
        pieces.append(element.text)
    for child in element:
        raw = rich_text(child)
        classes = child.get("class", "").split()
        if child.get("baseline-shift") == "sub":
            # Matplotlib mathtext italicizes variables by default. Diagram
            # indices are labels, so force every subscript to upright roman.
            raw = rf"$_{{\mathrm{{{raw}}}}}$"
        elif "vector" in classes and raw == "τ̂":
            raw = r"$\hat{\mathbf{\mathrm{τ}}}$"
        elif "vector" in classes and raw == "ν̂":
            raw = r"$\hat{\mathbf{\mathrm{ν}}}$"
        # Nested bold tspans are left as text here. Parent-level matrix/vector
        # classes are still handled through FontProperties; wrapping a nested
        # expression that already contains a subscript would create invalid
        # adjacent mathtext delimiters.
        pieces.append(raw)
        if child.tail:
            pieces.append(child.tail)
    return "".join(pieces).strip()


class SvgToMatplotlib:
    def __init__(self, ax: plt.Axes):
        self.ax = ax
        self._layer = 0

    def _next_layer(self) -> int:
        """Preserve SVG paint order across Matplotlib artist types."""
        self._layer += 1
        return self._layer

    def draw(self, element: ET.Element, parent_transform: Affine2D | None = None) -> None:
        if element.get("display") == "none":
            return
        parent_transform = parent_transform or Affine2D()
        transform = svg_transform(element.get("transform")) + parent_transform
        tag = element.tag.removeprefix(SVG_NS)
        style = merged_style(element)

        if tag in {"svg", "g"}:
            pass
        elif tag == "rect":
            x, y = float(element.get("x", 0)), float(element.get("y", 0))
            width, height = float(element.get("width", 0)), float(element.get("height", 0))
            radius = float(element.get("rx", 0))
            kwargs = mpl_patch_style(style)
            if radius:
                artist = patches.FancyBboxPatch(
                    (x, y), width, height,
                    boxstyle=patches.BoxStyle("Round", pad=0, rounding_size=radius), **kwargs,
                )
            else:
                artist = patches.Rectangle((x, y), width, height, **kwargs)
            artist.set_zorder(self._next_layer())
            artist.set_transform(transform + self.ax.transData)
            self.ax.add_patch(artist)
        elif tag == "line":
            points = [(float(element.get("x1", 0)), float(element.get("y1", 0))),
                      (float(element.get("x2", 0)), float(element.get("y2", 0)))]
            self._polyline(points, style, transform)
        elif tag == "polyline":
            values = floats(element.get("points", ""))
            self._polyline(list(zip(values[::2], values[1::2])), style, transform)
        elif tag == "circle":
            artist = patches.Circle(
                (float(element.get("cx", 0)), float(element.get("cy", 0))),
                float(element.get("r", 0)), **mpl_patch_style(style),
            )
            artist.set_zorder(self._next_layer())
            artist.set_transform(transform + self.ax.transData)
            self.ax.add_patch(artist)
        elif tag == "ellipse":
            artist = patches.Ellipse(
                (float(element.get("cx", 0)), float(element.get("cy", 0))),
                2 * float(element.get("rx", 0)), 2 * float(element.get("ry", 0)),
                **mpl_patch_style(style),
            )
            artist.set_zorder(self._next_layer())
            artist.set_transform(transform + self.ax.transData)
            self.ax.add_patch(artist)
        elif tag == "path":
            path = parse_path(element.get("d", ""))
            if style.pop("arrow", False):
                self._arrow(path, style, transform)
            else:
                artist = patches.PathPatch(path, **mpl_patch_style(style))
                artist.set_zorder(self._next_layer())
                artist.set_transform(transform + self.ax.transData)
                self.ax.add_patch(artist)
        elif tag == "text":
            self._text(element, style, transform)

        if tag != "text":
            for child in element:
                child_tag = child.tag.removeprefix(SVG_NS)
                if child_tag not in {"defs", "title", "desc", "style", "marker", "linearGradient", "stop"}:
                    self.draw(child, transform)

    def _polyline(self, points: list[tuple[float, float]], style: dict, transform: Affine2D) -> None:
        transformed = transform.transform(points)
        kwargs = mpl_patch_style(style)
        color = kwargs.pop("edgecolor", kwargs.pop("facecolor", "black"))
        linewidth = kwargs.pop("linewidth", PX_TO_PT)
        linestyle = kwargs.pop("linestyle", "-")
        self.ax.plot(transformed[:, 0], transformed[:, 1], color=color,
                     linewidth=linewidth, linestyle=linestyle, solid_capstyle="butt",
                     zorder=self._next_layer())

    def _arrow(self, path: MplPath, style: dict, transform: Affine2D) -> None:
        transformed_path = transform.transform_path(path)
        separate_arrowhead = style.pop("separate_arrowhead", False)
        kwargs = mpl_patch_style(style)
        color = kwargs.pop("edgecolor", "black")
        linewidth = kwargs.pop("linewidth", 1.0)
        linestyle = kwargs.pop("linestyle", "-")
        if separate_arrowhead:
            previous = transformed_path.vertices[-2]
            tip = transformed_path.vertices[-1]
            dx, dy = tip - previous
            length = math.hypot(dx, dy)
            ux, uy = dx / length, dy / length
            nx, ny = -uy, ux
            base_x, base_y = tip[0] - 9 * ux, tip[1] - 9 * uy
            shaft_vertices = transformed_path.vertices.copy()
            shaft_vertices[-1] = (base_x, base_y)
            shaft_path = MplPath(shaft_vertices, transformed_path.codes)
            shaft = patches.PathPatch(
                shaft_path, facecolor="none", edgecolor=color,
                linewidth=linewidth, linestyle=linestyle,
                capstyle="butt", joinstyle="miter", zorder=self._next_layer(),
            )
            self.ax.add_patch(shaft)
            arrowhead = patches.Polygon(
                [tip, (base_x + 4.5 * nx, base_y + 4.5 * ny),
                 (base_x - 4.5 * nx, base_y - 4.5 * ny)],
                closed=True, facecolor=color, edgecolor=color, linewidth=0,
                zorder=self._next_layer(),
            )
            self.ax.add_patch(arrowhead)
            return
        artist = patches.FancyArrowPatch(
            path=transformed_path, arrowstyle="-|>", mutation_scale=10 * PX_TO_PT,
            linewidth=linewidth, linestyle=linestyle, color=color,
            shrinkA=0, shrinkB=0, joinstyle="miter", capstyle="butt",
            zorder=self._next_layer(),
        )
        self.ax.add_patch(artist)

    def _text(self, element: ET.Element, style: dict, transform: Affine2D) -> None:
        x, y = float(element.get("x", 0)), float(element.get("y", 0))
        x, y = transform.transform((x, y))
        matrix = transform.get_matrix()
        svg_angle = math.degrees(math.atan2(matrix[1, 0], matrix[0, 0]))
        scale = math.hypot(matrix[0, 0], matrix[1, 0])
        color = element.get("fill", style.get("color", style.get("facecolor", "#27313a")))
        fontsize = float(style.get("fontsize", 18)) * PX_TO_PT * scale
        family = style.get("fontfamily", "Times New Roman")
        weight = style.get("fontweight", "normal")
        halo_width = float(style.get("halo_width", 0)) * PX_TO_PT
        anchor = element.get("text-anchor", "start")
        horizontal = {"start": "left", "middle": "center", "end": "right"}.get(anchor, "left")
        artist = self.ax.text(
            x, y, rich_text(element), color=color, fontsize=fontsize,
            fontproperties=FontProperties(family=family, weight=weight, size=fontsize),
            ha=horizontal, va="baseline", rotation=-svg_angle,
            rotation_mode="anchor", clip_on=False, zorder=self._next_layer(),
        )
        if halo_width:
            artist.set_path_effects([
                path_effects.Stroke(linewidth=halo_width, foreground="#ffffff", joinstyle="round"),
                path_effects.Normal(),
            ])


def build_figure(source_svg: Path) -> plt.Figure:
    mpl.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })
    x0, y0, width, height = VIEWBOX
    figure = plt.figure(figsize=(width / SVG_DPI, height / SVG_DPI), dpi=SVG_DPI, facecolor="white")
    axes = figure.add_axes([0, 0, 1, 1])
    axes.set_xlim(x0, x0 + width)
    axes.set_ylim(y0 + height, y0)
    axes.set_aspect("equal", adjustable="box")
    axes.axis("off")
    SvgToMatplotlib(axes).draw(ET.parse(source_svg).getroot())
    return figure


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    default_source = script_dir.parent / "MyPaper" / "figures" / "system_model_cv_framework.svg"
    default_output = default_source.with_name("system_model_cv_framework_matplotlib")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=default_source)
    parser.add_argument("--output", type=Path, default=default_output,
                        help="Output path without an extension")
    args = parser.parse_args()

    figure = build_figure(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".svg"), format="svg", dpi=SVG_DPI,
                   facecolor="white", metadata={"Creator": "Matplotlib"})
    figure.savefig(args.output.with_suffix(".pdf"), format="pdf", dpi=SVG_DPI,
                   facecolor="white", metadata={"Creator": "Matplotlib"})
    figure.savefig(args.output.with_suffix(".png"), format="png", dpi=SVG_DPI,
                   facecolor="white")
    plt.close(figure)
    print(f"Wrote {args.output.with_suffix('.svg')}")
    print(f"Wrote {args.output.with_suffix('.pdf')}")
    print(f"Wrote {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
