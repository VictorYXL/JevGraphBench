"""Build deterministic GitHub-compatible result graphics from manuscript display data.

Run from any directory: python assets/benchmark/v33/render.py [--check]
Only the Python standard library is required. No model calls or network access.
"""

import argparse
from html import escape
import json
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
INK = "#172b4d"
MUTED = "#52647d"
TEAL = "#087f8c"
BLUE = "#4258b8"
PURPLE = "#7850a3"
COLORS = {"native": TEAL, "generation": BLUE, "scoring": "#b46c13",
          "no_reasoning": "#245fc7", "reasoning": PURPLE}


class SVG:
    def __init__(self, height, title, description):
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="{height}" '
            f'viewBox="0 0 1120 {height}" role="img" aria-labelledby="title desc">',
            f"<title id=\"title\">{escape(title)}</title><desc id=\"desc\">{escape(description)}</desc>",
            '<g font-family="Arial, Helvetica, sans-serif">',
        ]
        self.rect(0, 0, 1120, height, "#ffffff", radius=18)
        self.rect(1, 1, 1118, height - 2, "none", stroke="#dbe4ee", radius=18)

    def rect(self, x, y, w, h, fill, radius=0, stroke=None, dash=None):
        extra = f' stroke="{stroke}"' if stroke else ""
        if dash:
            extra += f' stroke-dasharray="{dash}"'
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
                          f'rx="{radius}" fill="{fill}"{extra}/>')

    def text(self, x, y, value, size=16, color=INK, weight=400, anchor="start"):
        self.parts.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" '
                          f'font-weight="{weight}" text-anchor="{anchor}">{escape(str(value))}</text>')

    def line(self, x1, y1, x2, y2, color="#e3eaf2", width=1):
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                          f'stroke="{color}" stroke-width="{width}"/>')

    def finish(self):
        result = "\n".join([*self.parts, "</g></svg>", ""])
        ET.fromstring(result)
        return result


def mix(low, high, ratio):
    a = tuple(int(low[i:i + 2], 16) for i in (1, 3, 5))
    b = tuple(int(high[i:i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(x + (y - x) * ratio):02x}" for x, y in zip(a, b))


def yrow(index):
    return 178 + index * 37 + (10 if index >= 4 else 0) + (10 if index >= 11 else 0)


def label(svg, model, x, y, size=17):
    svg.rect(x, y - 12, 4, 16, COLORS[model["family"]], radius=2)
    svg.text(x + 13, y + 1, model["label"], size=size,
             weight=700 if model["id"] in ("jev", "gpt54", "astra") else 400)


def heatmap(data, kind):
    rq1 = kind == "rq1"
    title = "Structure is not a single skill" if rq1 else "More information is not always better"
    svg = SVG(802, title, "All fourteen configurations. Values are percentages; higher is better. "
              "Astra is a separate reasoning-enabled reference. U means unsupported.")
    svg.text(30, 37, "01 / GRAPH COGNITION" if rq1 else "02 / GRAPH + TEXT", 14, TEAL, 700)
    svg.text(30, 75, title, 30, weight=700)
    svg.text(30, 105, "800 queries  /  4 data sources  /  Equal weight per size, 8-12 vertices" if rq1
             else "100 queries per dataset  /  Same query IDs and candidates across five conditions",
             16, MUTED)
    left, width = (266, 132) if rq1 else (248, 78)
    headers = data["rq1_tasks"] if rq1 else data["rq2_conditions"] * 2
    if not rq1:
        svg.text(443, 135, "ogbn-arxiv  /  Accuracy", 16, BLUE, 700, "middle")
        svg.text(858, 135, "STaRK-Prime  /  Hit@1", 16, TEAL, 700, "middle")
    for j, heading in enumerate(headers):
        x = left + j * width + (24 if not rq1 and j >= 5 else 0)
        if not rq1 and j == 9:
            heading = "A*"
        svg.text(x + (width - 6) / 2, 155, heading, 15, MUTED, 700, "middle")
    for i, model in enumerate(data["models"]):
        y = yrow(i)
        if model["id"] == "jev":
            svg.rect(18, y - 18, 1084, 36, "#eff9f8", radius=7)
        if model["id"] == "astra":
            svg.rect(18, y - 18, 1084, 36, "#f4f0f8", radius=7)
        label(svg, model, 30, y + 5)
        values = model["rq1"] if rq1 else model["arxiv"] + model["prime"]
        for j, value in enumerate(values):
            x = left + j * width + (24 if not rq1 and j >= 5 else 0)
            color = "#eef1f5" if value is None else mix("#edf7f6", "#066c78", value / 100)
            svg.rect(x, y - 15, width - 6, 30, color, radius=5)
            value_text = "U" if value is None else (f"{value:.2f}" if rq1 else f"{value:g}")
            svg.text(x + (width - 6) / 2, y + 6, value_text, 18,
                     "#ffffff" if value is not None and value >= 60 else INK, 600, "middle")
    svg.line(30, 733, 1090, 733)
    svg.text(30, 761, "Darker = higher accuracy", 15, MUTED)
    for i in range(100):
        svg.rect(225 + i * 1.5, 748, 1.5, 13, mix("#edf7f6", "#066c78", i / 99))
    svg.text(391, 761, "0-100%", 14, MUTED)
    svg.text(30, 784, "* Astra: observed medium reasoning. GPT-5.4: explicit none. Fixed model order, not a combined rank.",
             14, MUTED)
    return svg.finish()


def optimization(data):
    svg = SVG(1420, "Good choices need good solutions",
              "All fourteen configurations across four optimization tasks. A direct, C heuristic proposals. "
              "Lower gaps are better. Each task has its own axis; dashed bars and bracketed counts mark partial coverage.")
    svg.text(30, 37, "03 / SEQUENTIAL DECISIONS", 14, TEAL, 700)
    svg.text(30, 75, "Good choices need good solutions", 30, weight=700)
    svg.text(30, 106, "Mean reference gap on completed graphs  /  Lower is better  /  Ten graphs scheduled per mode", 16, MUTED)
    svg.rect(30, 128, 23, 10, "#8796ac", radius=2)
    svg.text(62, 138, "A  Direct construction", 15, MUTED)
    svg.rect(290, 128, 23, 10, TEAL, radius=2)
    svg.text(322, 138, "C  Heuristic proposals", 15, MUTED)
    svg.text(610, 138, "Right-hand values: A / C", 15, MUTED)
    maxima = (600, 120, 100, 0.6)
    for task_index, task in enumerate(data["rq3_tasks"]):
        x = 22 + (task_index % 2) * 549
        top = 165 + (task_index // 2) * 595
        svg.rect(x, top, 527, 575, "#f9fbfd", radius=12, stroke="#e3eaf2")
        unit = "absolute Q gap" if task_index == 3 else "gap %"
        svg.text(x + 18, top + 30, task, 21, weight=700)
        svg.text(x + 18, top + 54, unit + "  |  independent task scale", 13, MUTED)
        px, pw = x + 168, 185
        maximum = maxima[task_index]
        for tick in (0, maximum / 2, maximum):
            tx = px + pw * tick / maximum
            svg.line(tx, top + 88, tx, top + 535)
            svg.text(tx, top + 79, f"{tick:g}", 12, MUTED, anchor="middle")
        svg.text(x + 442, top + 79, "A / C", 13, MUTED, 700, "middle")
        for i, model in enumerate(data["models"]):
            y = top + 102 + i * 31
            a, c, na, nc = model["rq3"][task_index]
            if model["id"] == "jev":
                svg.rect(x + 7, y - 12, 512, 30, "#e7f5f3", radius=4)
            if model["id"] == "astra":
                svg.rect(x + 7, y - 12, 512, 30, "#f0ebf7", radius=4)
            label(svg, model, x + 15, y + 6, size=12)
            for offset, value, count, color in ((-7, a, na, "#8796ac"), (5, c, nc, TEAL)):
                if value is None:
                    continue
                bar_width = value / maximum * pw
                if value == 0:
                    svg.line(px, y + offset, px, y + offset + 8, color, 2)
                else:
                    svg.rect(px, y + offset, bar_width, 8, color if count == 10 else "#ffffff",
                             radius=2, stroke=color if count < 10 else None,
                             dash="3 2" if count < 10 else None)
            def formatted(value, count):
                if value is None:
                    return "--"
                number = f"{value:.4f}" if task_index == 3 else f"{value:.2f}"
                return number if count == 10 else f"{number} [{count}]"
            svg.text(x + 511, y + 6, f"{formatted(a, na)} / {formatted(c, nc)}", 12,
                     weight=700 if model["id"] in ("jev", "gpt54", "astra") else 400, anchor="end")
        svg.text(x + 18, top + 557, "[n] = n/10 complete; -- = no completed trajectory", 12, MUTED)
    svg.text(30, 1379, "Partial subsets are not directly rankable against full coverage. Forced-only completions do not establish model support.",
             14, MUTED)
    svg.text(30, 1403, "* Astra uses reasoning; GPT-5.4 uses none. Modularity uses certified references; MaxCut uses historical BKS.", 14, MUTED)
    return svg.finish()


def hero():
    svg = SVG(314, "GraphDecide", "Benchmarking System One Models on Graph Tasks. "
              "14 model-interface configurations, 800 structural queries, 1000 graph-text decisions, 80 construction trajectories per configuration.")
    svg.rect(1, 1, 1118, 312, "#101f37", radius=18)
    for u, v in (((890, 43), (1000, 78)), ((1000, 78), (940, 145)), ((940, 145), (1070, 174)),
                 ((890, 43), (850, 128)), ((850, 128), (940, 145)), ((1000, 78), (1080, 40))):
        svg.line(*u, *v, "#285365", 2)
    for x, y in ((890, 43), (1000, 78), (940, 145), (1070, 174), (850, 128), (1080, 40)):
        svg.parts.append(f'<circle cx="{x}" cy="{y}" r="6" fill="#55cbbb"/>')
    svg.text(36, 43, "GRAPH BENCHMARK  /  V33", 14, "#77ded2", 700)
    svg.text(36, 103, "GraphDecide", 51, "#ffffff", 700)
    svg.text(36, 141, "Benchmarking System One Models on Graph Tasks", 24, "#d9e6f4")
    svg.text(36, 177, "Know the graph. Use the evidence. Build the solution.", 17, "#9aadc5")
    metrics = [("14", "model configurations"), ("800", "structural queries"),
               ("1,000", "graph-text decisions"), ("80", "construction trajectories")]
    for i, (value, caption) in enumerate(metrics):
        x = 36 + i * 268
        svg.rect(x, 211, 246, 78, "#192e49", radius=9)
        svg.text(x + 17, 246, value, 29, "#ffffff", 700)
        svg.text(x + 17, 273, caption, 14, "#b9cadf")
    return svg.finish()


def tables(data):
    text = [
        "# Complete v33 result tables", "",
        "Manuscript-transcribed display values; not fresh experiments. "
        "Sources: Appendix C (RQ1), Table 1 (RQ2), Appendix E (RQ3). "
        "[Source data](../assets/benchmark/v33/display-data.json) · [Overview](../README.md).", "",
        "GPT-5.4 uses explicit no reasoning. GPT-6-Astra uses observed medium reasoning "
        "and is not a compute-matched baseline. Fixed model order is not an overall ranking.", "",
        "## RQ1: size-equal task accuracy (%)", "",
        "| Model | " + " | ".join(data["rq1_tasks"]) + " |",
        "| --- | " + " | ".join(["---:"] * 6) + " |",
    ]
    for model in data["models"]:
        text.append("| " + model["label"] + " | " + " | ".join(f"{v:.4f}" for v in model["rq1"]) + " |")
    text += ["", "Higher is better. Accuracy is averaged equally over sizes 8–12 within each task, "
             "not pooled across queries. Degree has 400 queries; each other task has 80.", ""]
    for dataset, title in (("arxiv", "ogbn-arxiv accuracy (%)"), ("prime", "STaRK-Prime Hit@1 (%)")):
        text += [f"## RQ2: {title}", "", "| Model | T | G | TG | BAG | A" +
                 ("*" if dataset == "prime" else "") + " |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for model in data["models"]:
            text.append("| " + model["label"] + " | " +
                        " | ".join("U" if v is None else str(v) for v in model[dataset]) + " |")
        text += ["", "100 queries per condition; higher is better. U = unsupported, not zero. "
                 "Prime uses gold-containing candidate sets, not end-to-end retrieval.", ""]
    for index, task in enumerate(data["rq3_tasks"]):
        unit = "absolute Q" if index == 3 else "%"
        text += [f"## RQ3: {task} ({unit} gap)", "",
                 "| Model | A: direct | Complete A | C: proposals | Complete C |",
                 "| --- | ---: | ---: | ---: | ---: |"]
        for model in data["models"]:
            a, c, na, nc = model["rq3"][index]
            def fmt(value):
                return "—" if value is None else f"{value:.7f}" if index == 3 else f"{value:.4f}"
            text.append(f"| {model['label']} | {fmt(a)} | {na}/10 | {fmt(c)} | {nc}/10 |")
        text += ["", "Lower is better. Means use completed trajectories; partial subsets are not "
                 "ranked against full coverage. A dash is undefined quality, not zero. "
                 "A zero-gap forced-only completion does not establish model support.", ""]
    text += [
        "## Reading the results", "",
        "- RQ1/2 invalid outputs count as wrong. RQ3 incomplete/unsupported trajectories "
        "remain in coverage, without an imputed objective.",
        "- GPT-5.4 results are final selections after invalid-only retries, not single-attempt claims. "
        "[Retry accounting](../assets/benchmark/v33/gpt54/retry_summary.csv) is separate.",
        "- RQ3: TSP/LT use exact optima; signed MaxCut uses historical BKS; modularity uses "
        "numerical MILP certificates. Non-GPT modularity objectives were rounded to four decimals "
        "before rebasing; printed extra gap digits do not imply extra measurement precision.",
        "- Compare paired modes only on jointly completed graphs. Their paired difference "
        "cannot generally be reconstructed by subtracting means with unequal coverage.",
        "- [Reproduction and model settings](reproduction.md).", "",
    ]
    return "\n".join(text)


def build(data):
    assert len(data["models"]) == 14
    assert len({m["id"] for m in data["models"]}) == 14
    for model in data["models"]:
        assert len(model["rq1"]) == 6 and all(0 <= v <= 100 for v in model["rq1"])
        assert all(len(model[d]) == 5 and all(v is None or 0 <= v <= 100 for v in model[d])
                   for d in ("arxiv", "prime"))
        assert len(model["rq3"]) == 4
        for a, c, na, nc in model["rq3"]:
            assert 0 <= na <= 10 and 0 <= nc <= 10
            assert (a is None) == (na == 0) and (c is None) == (nc == 0)
    return {
        ROOT / "overview.svg": hero(), ROOT / "rq1.svg": heatmap(data, "rq1"),
        ROOT / "rq2.svg": heatmap(data, "rq2"), ROOT / "rq3.svg": optimization(data),
        ROOT.parents[2] / "docs/results.md": tables(data),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if generated assets differ")
    args = parser.parse_args()
    data = json.loads((ROOT / "display-data.json").read_text())
    for path, content in build(data).items():
        if args.check:
            if not path.exists() or path.read_text() != content:
                raise ValueError(f"Generated asset differs: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        print(("Checked " if args.check else "Wrote ") + str(path.relative_to(ROOT.parents[2])))


if __name__ == "__main__":
    main()
