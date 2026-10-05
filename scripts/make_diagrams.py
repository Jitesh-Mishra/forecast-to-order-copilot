"""README diagram (light + dark) for forecast-to-order-copilot. Numbers come from reports/.

python scripts/make_diagrams.py  ->  docs/img/pipeline-{light,dark}.svg
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img"
REP = ROOT / "reports"

THEMES = {
    "light": dict(bg="#ffffff", card="#f6f8fa", border="#d0d7de", ink="#1f2328", ink2="#59636e", faint="#e6e9ed",
                  a="#2a78d6", a_soft="#e3eefb", b="#eb6834", b_soft="#fde8de", c="#1baf7a", c_soft="#dcf4ea",
                  muted="#afb8c1"),
    "dark": dict(bg="#0d1117", card="#161b22", border="#30363d", ink="#e6edf3", ink2="#9198a1", faint="#21262d",
                 a="#3987e5", a_soft="#13263d", b="#d95926", b_soft="#3a1d12", c="#199e70", c_soft="#0f2e23",
                 muted="#484f58"),
}
FONT = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class SVG:
    def __init__(self, w, h, t, label):
        self.w, self.h, self.t, self.label, self.parts = w, h, t, label, []

    def add(self, s):
        self.parts.append(s)

    def text(self, x, y, s, size=13, color=None, weight=400, anchor="start", mono=False, italic=False):
        self.add(f'<text x="{x}" y="{y}" font-family="{MONO if mono else FONT}" font-size="{size}" '
                 f'font-weight="{weight}" fill="{color or self.t["ink"]}" text-anchor="{anchor}"'
                 f'{" font-style=\"italic\"" if italic else ""}>{esc(s)}</text>')

    def rect(self, x, y, w, h, fill, stroke=None, rx=10, dash=None, sw=1.5):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" '
                 f'stroke="{stroke or "none"}" stroke-width="{sw}"{d}/>')

    def line(self, x1, y1, x2, y2, color=None, sw=1.5, dash=None, arrow=False, marker="ar"):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#{marker})"' if arrow else ""
        self.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color or self.t["ink2"]}" '
                 f'stroke-width="{sw}" stroke-linecap="round"{d}{m}/>')

    def path(self, d, color=None, sw=1.5, dash=None, arrow=False, marker="ar"):
        da = f' stroke-dasharray="{dash}"' if dash else ""
        m = f' marker-end="url(#{marker})"' if arrow else ""
        self.add(f'<path d="{d}" fill="none" stroke="{color or self.t["ink2"]}" stroke-width="{sw}" '
                 f'stroke-linecap="round" stroke-linejoin="round"{da}{m}/>')

    def circle(self, x, y, r, fill, stroke=None, sw=1.5):
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke or "none"}" stroke-width="{sw}"/>')

    def pill(self, x, y, s, fill, color, size=11.5, anchor="middle", pad=9):
        w = len(s) * size * 0.58 + 2 * pad
        x0 = x - w / 2 if anchor == "middle" else x
        self.rect(x0, y - size - 3, w, size + 10, fill, rx=(size + 10) / 2)
        self.text(x0 + w / 2, y + 1, s, size=size, color=color, weight=600, anchor="middle")

    def node(self, x, y, w, h, title, sub=None, accent=None, dash=None, mono_title=True):
        t = self.t
        fill = t[f"{accent}_soft"] if accent else t["card"]
        stroke = t[accent] if accent else t["border"]
        self.rect(x, y, w, h, fill, stroke, dash=dash)
        cy = y + h / 2 + (-4 if sub else 5)
        self.text(x + w / 2, cy, title, size=14, weight=650, anchor="middle", mono=mono_title)
        if sub:
            for i, s in enumerate(sub if isinstance(sub, list) else [sub]):
                self.text(x + w / 2, cy + 19 + i * 16, s, size=12, color=t["ink2"], anchor="middle")

    def render(self):
        t = self.t
        defs = (f'<defs>'
                f'<marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="{t["ink2"]}"/></marker>'
                f'<marker id="ara" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="{t["a"]}"/></marker>'
                f'<marker id="arb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
                f'<path d="M0,0 L10,5 L0,10 z" fill="{t["b"]}"/></marker></defs>')
        return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" width="{self.w}" '
                f'height="{self.h}" role="img" aria-label="{esc(self.label)}">{defs}'
                f'<rect width="{self.w}" height="{self.h}" rx="14" fill="{t["bg"]}" stroke="{t["border"]}"/>'
                + "".join(self.parts) + "</svg>")


def header(s, title, sub):
    s.text(32, 42, title, size=19, weight=700)
    s.text(32, 64, sub, size=13, color=s.t["ink2"])


# ----------------------------------------------------------------------------- pipeline
def pipeline(t):
    import pandas as pd
    bud = pd.read_csv(REP / "budget_test.csv")
    lam = bud[bud.budget_frac == 0.8].shadow_price_avg.iloc[0]
    s = SVG(1080, 500, t, "Pipeline: weekly quantile forecasts become cost-optimal order-up-to levels, a budget LP "
                          "trades service across SKUs, and a replay simulator scores every policy on held-out sales.")
    header(s, "From a forecast to an order the business can defend",
           "The forecast is an input. The deliverable is the ordering policy and its simulated cost.")
    # config
    s.rect(668, 84, 392, 38, t["b_soft"], t["b"], rx=10)
    s.text(864, 108, "config.yaml  ·  margin 30%  ·  holding 25%/yr  ·  spoilage 5%/wk", size=12, color=t["b"],
           weight=600, anchor="middle")
    xs, w, y, h = [32, 244, 456, 668, 880], 180, 146, 104
    s.node(xs[0], y, w, h, "M5 sales", ["Walmart CA_1 · FOODS", "1,437 SKUs · 277 weeks"], mono_title=False)
    s.node(xs[1], y, w, h, "features", ["only weeks < origin", "lags · price · SNAP"])
    s.node(xs[2], y, w, h, "forecast", ["LightGBM P30 to P98", "vs TSB · MA4 · naive"], accent="a")
    s.node(xs[3], y, w, h, "policy", ["CR = cu / (cu + co)", "S = Q(CR), per SKU"], accent="c")
    s.node(xs[4], y, w, h, "budget LP", ["HiGHS · pick service", f"per SKU · λ = {lam:.3f}"], accent="c")
    for a, b in zip(xs[:-1], xs[1:]):
        s.line(a + w, y + h / 2, b - 4, y + h / 2, arrow=True)
    s.line(xs[3] + 90, 122, xs[3] + 90, y - 4, t["b"], 1.5, arrow=True, marker="arb")
    s.line(xs[4] + 90, 122, xs[4] + 90, y - 4, t["b"], 1.5, arrow=True, marker="arb")
    # row 2
    y2 = 300
    s.node(456, y2, 392, 86, "replay simulator", ["held-out weeks · lost sales · carry-over stock",
                                                    "cost = co x leftover + cu x lost"], accent="b")
    s.node(880, y2, 180, 86, "decision", ["frontier · sensitivity", "exec brief"], mono_title=False)
    s.path(f"M{xs[0]+90},{y+h} L{xs[0]+90},{y2+43} L452,{y2+43}", arrow=True)
    s.text(136, y2 + 34, "actual weekly sales (never seen by the forecaster)", size=11, color=t["ink2"])
    s.line(xs[3] + 90, y + h, xs[3] + 90, y2 - 4, arrow=True)
    s.text(xs[3] + 98, y + h + 26, "S", size=11, color=t["ink2"], mono=True)
    s.path(f"M{xs[4]+60},{y+h} C{xs[4]+60},{y+h+30} 828,{y2-24} 828,{y2-4}", arrow=True)
    s.text(xs[4] + 70, y + h + 26, "budgeted S", size=11, color=t["ink2"], mono=True)
    s.line(848, y2 + 43, 876, y2 + 43, arrow=True)
    # next: copilot
    s.rect(456, 418, 604, 56, t["bg"], t["border"], rx=10, dash="5 4")
    s.text(476, 442, "next  ·  planner copilot", size=13, weight=650)
    s.text(476, 461, "Groq tool-calling over DuckDB tables · every number comes from a tool, never the LLM", size=12,
           color=t["ink2"])
    s.line(970, y2 + 86, 970, 414, dash="4 4", arrow=True)
    return s


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for theme, tok in THEMES.items():
        (OUT / f"pipeline-{theme}.svg").write_text(pipeline(tok).render())
    print("ok")
