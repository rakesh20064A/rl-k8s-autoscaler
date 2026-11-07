"""theme.py -- Visual language: precision-instrument / engineering-console
redesign. Deliberately NOT an AI-dashboard, SaaS-admin, or neon control
room: flat surfaces, thin hairlines, tight type scale, color used only
semantically. No external font or network dependency -- system font
stacks only.
"""

from __future__ import annotations

PALETTE = {
    "canvas": "#F4F2EC",
    "surface": "#FBFAF6",
    "surface_2": "#E9E6DE",
    "ink": "#171A18",
    "muted": "#62665F",
    "hairline": "#D1CEC4",
    "petrol": "#234A46",      # primary active state
    "vermilion": "#D34A32",   # attention / events / errors / PPO accent
    "olive": "#6D7651",       # healthy state / CPU accent
    "ochre": "#C18A28",       # warning / comparison accent / HPA accent
    "panel_dark": "#1B1F1D",        # evidence-panel background: dark charcoal/graphite
    "panel_dark_2": "#262B28",      # row hover / slightly-raised surface within the panel
    "panel_dark_border": "#333933",  # hairline border, slightly lighter than the panel bg
    "panel_dark_text": "#F2EFE7",    # default log text: warm off-white
    "panel_dark_muted": "#8F978E",   # timestamp: muted gray-green
    "panel_dark_accent": "#9FC2B0",  # category/source label: subtle sage/teal accent
    "panel_dark_emphasis": "#F2CDAA",  # emphasis/value text (event type): warm highlight
}

FONT_STACK_SANS = (
    '"Avenir Next", "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif'
)
FONT_STACK_MONO = (
    '"SFMono-Regular", "SF Mono", Menlo, Monaco, Consolas, monospace'
)

RADIUS = "3px"
BORDER_THIN = f"1px solid {PALETTE['hairline']}"

# Chart series colors (sec.8) -- never more than a few at once, no rainbow.
CHART_COLORS = {
    "traffic": PALETTE["petrol"],
    "hpa": PALETTE["ochre"],
    "ppo": PALETTE["vermilion"],
    "cpu": PALETTE["olive"],
    "latency": "#5E6260",
}


def inject_css() -> str:
    """Returns a single <style> block. Caller (app.py) is responsible for
    passing this to `st.markdown(..., unsafe_allow_html=True)` -- kept as a
    plain string here so it is importable/testable without Streamlit."""
    p = PALETTE
    return f"""
<style>
:root {{
  --canvas: {p['canvas']};
  --surface: {p['surface']};
  --surface-2: {p['surface_2']};
  --ink: {p['ink']};
  --muted: {p['muted']};
  --hairline: {p['hairline']};
  --petrol: {p['petrol']};
  --vermilion: {p['vermilion']};
  --olive: {p['olive']};
  --ochre: {p['ochre']};
  --panel-dark: {p['panel_dark']};
  --panel-dark-2: {p['panel_dark_2']};
  --panel-dark-border: {p['panel_dark_border']};
  --panel-dark-text: {p['panel_dark_text']};
  --panel-dark-muted: {p['panel_dark_muted']};
  --panel-dark-accent: {p['panel_dark_accent']};
  --panel-dark-emphasis: {p['panel_dark_emphasis']};
}}

/* ---- base canvas / chrome reduction --------------------------------- */

.stApp {{
  background-color: var(--canvas);
  color: var(--ink);
  font-family: {FONT_STACK_SANS};
  font-size: 14px;
}}

[data-testid="stHeader"] {{ background-color: var(--canvas); height: 0; }}
.block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1280px; }}
#MainMenu, footer {{ visibility: hidden; }}

/* Streamlit's native multipage sidebar nav must never appear alongside our
   own custom st.sidebar.radio page switcher (views/ is deliberately named
   to avoid auto-detection, but hide it defensively too). */
[data-testid="stSidebarNav"] {{ display: none; }}

/* Force every text-bearing surface (ours and Streamlit's native widgets) to
   our explicit palette, regardless of the viewer's OS/browser dark-mode
   preference -- demo_ui/.streamlit/config.toml forces the base theme to
   light, and these rules are the defensive second layer so no native
   widget can ever render near-white text on this light canvas. */
.stApp, .stApp p, .stApp span, .stApp div, .stApp label {{ color: var(--ink); }}
.stCaption, [data-testid="stCaptionContainer"], .stApp small {{ color: var(--muted) !important; }}
[data-testid="stWidgetLabel"] p {{ color: var(--muted) !important; }}

div[data-baseweb="select"] > div {{
  background-color: var(--surface) !important;
  color: var(--ink) !important;
  border-color: var(--hairline) !important;
}}
div[data-baseweb="select"] span {{ color: var(--ink) !important; }}
/* The select dropdown popover renders in a portal appended to <body>, so it
   needs its own (non-scoped) rule to pick up our colors. */
div[data-baseweb="popover"] li, div[data-baseweb="menu"] li {{
  background-color: var(--surface) !important;
  color: var(--ink) !important;
}}
div[data-baseweb="popover"] li:hover, div[data-baseweb="menu"] li:hover {{
  background-color: var(--surface-2) !important;
}}

/* Radio option text: Streamlit/BaseWeb may put the visible label in a
   <p>, <span>, or plain <div> depending on version/widget context (not
   always the <p> this rule used to assume alone), and can apply its own
   inline/emotion-cache color for the circle+text wrapper. Force every
   descendant of the label explicitly (not just label/p) so no radio's
   option text can render invisible against this light canvas regardless
   of which tag Streamlit happens to use. */
div[role="radiogroup"] label,
div[role="radiogroup"] label *,
div[role="radiogroup"] p,
div[role="radiogroup"] span,
div[role="radiogroup"] div {{ color: var(--ink) !important; }}
div[data-testid="stExpander"] summary, div[data-testid="stExpander"] p {{ color: var(--ink) !important; }}
div[data-testid="stExpander"] {{ background-color: var(--surface); border-color: var(--hairline) !important; border-radius: {RADIUS}; }}

h1, h2, h3, h4, h5 {{
  font-family: {FONT_STACK_SANS};
  font-weight: 600;
  letter-spacing: -0.01em;
  color: var(--ink);
}}
h1 {{ font-size: 20px; margin-bottom: 0; }}
h2 {{ font-size: 16px; }}
h3 {{ font-size: 14px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}

.mono, .evidence-log, .tabular-num {{
  font-family: {FONT_STACK_MONO};
  font-variant-numeric: tabular-nums;
}}

hr {{ border: none; border-top: {BORDER_THIN}; margin: 0.9rem 0; }}

/* flatten Streamlit's default widget chrome */
.stButton > button {{
  border-radius: {RADIUS};
  border: 1px solid var(--ink);
  background-color: var(--surface);
  color: var(--ink);
  font-size: 13px;
  font-weight: 600;
  padding: 0.35rem 1rem;
  box-shadow: none;
}}
.stButton > button:hover {{ background-color: var(--ink); color: var(--surface); border-color: var(--ink); }}
.stButton > button:disabled {{ opacity: 0.35; }}

div[data-baseweb="select"] > div, .stTextInput > div > div, div[data-testid="stWidgetLabel"] {{
  border-radius: {RADIUS} !important;
  font-size: 13px;
}}
div[data-testid="stWidgetLabel"] label {{
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--muted);
  font-weight: 600;
}}
div[role="radiogroup"] {{ gap: 0.1rem; }}
div[role="radiogroup"] label {{ font-size: 13px; }}

/* ---- top bar ---------------------------------------------------------- */

.du-topbar {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  border-bottom: 2px solid var(--ink);
  padding-bottom: 0.5rem;
  margin-bottom: 0.4rem;
}}
.du-topbar-title {{ font-size: 12px; letter-spacing: 0.12em; color: var(--muted); text-transform: uppercase; }}
.du-topbar-name {{ font-size: 20px; font-weight: 700; letter-spacing: -0.01em; }}
.du-topbar-sub {{ font-size: 12px; color: var(--muted); margin-top: 2px; }}

/* ---- status / mode indicator ------------------------------------------ */

.du-mode-indicator {{
  display: inline-block;
  font-family: {FONT_STACK_MONO};
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.08em;
  padding: 2px 8px;
  border-radius: {RADIUS};
  border: 1px solid currentColor;
}}
.du-mode-live {{ color: var(--petrol); }}
.du-mode-demo {{ color: var(--ochre); }}

/* ---- control strip ------------------------------------------------------ */

.du-control-strip {{
  background-color: var(--surface-2);
  border: {BORDER_THIN};
  border-top: 3px solid var(--petrol);
  border-radius: {RADIUS};
  padding: 0.7rem 1rem 0.3rem 1rem;
  margin-bottom: 0.6rem;
}}

/* ---- panels / surfaces ------------------------------------------------ */

.du-panel {{
  background-color: var(--surface);
  border: {BORDER_THIN};
  border-radius: {RADIUS};
  padding: 0.7rem 0.9rem;
}}

.du-section-label {{
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  margin-bottom: 0.3rem;
}}

/* ---- instrument metric strip (sec.7) ---------------------------------- */

.du-metric-strip {{
  display: flex;
  background-color: var(--surface);
  border: {BORDER_THIN};
  border-radius: {RADIUS};
  padding: 0.6rem 0;
  margin-bottom: 0.6rem;
}}
.du-metric-cell {{
  flex: 1;
  padding: 0 1.1rem;
  border-left: {BORDER_THIN};
}}
.du-metric-cell:first-child {{ border-left: none; }}
.du-metric-cell .du-metric-label {{
  font-size: 10.5px;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  color: var(--muted);
  font-weight: 600;
}}
.du-metric-cell .du-metric-value {{
  font-family: {FONT_STACK_MONO};
  font-variant-numeric: tabular-nums;
  font-size: 22px;
  font-weight: 700;
  color: var(--ink);
  line-height: 1.3;
}}

/* ---- cards (used sparingly: controller panel, status dims) ------------ */

.du-card {{
  background-color: var(--surface);
  border: {BORDER_THIN};
  border-left: 3px solid var(--muted);
  border-radius: {RADIUS};
  padding: 0.5rem 0.75rem;
}}
.du-card.controller-ppo {{ border-left-color: var(--vermilion); }}
.du-card.controller-hpa {{ border-left-color: var(--ochre); }}
.du-card.status-ok {{ border-left-color: var(--olive); }}
.du-card.status-warn {{ border-left-color: var(--ochre); }}
.du-card.status-error {{ border-left-color: var(--vermilion); }}
.du-card.active {{ border-left-color: var(--petrol); border-left-width: 4px; }}

.du-badge {{
  display: inline-block;
  border: 1px solid var(--hairline);
  border-radius: {RADIUS};
  padding: 1px 7px;
  font-size: 10.5px;
  font-family: {FONT_STACK_MONO};
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--muted);
}}
.du-badge.status-ok {{ color: var(--olive); border-color: var(--olive); }}
.du-badge.status-warn {{ color: var(--ochre); border-color: var(--ochre); }}
.du-badge.status-error {{ color: var(--vermilion); border-color: var(--vermilion); }}

/* ---- pipeline stepper (Auto mode, sec.12) ------------------------------ */

.du-pipeline {{
  display: flex;
  align-items: center;
  background-color: var(--surface);
  border: {BORDER_THIN};
  border-radius: {RADIUS};
  padding: 0.7rem 1rem;
  margin-bottom: 0.6rem;
}}
.du-pipeline-step {{
  display: flex;
  flex-direction: column;
  align-items: center;
  font-family: {FONT_STACK_MONO};
  font-size: 11px;
  letter-spacing: 0.04em;
  color: var(--muted);
  min-width: 90px;
}}
.du-pipeline-dot {{ font-size: 15px; line-height: 1; margin-bottom: 3px; }}
.du-pipeline-step.done {{ color: var(--olive); }}
.du-pipeline-step.active {{ color: var(--petrol); font-weight: 700; }}
.du-pipeline-connector {{
  flex: 1;
  height: 1px;
  background-color: var(--hairline);
  margin: 0 4px 16px 4px;
}}
.du-pipeline-connector.done {{ background-color: var(--olive); }}

/* ---- pod list (sec.10) ------------------------------------------------- */

.du-pod-row {{
  display: flex;
  justify-content: space-between;
  font-family: {FONT_STACK_MONO};
  font-size: 12.5px;
  padding: 3px 0;
  border-bottom: 1px solid var(--hairline);
}}
.du-pod-row:last-child {{ border-bottom: none; }}
.du-pod-dot {{ margin-right: 6px; }}
.du-pod-dot.ready {{ color: var(--olive); }}
.du-pod-dot.starting {{ color: var(--ochre); }}
.du-pod-dot.failed {{ color: var(--vermilion); }}
.du-pod-dot.terminating {{ color: var(--muted); }}

/* ---- event timeline / evidence (sec.11), dark terminal-style panel ------
   Every text role below gets an EXPLICIT color scoped under
   `.du-evidence-panel` with a class selector on the element itself
   (specificity 0,2,0 or 0,2,1), which beats the page-wide contrast-safety
   rule `.stApp div/span {{ color: var(--ink) }}` (specificity 0,1,1) --
   without this, this dark panel's text silently inherits the light-theme
   ink color and reads as near-invisible dark-on-dark. Never rely on
   inherited/cascaded color inside this panel. */

.du-evidence-panel {{
  background-color: var(--panel-dark);
  border: 1px solid var(--panel-dark-border);
  border-radius: {RADIUS};
  padding: 0.65rem 0.9rem;
  max-height: 380px;
  overflow-y: auto;
  font-family: {FONT_STACK_MONO};
  font-size: 13px;
  line-height: 1.55;
}}
.du-evidence-panel .du-evidence-row {{
  color: var(--panel-dark-text);
  white-space: pre;
  padding: 1px 0;
  border-bottom: 1px solid var(--panel-dark-border);
}}
.du-evidence-panel .du-evidence-row:last-child {{ border-bottom: none; }}
.du-evidence-panel .du-evidence-row:hover {{ background-color: var(--panel-dark-2); }}
.du-evidence-panel .du-evidence-ts {{
  color: var(--panel-dark-muted);
  margin-right: 0.9em;
}}
.du-evidence-panel .du-evidence-source {{
  color: var(--panel-dark-accent);
  margin-right: 0.9em;
  display: inline-block;
}}
.du-evidence-panel .du-evidence-event {{
  color: var(--panel-dark-emphasis);
  font-weight: 600;
  margin-right: 0.9em;
}}
.du-evidence-panel .du-evidence-message {{
  color: var(--panel-dark-text);
}}
.du-evidence-panel .du-evidence-empty {{
  color: var(--panel-dark-muted);
  border-bottom: none;
}}

/* ---- comparison table (sec.13) ----------------------------------------- */

.du-cmp-table {{
  width: 100%;
  border-collapse: collapse;
  font-family: {FONT_STACK_MONO};
  font-size: 13px;
}}
.du-cmp-table th {{
  text-align: right;
  font-family: {FONT_STACK_SANS};
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--muted);
  border-bottom: 2px solid var(--ink);
  padding: 4px 10px;
}}
.du-cmp-table th:first-child {{ text-align: left; }}
.du-cmp-table td {{
  text-align: right;
  padding: 5px 10px;
  border-bottom: 1px solid var(--hairline);
}}
.du-cmp-table td:first-child {{ text-align: left; font-family: {FONT_STACK_SANS}; }}
.du-delta-pos {{ color: var(--vermilion); }}
.du-delta-neg {{ color: var(--olive); }}

/* ---- editorial conclusion block (sec.14) -------------------------------- */

.du-conclusion {{
  background-color: var(--surface);
  border-left: 3px solid var(--petrol);
  border-radius: {RADIUS};
  padding: 1rem 1.2rem;
  font-size: 14.5px;
  line-height: 1.6;
}}
.du-conclusion-headline {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); margin-bottom: 0.5rem; font-weight: 700; }}

/* ---- history table ------------------------------------------------------ */

.du-history-table {{
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}}
.du-history-table th {{
  text-align: left;
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--muted);
  border-bottom: 2px solid var(--ink);
  padding: 4px 8px;
}}
.du-history-table td {{ padding: 5px 8px; border-bottom: 1px solid var(--hairline); font-family: {FONT_STACK_MONO}; font-size: 12.5px; }}

/* ---- misc -------------------------------------------------------------- */

.du-divider {{ border: none; border-top: {BORDER_THIN}; margin: 0.7rem 0; }}
.du-metric-label {{ font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); }}
.du-metric-value {{ font-family: {FONT_STACK_MONO}; font-variant-numeric: tabular-nums; font-size: 18px; font-weight: 700; }}
.du-offline-banner {{
  display: inline-block;
  font-family: {FONT_STACK_MONO};
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.07em;
  color: var(--ochre);
  border: 1px solid var(--ochre);
  border-radius: {RADIUS};
  padding: 3px 10px;
  margin-bottom: 0.6rem;
}}
</style>
"""
