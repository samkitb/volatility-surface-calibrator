"""
IMPORTANT NOTES:
Delta — how much the option price moves for every $1 move in the stock. 
A delta of 0.6 means if Apple goes up $1, your option goes up $0.60. 
This is the most watched Greek — it tells you your directional exposure.

Gamma — how fast delta itself changes as the stock moves. 
High gamma means your delta is unstable and changing rapidly. 

Vega — how much the option price changes for every 1% move in implied volatility. 
This is what volatility traders care about most — it measures your direct exposure to IV moves.

Theta — how much value the option loses each day just from time passing. 
Options decay in value as expiry approaches even if the stock doesn't move. 
Theta is always negative for option buyers — time is your enemy.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from network import VolSurfaceNet, predict_surface
from scipy.stats import norm


# ═══════════════════════════════════════════════════════════════════════
#  App setup
# ═══════════════════════════════════════════════════════════════════════

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True,
)
app.title = "Vol Surface Calibrator"

MONEYNESS_GRID = np.linspace(0.70, 1.30, 60)
TIME_GRID      = np.linspace(0.04, 1.0,  40)

GREEK_META = {
    "Delta": {"color": "#5DCAA5", "desc": "$ move per $1 stock move"},
    "Gamma": {"color": "#4facfe", "desc": "delta sensitivity to spot"},
    "Vega":  {"color": "#EF9F27", "desc": "$ move per 1% IV change"},
    "Theta": {"color": "#D85A30", "desc": "daily time decay"},
    "IV":    {"color": "#ffffff", "desc": "annualized implied volatility"},
}

# ═══════════════════════════════════════════════════════════════════════
#  Helper functions (unchanged logic)
# ═══════════════════════════════════════════════════════════════════════

def load_model(ticker):
    path = f"models/{ticker.lower()}_vol_surface.pt"
    if not os.path.exists(path):
        return None, None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = VolSurfaceNet(hidden_size=checkpoint.get("hidden_size", 64))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def compute_greeks(S, K, T, r, iv):
    if T <= 0 or iv <= 0:
        return {}
    d1 = (np.log(S / K) + (r + 0.5 * iv**2) * T) / (iv * np.sqrt(T))
    d2 = d1 - iv * np.sqrt(T)
    delta = round(float(norm.cdf(d1)), 4)
    gamma = round(float(norm.pdf(d1) / (S * iv * np.sqrt(T))), 6)
    vega  = round(float(S * norm.pdf(d1) * np.sqrt(T) / 100), 4)
    theta = round(float((-(S * norm.pdf(d1) * iv) / (2 * np.sqrt(T)) -
             r * K * np.exp(-r * T) * norm.cdf(d2)) / 365), 4)
    return {"Delta": delta, "Gamma": gamma, "Vega": vega,
            "Theta": theta, "IV": f"{iv*100:.1f}%"}


def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = int(h[:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ═══════════════════════════════════════════════════════════════════════
#  Build functions
# ═══════════════════════════════════════════════════════════════════════

def build_surface_fig(ticker, model, shock=0, raw_df=None):
    iv_surface = predict_surface(model, MONEYNESS_GRID, TIME_GRID)
    iv_surface = np.clip(iv_surface + shock, 0.01, None)
    K_grid, T_grid = np.meshgrid(MONEYNESS_GRID, TIME_GRID)
    
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=K_grid,
        y=T_grid * 365,
        z=iv_surface * 100,
        colorscale="Jet",
        colorbar=dict(
            title=dict(text="IV (%)", side="right",
                       font=dict(size=11, color="#99aabb")),
            ticksuffix="%", len=0.55,
            bgcolor="rgba(0,0,0,0)",
            tickfont=dict(color="#99aabb", size=10),
            outlinewidth=0,
        ),
        opacity=0.92,
        contours=dict(
            x=dict(show=True, width=1, color="rgba(79,172,254,0.1)",
                   highlightwidth=1),
            y=dict(show=True, width=1, color="rgba(79,172,254,0.1)",
                   highlightwidth=1),
            z=dict(show=True, width=1, color="rgba(255,255,255,0.12)",
                   highlightwidth=2),
        ),
        name="Neural network surface",
        hovertemplate=(
            "Moneyness: %{x:.3f}<br>"
            "Days to expiry: %{y:.0f}<br>"
            "IV: %{z:.1f}%<extra></extra>"
        ),
    ))

    if raw_df is not None:
        df_plot = raw_df.copy()
        if shock != 0:
            df_plot["implied_vol"] = (df_plot["implied_vol"] + shock).clip(0.01)
        df_plot = df_plot[df_plot["implied_vol"] <= 0.45]  # add this line
        fig.add_trace(go.Scatter3d(
            x=df_plot["moneyness"],
            y=df_plot["time_to_expiry"] * 365,
            z=df_plot["implied_vol"] * 100,
            mode="markers",
            marker=dict(size=4, color="#FFD700", opacity=1.0),
            name="Market data",
            hovertemplate=(
                "Moneyness: %{x:.3f}<br>"
                "Days: %{y:.0f}<br>"
                "IV: %{z:.1f}%<extra></extra>"
            ),
        ))

    axis_common = dict(
        backgroundcolor="#060614",
        gridcolor="#222244",
        showbackground=True,
        title=dict(font=dict(size=13, color="#4facfe")),
        tickfont=dict(size=11, color="#ddeeff"),
        color="#ddeeff",
        showspikes=False,
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="Moneyness (K/S)",
            yaxis_title="Days to expiry",
            zaxis_title="Implied vol (%)",
            xaxis=axis_common,
            yaxis=axis_common,
            zaxis=dict(**axis_common, range=[15, 55]),
            bgcolor="#000008",
            camera=dict(eye=dict(x=1.4, y=-1.6, z=0.9)),
            dragmode="turntable",
        ),
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#0a0a0f",
        font=dict(color="#ddd", size=11),
        margin=dict(l=0, r=0, t=30, b=0),
        title=dict(
            text=f"{ticker} Implied Volatility Surface",
            font=dict(size=12, color="#4facfe"),
            x=0.02, xanchor="left",
        ),
        legend=dict(
            bgcolor="rgba(15,15,26,0.9)",
            font=dict(color="#ccc", size=10),
            bordercolor="#1a1a2e",
            borderwidth=1,
            x=0.01, y=0.99,
        ),
        clickmode="event",
    )
    return fig


def build_metrics_panel(greeks, moneyness=None, days=None,
                        label="ATM Greeks — 30d", checkpoint=None,
                        is_live=False):
    if moneyness is not None and days is not None:
        is_live = True

    header = [html.Span(className="live-dot")] if is_live else []
    header.append(html.Span("GREEKS", style={
        "color": "#4facfe", "fontSize": "11px", "fontWeight": "600",
        "letterSpacing": "1.5px",
    }))

    children = [html.Div(header, style={"marginBottom": "6px"})]

    if moneyness is not None and days is not None:
        children.append(html.Div(
            f"M: {moneyness:.3f}  |  T: {days:.0f}d",
            style={"color": "#555", "fontSize": "10px", "marginBottom": "10px",
                   "letterSpacing": "0.5px"},
        ))
    else:
        children.append(html.Div(label, style={
            "color": "#555", "fontSize": "10px", "marginBottom": "10px",
        }))

    card_cls = "greek-card greek-flash" if is_live else "greek-card"

    for name, value in greeks.items():
        meta = GREEK_META.get(name, {"color": "#fff", "desc": ""})
        children.append(html.Div([
            html.Div(name, style={
                "color": "#666", "fontSize": "9px", "fontWeight": "500",
                "letterSpacing": "1px", "textTransform": "uppercase",
                "marginBottom": "2px",
            }),
            html.Div(str(value), style={
                "color": meta["color"], "fontSize": "18px",
                "fontWeight": "600", "lineHeight": "1.2",
            }),
            html.Div(meta["desc"], style={
                "color": "#3a3a4a", "fontSize": "9px", "marginTop": "2px",
            }),
        ], className=card_cls, style={"borderLeftColor": meta["color"]}))

    if checkpoint:
        epochs = checkpoint.get("epochs_trained", "?")
        loss = checkpoint.get("final_loss", 0)
        children.append(html.Div([
            html.Div(style={"borderTop": "1px solid #1a1a2e",
                            "margin": "10px 0 8px 0"}),
            html.Div(f"Epochs: {epochs}  |  Loss: {loss:.5f}",
                     style={"color": "#333", "fontSize": "9px"}),
        ]))

    return children


# ═══════════════════════════════════════════════════════════════════════
#  Layout
# ═══════════════════════════════════════════════════════════════════════

app.layout = html.Div([
    # ── Header bar ───────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Span(className="header-dot"),
            html.Span("VOL SURFACE CALIBRATOR", style={
                "color": "#4facfe", "fontSize": "14px", "fontWeight": "600",
                "letterSpacing": "3px", "verticalAlign": "middle",
            }),
        ], style={"display": "flex", "alignItems": "center"}),
        html.Div([
            html.Span("Neural Network Calibrated",
                       className="status-pill pill-blue"),
            html.Span("No-Arbitrage Constrained",
                       className="status-pill pill-green",
                       style={"marginLeft": "8px"}),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={
        "display": "flex", "justifyContent": "space-between",
        "alignItems": "center", "padding": "14px 0",
        "borderBottom": "1px solid #1a1a2e", "marginBottom": "16px",
    }),

    # ── Controls bar ─────────────────────────────────────────────
    html.Div([
        dbc.Row([
            dbc.Col([
                html.Label("TICKER", style={
                    "fontSize": "9px", "color": "#555",
                    "letterSpacing": "1.5px", "fontWeight": "500",
                    "marginBottom": "4px", "display": "block",
                }),
                html.Div([
                    html.Span("$", style={
                        "position": "absolute", "left": "10px", "top": "50%",
                        "transform": "translateY(-50%)", "color": "#4facfe",
                        "fontSize": "13px", "fontWeight": "600", "zIndex": "2",
                        "pointerEvents": "none",
                    }),
                    dcc.Input(id="ticker-input", value="AAPL", type="text",
                              style={"width": "100%"}),
                ], style={"position": "relative"}),
            ], width=2),

            dbc.Col([
                html.Div([
                    html.Label("VOL SHOCK", style={
                        "fontSize": "9px", "color": "#555",
                        "letterSpacing": "1.5px", "fontWeight": "500",
                        "display": "inline-block",
                    }),
                    html.Span(id="shock-badge", children="0%", style={
                        "background": "#4facfe", "color": "#0a0a0f",
                        "padding": "1px 8px", "borderRadius": "10px",
                        "fontSize": "10px", "fontWeight": "600",
                        "marginLeft": "8px", "display": "inline-block",
                    }),
                ], style={"marginBottom": "8px"}),
                dcc.Slider(
                    id="vol-shock", min=-50, max=50, step=5, value=0,
                    marks={i: {"label": f"{i}%", "style": {"color": "#333"}}
                           for i in range(-50, 51, 25)},
                ),
            ], width=5),

            dbc.Col([
                html.Label("MARKET DATA", style={
                    "fontSize": "9px", "color": "#555",
                    "letterSpacing": "1.5px", "fontWeight": "500",
                    "marginBottom": "6px", "display": "block",
                }),
                dbc.Switch(id="show-raw", value=True, label="",
                           style={"marginTop": "2px"}),
            ], width=2, style={
                "display": "flex", "flexDirection": "column",
                "alignItems": "center",
            }),

            dbc.Col([
                html.Label("", style={
                    "fontSize": "9px", "display": "block",
                    "marginBottom": "4px",
                }),
                html.Button("LOAD SURFACE", id="load-btn",
                            className="load-btn", n_clicks=0),
            ], width=3, style={
                "display": "flex", "flexDirection": "column",
                "justifyContent": "center",
            }),
        ], align="center"),
    ], className="panel-card", style={"marginBottom": "16px"}),

    # ── Main: Surface + Greeks ───────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Loading(
                    type="dot",
                    color="#4facfe",
                    children=dcc.Graph(
                        id="surface-plot",
                        style={"height": "560px"},
                        config={"displayModeBar": True, "scrollZoom": True},
                    ),
                ),
                html.Div(
                    "Click any point on the surface to inspect Greeks",
                    className="intro-tooltip",
                ),
            ], style={"position": "relative"}),
        ], width=9),

        dbc.Col([
            html.Div(
                id="metrics-panel",
                children=[
                    html.Div("GREEKS", style={
                        "color": "#4facfe", "fontSize": "11px",
                        "fontWeight": "600", "letterSpacing": "1.5px",
                        "marginBottom": "12px",
                    }),
                    html.Div("Load a surface to begin", style={
                        "color": "#333", "fontSize": "11px",
                    }),
                ],
                className="panel-card",
                style={"height": "560px", "overflowY": "auto"},
            ),
        ], width=3),
    ], style={"marginBottom": "16px"}),

    # ── Bottom charts ────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            html.Div([
                dcc.Graph(id="smile-plot", style={"height": "280px"})
            ], className="panel-card chart-card"),
        ], width=6),
        dbc.Col([
            html.Div([
                dcc.Graph(id="term-plot", style={"height": "280px"})
            ], className="panel-card chart-card"),
        ], width=6),
    ]),

    # ── Footer ───────────────────────────────────────────────────
    html.Div([
        dbc.Row([
            dbc.Col(
                html.Span("Data: Yahoo Finance via yfinance"),
                width=4,
            ),
            dbc.Col(
                html.Span(
                    "Model: Feedforward NN  |  4 layers \u00d7 64 neurons"
                    "  |  Arbitrage-constrained"
                ),
                width=4, style={"textAlign": "center"},
            ),
            dbc.Col(
                html.Span(id="live-timestamp"),
                width=4, style={"textAlign": "right"},
            ),
        ]),
    ], className="footer-bar"),

    # ── Hidden stores & intervals ────────────────────────────────
    dcc.Store(id="ticker-store", data="AAPL"),
    dcc.Store(id="shock-store", data=0),
    dcc.Interval(id="clock-interval", interval=1000, n_intervals=0),

], style={
    "background": "#0a0a0f", "minHeight": "100vh",
    "color": "white", "padding": "0 24px 16px 24px",
    "--Dash-Fill-Interactive-Strong": "#4facfe",
    "--Dash-Fill-Interactive-Weak": "rgba(79,172,254,0.15)",
    "--Dash-Stroke-Strong": "#1a1a2e",
    "--Dash-Stroke-Weak": "#1a1a2e",
    "--Dash-Text-Primary": "#ffffff",
    "--Dash-Text-Strong": "#ffffff",
    "--Dash-Text-Weak": "#666666",
    "--Dash-Text-Disabled": "#333333",
    "--Dash-Fill-Primary-Hover": "rgba(79,172,254,0.08)",
    "--Dash-Fill-Primary-Active": "rgba(79,172,254,0.12)",
    "--Dash-Fill-Disabled": "#1a1a2e",
    "--Dash-Shading-Strong": "#1a1a2e",
    "--Dash-Shading-Weak": "#0f0f1a",
    "--Dash-Fill-Inverse-Strong": "#0a0a0f",
})


# ═══════════════════════════════════════════════════════════════════════
#  Callbacks
# ═══════════════════════════════════════════════════════════════════════

@app.callback(
    Output("surface-plot", "figure"),
    Output("smile-plot", "figure"),
    Output("term-plot", "figure"),
    Output("metrics-panel", "children"),
    Output("ticker-store", "data"),
    Output("shock-store", "data"),
    Input("load-btn", "n_clicks"),
    Input("vol-shock", "value"),
    Input("show-raw", "value"),
    State("ticker-input", "value"),
    prevent_initial_call=False,
)
def update_dashboard(n_clicks, vol_shock_pct, show_raw, ticker):
    ticker = (ticker or "AAPL").upper()
    model, checkpoint = load_model(ticker)
    shock = (vol_shock_pct or 0) / 100

    empty = go.Figure()
    empty.update_layout(
        paper_bgcolor="#0a0a0f", plot_bgcolor="#0f0f1a",
        font=dict(color="#333"),
        margin=dict(l=40, r=10, t=30, b=30),
        xaxis=dict(gridcolor="#1a1a2e", zerolinecolor="#1a1a2e"),
        yaxis=dict(gridcolor="#1a1a2e", zerolinecolor="#1a1a2e"),
    )

    if model is None:
        msg = [
            html.Div("GREEKS", style={
                "color": "#4facfe", "fontSize": "11px", "fontWeight": "600",
                "letterSpacing": "1.5px", "marginBottom": "12px",
            }),
            html.Div(f"No model found for {ticker}", style={
                "color": "#D85A30", "fontSize": "11px", "marginBottom": "4px",
            }),
            html.Div("Run train.py first to generate a model.", style={
                "color": "#444", "fontSize": "10px",
            }),
        ]
        return empty, empty, empty, msg, ticker, shock

    raw_df = None
    if show_raw:
        csv_path = f"data/{ticker.lower()}_vol_surface.csv"
        if os.path.exists(csv_path):
            raw_df = pd.read_csv(csv_path)

    surface_fig = build_surface_fig(ticker, model, shock, raw_df)

    # ── Vol smile chart ──────────────────────────────────────────
    smile_times = [30, 60, 90, 180]
    smile_fig = go.Figure()
    colors = ["#4facfe", "#5DCAA5", "#EF9F27", "#D85A30"]

    for t_days, color in zip(smile_times, colors):
        T = t_days / 365
        T_arr = np.full_like(MONEYNESS_GRID, T)
        X = torch.tensor(
            np.column_stack([MONEYNESS_GRID, np.log(T_arr)]),
            dtype=torch.float32,
        )
        with torch.no_grad():
            iv_smile = model(X).numpy()
        iv_smile = np.clip(iv_smile + shock, 0.01, None)
        smile_fig.add_trace(go.Scatter(
            x=MONEYNESS_GRID, y=iv_smile * 100,
            mode="lines", name=f"{t_days}d",
            line=dict(width=2.5, color=color),
            fill="tozeroy",
            fillcolor=hex_to_rgba(color, 0.08),
        ))

    smile_fig.update_layout(
        title=dict(text="VOL SMILE BY EXPIRY",
                   font=dict(size=10, color="#4facfe")),
        xaxis_title="Moneyness", yaxis_title="IV (%)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#99aabb", size=10),
        margin=dict(l=50, r=10, t=30, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)",
                    font=dict(size=10, color="#99aabb")),
        xaxis=dict(gridcolor="#1a1a2e", zerolinecolor="#1a1a2e",
                   tickfont=dict(size=10, color="#99aabb"),
                   title=dict(font=dict(size=11, color="#4facfe"))),
        yaxis=dict(gridcolor="#1a1a2e", zerolinecolor="#1a1a2e",
                   tickfont=dict(size=10, color="#99aabb"),
                   title=dict(font=dict(size=11, color="#4facfe"))),
        transition=dict(duration=400),
    )

    # ── Term structure chart ─────────────────────────────────────
    X_atm = torch.tensor(
        np.column_stack([
            np.full_like(TIME_GRID, 1.0),
            np.log(TIME_GRID),
        ]),
        dtype=torch.float32,
    )
    with torch.no_grad():
        atm_ivs = model(X_atm).numpy()
    atm_ivs = np.clip(atm_ivs + shock, 0.01, None)

    term_fig = go.Figure(go.Scatter(
        x=TIME_GRID * 365, y=atm_ivs * 100,
        mode="lines+markers",
        line=dict(color="#4facfe", width=2.5),
        marker=dict(size=3, color="#4facfe"),
        fill="tozeroy",
        fillcolor="rgba(79,172,254,0.15)",
    ))
    term_fig.update_layout(
        title=dict(text="ATM TERM STRUCTURE",
                   font=dict(size=10, color="#4facfe")),
        xaxis_title="Days to expiry", yaxis_title="IV (%)",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#99aabb", size=10),
        margin=dict(l=50, r=10, t=30, b=40),
        xaxis=dict(gridcolor="#1a1a2e", zerolinecolor="#1a1a2e",
                   tickfont=dict(size=10, color="#99aabb"),
                   title=dict(font=dict(size=11, color="#4facfe"))),
        yaxis=dict(gridcolor="#1a1a2e", zerolinecolor="#1a1a2e",
                   tickfont=dict(size=10, color="#99aabb"),
                   title=dict(font=dict(size=11, color="#4facfe"))),
        transition=dict(duration=400),
    )

    # Default Greeks at ATM 30d
    atm_iv_30 = float(np.clip(
        predict_surface(model, np.array([1.0]),
                        np.array([30 / 365]))[0][0] + shock, 0.01, None
    ))
    greeks = compute_greeks(S=100, K=100, T=30 / 365, r=0.045, iv=atm_iv_30)
    metrics = build_metrics_panel(greeks, checkpoint=checkpoint)

    return surface_fig, smile_fig, term_fig, metrics, ticker, shock


@app.callback(
    Output("metrics-panel", "children", allow_duplicate=True),
    Input("surface-plot", "clickData"),
    State("ticker-store", "data"),
    State("shock-store", "data"),
    prevent_initial_call=True
)
def update_greeks_on_click(clickData, ticker, shock):
    if clickData is None:
        return dash.no_update

    try:
        point = clickData["points"][0]
        moneyness = float(point["x"])
        days      = float(point["y"])
        T         = days / 365      # convert days to years correctly

        if T <= 0 or moneyness <= 0:
            return dash.no_update

        model, checkpoint = load_model(ticker)
        if model is None:
            return dash.no_update

        X = torch.tensor(
            [[moneyness, np.log(max(T, 0.001))]],
            dtype=torch.float32
        )
        with torch.no_grad():
            iv = float(model(X).numpy()[0]) + (shock or 0)
        iv = max(iv, 0.01)

        greeks = compute_greeks(
            S=100,
            K=100 * moneyness,
            T=T,              # already in years now
            r=0.045,
            iv=iv
        )

        return build_metrics_panel(
            greeks,
            moneyness=moneyness,
            days=days,
            checkpoint=checkpoint
        )

    except Exception as e:
        return dash.no_update


@app.callback(
    Output("shock-badge", "children"),
    Input("vol-shock", "value"),
)
def update_shock_badge(value):
    v = value or 0
    return f"{v:+d}%" if v != 0 else "0%"


@app.callback(
    Output("live-timestamp", "children"),
    Input("clock-interval", "n_intervals"),
)
def update_timestamp(n):
    return datetime.now().strftime("%Y-%m-%d  %H:%M:%S")


if __name__ == "__main__":
    app.run(debug=True)
