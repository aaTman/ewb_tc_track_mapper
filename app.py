"""Panel web dashboard for the TC track and landfall viewer.

Run with:
    panel serve app.py --port 9999 --allow-websocket-origin='*'
"""

import pathlib
import threading

import folium
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import panel as pn
import xarray as xr

from generate import ALL_MODELS, DEFAULT_OUTPUT_DIR, generate

import extremeweatherbench as ewb

pn.extension(
    sizing_mode="stretch_width",
    raw_css=[
        """
        html, body { overflow-y: hidden !important; }
        /* make Bokeh's HTML-pane div fill its allocated height */
        .bk-panel-models-markup-HTML > div { height: 100% !important; }
        """
    ],
)

# ── helpers ───────────────────────────────────────────────────────────────────

def _load_tc_cases():
    all_cases = ewb.load_cases()
    return sorted(
        [c for c in all_cases if c.event_type == "tropical_cyclone"],
        key=lambda c: c.case_id_number,
    )


def _nc_path(case_id: int, model: str) -> pathlib.Path:
    return DEFAULT_OUTPUT_DIR / model / f"case_{case_id:03d}.nc"


# ── map builder ───────────────────────────────────────────────────────────────

def _norm_lon(lons):
    """Convert any longitude array to ±180 convention."""
    return ((np.asarray(lons) + 180) % 360) - 180


def _build_map(nc_path: pathlib.Path) -> str:
    """Read a generated netcdf and return Folium map HTML string."""
    ds = xr.open_dataset(nc_path, decode_timedelta=False)

    center_lat = float(np.nanmean(ds["observed_lat"].values))
    center_lon = float(np.nanmean(_norm_lon(ds["observed_lon"].values)))

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=5,
        tiles="CartoDB positron",
        width="100%",
        height="100%",
    )
    m.get_root().width = "100%"
    m.get_root().height = "100%"

    # ── observed track ────────────────────────────────────────────────────
    obs_lats = ds["observed_lat"].values
    obs_lons = _norm_lon(ds["observed_lon"].values)
    valid = ~(np.isnan(obs_lats) | np.isnan(obs_lons))
    obs_coords = list(zip(obs_lats[valid].tolist(), obs_lons[valid].tolist()))
    if obs_coords:
        folium.PolyLine(
            obs_coords,
            color="black",
            weight=3,
            tooltip="IBTrACS observed track",
        ).add_to(m)
        folium.Marker(
            obs_coords[0],
            tooltip="Storm start",
            icon=folium.Icon(color="black", icon="flag"),
        ).add_to(m)

    # ── observed landfalls ────────────────────────────────────────────────
    if "obs_landfall_lat" in ds:
        for lat, lon in zip(
            ds["obs_landfall_lat"].values,
            ds["obs_landfall_lon"].values,
        ):
            if not (np.isnan(lat) or np.isnan(lon)):
                folium.CircleMarker(
                    location=[float(lat), float(lon)],
                    radius=8,
                    color="black",
                    fill=True,
                    fill_color="black",
                    fill_opacity=0.9,
                    tooltip="Observed landfall",
                ).add_to(m)

    # ── forecast tracks (one colour per unique init_time) ─────────────────
    if "detection_init_time" in ds:
        init_times = np.unique(ds["detection_init_time"].values)
        cmap = plt.get_cmap("rainbow", max(len(init_times), 1))
        for i, init_t in enumerate(init_times):
            rgba = cmap(i / max(len(init_times) - 1, 1))
            colour = matplotlib.colors.to_hex(rgba)
            mask = ds["detection_init_time"].values == init_t
            fc_lats = ds["detection_lat"].values[mask]
            fc_lons = _norm_lon(ds["detection_lon"].values[mask])
            fc_valid = ~(np.isnan(fc_lats) | np.isnan(fc_lons))
            coords = list(
                zip(fc_lats[fc_valid].tolist(), fc_lons[fc_valid].tolist())
            )
            if coords:
                label = str(init_t)[:16]
                folium.PolyLine(
                    coords,
                    color=colour,
                    weight=2,
                    opacity=0.75,
                    tooltip=f"Forecast init {label}",
                ).add_to(m)

    # ── forecast landfalls ────────────────────────────────────────────────
    if "fc_landfall_lat" in ds:
        for lat, lon in zip(
            ds["fc_landfall_lat"].values,
            ds["fc_landfall_lon"].values,
        ):
            if not (np.isnan(lat) or np.isnan(lon)):
                folium.CircleMarker(
                    location=[float(lat), float(lon)],
                    radius=6,
                    color="red",
                    fill=True,
                    fill_color="red",
                    fill_opacity=0.8,
                    tooltip="Forecast landfall",
                ).add_to(m)

    # ── legend ────────────────────────────────────────────────────────────
    legend_items = []
    legend_items.append(
        '<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">'
        '<svg width="28" height="6"><line x1="0" y1="3" x2="28" y2="3" '
        'stroke="black" stroke-width="3"/></svg>'
        '<span>IBTrACS observed</span></div>'
    )
    if "obs_landfall_lat" in ds:
        legend_items.append(
            '<div style="display:flex;align-items:center;gap:6px;'
            'margin-bottom:4px">'
            '<svg width="16" height="16"><circle cx="8" cy="8" r="7" '
            'fill="black"/></svg>'
            '<span>Observed landfall</span></div>'
        )
    if "detection_init_time" in ds:
        init_times = np.unique(ds["detection_init_time"].values)
        cmap2 = plt.get_cmap("rainbow", max(len(init_times), 1))
        for i, init_t in enumerate(init_times):
            rgba = cmap2(i / max(len(init_times) - 1, 1))
            colour = matplotlib.colors.to_hex(rgba)
            label = str(init_t)[:16].replace("T", " ")
            legend_items.append(
                f'<div style="display:flex;align-items:center;gap:6px;'
                f'margin-bottom:2px">'
                f'<svg width="28" height="6"><line x1="0" y1="3" x2="28" y2="3" '
                f'stroke="{colour}" stroke-width="2.5"/></svg>'
                f'<span style="font-size:11px">{label}</span></div>'
            )
    if "fc_landfall_lat" in ds:
        legend_items.append(
            '<div style="display:flex;align-items:center;gap:6px;'
            'margin-top:2px">'
            '<svg width="16" height="16"><circle cx="8" cy="8" r="6" '
            'fill="red" fill-opacity="0.8"/></svg>'
            '<span>Forecast landfall</span></div>'
        )

    legend_html = (
        '<div style="position:absolute;bottom:30px;right:10px;z-index:9999;'
        'background:rgba(255,255,255,0.92);padding:10px 14px;'
        'border-radius:6px;box-shadow:0 1px 5px rgba(0,0,0,0.3);'
        'font-family:sans-serif;font-size:12px;max-height:60vh;'
        'overflow-y:auto;pointer-events:none;">'
        "<b>Legend</b><hr style='margin:4px 0'>"
        + "".join(legend_items)
        + "</div>"
    )
    m.get_root().html.add_child(folium.Element(legend_html))

    ds.close()

    iframe_html = m._repr_html_()
    return (
        '<div style="width:100%;height:100%;">'
        + iframe_html
        + "</div>"
    )


# ── widgets ───────────────────────────────────────────────────────────────────

_tc_cases = _load_tc_cases()
_case_options = {
    f"{c.case_id_number}: {c.title}": c.case_id_number for c in _tc_cases
}

case_select = pn.widgets.Select(
    name="Case",
    options=list(_case_options.keys()),
    width=320,
)
model_select = pn.widgets.Select(
    name="Model",
    options=ALL_MODELS,
    width=200,
)
view_btn = pn.widgets.Button(
    name="View",
    button_type="primary",
    width=90,
)
generate_btn = pn.widgets.Button(
    name="Generate & View",
    button_type="success",
    width=130,
)
status_bar = pn.pane.Str(
    "",
    styles={
        "color": "#555",
        "font-size": "13px",
        "padding": "2px 0",
    },
    margin=(0, 4),
)


def _data_indicator(case_label, model):
    case_id = _case_options[case_label]
    exists = _nc_path(case_id, model).exists()
    if exists:
        html = (
            '<span title="Data available" style="'
            "display:inline-flex;align-items:center;gap:5px;"
            "font-size:13px;color:#2d7a2d;font-family:sans-serif;"
            '">'
            '<span style="font-size:16px;">&#x2705;</span> available'
            "</span>"
        )
    else:
        html = (
            '<span title="Not yet generated" style="'
            "display:inline-flex;align-items:center;gap:5px;"
            "font-size:13px;color:#888;font-family:sans-serif;"
            '">'
            '<span style="font-size:16px;">&#x26AA;</span> not generated'
            "</span>"
        )
    return pn.pane.HTML(html, width=150, height=30)


data_indicator = pn.bind(
    _data_indicator,
    case_label=case_select,
    model=model_select,
)

_NO_DATA_HTML = """
<div style="display:flex;align-items:center;justify-content:center;
            width:100%;height:100%;
            color:#888;font-family:sans-serif;font-size:16px;">
  No data loaded &mdash; select a case &amp; model, then click <b>View</b>
  or <b>Generate &amp; View</b>.
</div>
"""

map_pane = pn.pane.HTML(
    _NO_DATA_HTML,
    sizing_mode="stretch_width",
    height=560,
)


# ── callback helpers ──────────────────────────────────────────────────────────

def _current_case_id() -> int:
    return _case_options[case_select.value]


def _set_status(msg: str) -> None:
    status_bar.object = msg


def _load_and_display(nc_path: pathlib.Path) -> None:
    try:
        _set_status("Rendering map…")
        html = _build_map(nc_path)
        map_pane.object = html
        _set_status(f"Loaded: {nc_path.name}")
    except Exception as exc:
        _set_status(f"Error rendering map: {exc}")


# ── button callbacks ──────────────────────────────────────────────────────────

def _on_view(event):
    case_id = _current_case_id()
    model = model_select.value
    nc_path = _nc_path(case_id, model)
    if not nc_path.exists():
        _set_status(
            f"No data for case {case_id} / {model}. "
            "Use 'Generate & View' to create it."
        )
        return
    _load_and_display(nc_path)


def _on_generate(event):
    case_id = _current_case_id()
    model = model_select.value

    generate_btn.disabled = True
    view_btn.disabled = True
    _set_status(f"Generating case {case_id} / {model}… (this may take minutes)")

    def _worker():
        try:
            nc_path = generate(case_id=case_id, model_name=model)
            _load_and_display(nc_path)
            # nudge the indicator to refresh
            case_select.param.trigger("value")
        except Exception as exc:
            _set_status(f"Generation failed: {exc}")
        finally:
            generate_btn.disabled = False
            view_btn.disabled = False

    threading.Thread(target=_worker, daemon=True).start()


view_btn.on_click(_on_view)
generate_btn.on_click(_on_generate)


# ── layout ────────────────────────────────────────────────────────────────────

_title = pn.pane.Markdown(
    "## TC Track & Landfall Viewer",
    margin=(4, 4, 0, 4),
)
_controls = pn.Row(
    _title,
    pn.Spacer(width=20),
    case_select,
    model_select,
    data_indicator,
    view_btn,
    generate_btn,
    align="center",
    sizing_mode="stretch_width",
    margin=(4, 4),
)

dashboard = pn.Column(
    _controls,
    status_bar,
    map_pane,
    sizing_mode="stretch_width",
    margin=(4, 10),
)

dashboard.servable()
