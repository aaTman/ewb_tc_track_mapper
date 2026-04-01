"""Panel web dashboard for the TC track and landfall viewer.

Run with:
    panel serve app.py --port 9999 --allow-websocket-origin='*'
"""

import json
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
    """Read a generated netcdf and return an interactive Folium map string.

    Track dots are hidden until a line is clicked. Clicking a forecast or
    observed line reveals its dots, opens a floating wind/pressure chart
    panel, and enables hover-on-dot → chart crosshair interaction.
    """
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

    def _make_style(c, w=2, op=0.7):
        return lambda f, _c=c, _w=w, _op=op: {
            "color": _c, "weight": _w, "opacity": _op,
        }

    def _make_highlight(c):
        return lambda f, _c=c: {"color": _c, "weight": 5, "opacity": 1.0}

    def _safe(v):
        """Convert numpy scalar to JSON-safe Python type."""
        return None if (v is None or np.isnan(v)) else float(v)

    # ── 1. Collect observed data ──────────────────────────────────────────
    obs_lats = ds["observed_lat"].values
    obs_lons = _norm_lon(ds["observed_lon"].values)
    obs_times = ds["observed_valid_time"].values
    obs_slps = ds["observed_slp"].values
    obs_winds = ds["observed_wind"].values
    valid = ~(np.isnan(obs_lats) | np.isnan(obs_lons))

    obs_json = {
        "lats":  obs_lats[valid].tolist(),
        "lons":  obs_lons[valid].tolist(),
        "times": [str(t)[:16] for t in obs_times[valid]],
        "slp":   [_safe(v) for v in obs_slps[valid]],
        "wind":  [_safe(v) for v in obs_winds[valid]],
    }

    # ── 2. Collect forecast data and build GeoJson lines ──────────────────
    tracks_json = {}
    if "detection_init_time" in ds:
        init_times = np.unique(ds["detection_init_time"].values)
        cmap = plt.get_cmap("rainbow", max(len(init_times), 1))
        for i, init_t in enumerate(init_times):
            rgba = cmap(i / max(len(init_times) - 1, 1))
            colour = matplotlib.colors.to_hex(rgba)
            label = str(init_t)[:16].replace("T", " ")

            mask = ds["detection_init_time"].values == init_t
            fc_lats = ds["detection_lat"].values[mask]
            fc_lons = _norm_lon(ds["detection_lon"].values[mask])
            fc_vtimes = ds["detection_valid_time"].values[mask]
            fc_slps = ds["detection_slp"].values[mask]
            fc_winds = ds["detection_wind"].values[mask]
            fc_valid = ~(np.isnan(fc_lats) | np.isnan(fc_lons))

            lonlat = [
                [float(lon), float(lat)]
                for lat, lon in zip(fc_lats[fc_valid], fc_lons[fc_valid])
            ]
            if not lonlat:
                continue

            tracks_json[label] = {
                "color": colour,
                "lats":  fc_lats[fc_valid].tolist(),
                "lons":  fc_lons[fc_valid].tolist(),
                "times": [str(t)[:16] for t in fc_vtimes[fc_valid]],
                "slp":   [_safe(v) for v in fc_slps[fc_valid]],
                "wind":  [_safe(v) for v in fc_winds[fc_valid]],
            }

            # Line only — dots are created by JS
            folium.GeoJson(
                {"type": "Feature",
                 "properties": {"ewb_init": label},
                 "geometry": {"type": "LineString", "coordinates": lonlat}},
                style_function=_make_style(colour),
                highlight_function=_make_highlight(colour),
                tooltip=f"Forecast init: {label} — click to inspect",
            ).add_to(m)

    # ── 3. Observed track GeoJson line ────────────────────────────────────
    obs_lonlat = [
        [float(lon), float(lat)]
        for lat, lon in zip(obs_lons[valid], obs_lats[valid])
    ]
    if obs_lonlat:
        folium.GeoJson(
            {"type": "Feature",
             "properties": {"ewb_obs": True},
             "geometry": {"type": "LineString", "coordinates": obs_lonlat}},
            style_function=_make_style("black", w=3, op=0.9),
            highlight_function=_make_highlight("black"),
            tooltip="IBTrACS observed — click to inspect",
        ).add_to(m)

        folium.Marker(
            [obs_lats[valid][0], obs_lons[valid][0]],
            tooltip="Storm start",
            icon=folium.Icon(color="black", icon="flag"),
        ).add_to(m)

    # ── 4. Landfall markers (always visible) ──────────────────────────────
    if "obs_landfall_lat" in ds:
        for lat, lon in zip(
            ds["obs_landfall_lat"].values,
            ds["obs_landfall_lon"].values,
        ):
            if not (np.isnan(lat) or np.isnan(lon)):
                folium.CircleMarker(
                    location=[float(lat), float(lon)],
                    radius=9, color="black",
                    fill=True, fill_color="black", fill_opacity=0.9,
                    tooltip="Observed landfall",
                ).add_to(m)

    if "fc_landfall_lat" in ds:
        for lat, lon in zip(
            ds["fc_landfall_lat"].values,
            ds["fc_landfall_lon"].values,
        ):
            if not (np.isnan(lat) or np.isnan(lon)):
                folium.CircleMarker(
                    location=[float(lat), float(lon)],
                    radius=7, color="red",
                    fill=True, fill_color="red", fill_opacity=0.85,
                    tooltip="Forecast landfall",
                ).add_to(m)

    # ── 5. Inject serialized data ─────────────────────────────────────────
    data_script = (
        "<script>\n"
        f"var EWB_TRACKS = {json.dumps(tracks_json)};\n"
        f"var EWB_OBS    = {json.dumps(obs_json)};\n"
        "</script>"
    )
    m.get_root().html.add_child(folium.Element(data_script))

    # ── 6. Chart.js CDN ───────────────────────────────────────────────────
    m.get_root().header.add_child(folium.Element(
        '<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0'
        '/dist/chart.umd.min.js"></script>'
    ))

    # ── 7. Floating chart panel HTML ──────────────────────────────────────
    panel_html = """
<div id="ewb-panel" style="
  display:none;position:absolute;top:10px;left:10px;z-index:9999;
  width:320px;background:white;border-radius:8px;
  box-shadow:0 2px 12px rgba(0,0,0,0.25);padding:14px 16px;
  font-family:sans-serif;font-size:12px;">
  <div style="display:flex;justify-content:space-between;
              align-items:center;margin-bottom:10px;">
    <b id="ewb-panel-title" style="font-size:13px;"></b>
    <span id="ewb-panel-close"
          style="cursor:pointer;font-size:18px;color:#888;line-height:1;">
      &#x2715;
    </span>
  </div>
  <canvas id="ewb-chart"></canvas>
</div>"""
    m.get_root().html.add_child(folium.Element(panel_html))

    # ── 8. Interactive JS logic ───────────────────────────────────────────
    js = """
<script>
(function () {
  "use strict";

  var ewbChart   = null;
  var dotLayers  = {};   // label -> L.LayerGroup (forecast dots)
  var obsDotLayer = null;
  var activeLabel = null;

  /* ── helpers ─────────────────────────────────────────────────────── */

  function getMap() {
    var keys = Object.keys(window);
    for (var i = 0; i < keys.length; i++) {
      var v = window[keys[i]];
      if (v && typeof v === "object" && v instanceof L.Map) return v;
    }
    return null;
  }

  function fmtVal(v, unit) {
    return (v != null && isFinite(v)) ? v.toFixed(1) + " " + unit : "\u2014";
  }

  function makeTip(title, time, slp, wind) {
    var slpHpa = (slp != null) ? slp / 100 : null;
    return "<b>" + title + "</b><br>Valid: " + time
      + "<br>SLP: "  + fmtVal(slpHpa, "hPa")
      + "<br>Wind: " + fmtVal(wind, "m/s");
  }

  /* ── dot layer creation ──────────────────────────────────────────── */

  function createDotLayers(leafMap) {
    /* Custom pane above Leaflet's default overlayPane (z 400) so dots
       always render on top of Python-added landfall CircleMarkers. */
    if (!leafMap.getPane("ewbDotPane")) {
      leafMap.createPane("ewbDotPane");
      leafMap.getPane("ewbDotPane").style.zIndex = 450;
      leafMap.getPane("ewbDotPane").style.pointerEvents = "auto";
    }

    /* forecast dots — hidden initially */
    Object.keys(EWB_TRACKS).forEach(function (label) {
      var d     = EWB_TRACKS[label];
      var group = L.layerGroup().addTo(leafMap);
      dotLayers[label] = group;

      d.lats.forEach(function (lat, idx) {
        var lon = d.lons[idx];
        if (lat == null || lon == null) return;
        var mk = L.circleMarker([lat, lon], {
          radius: 5, color: d.color, fillColor: d.color,
          fillOpacity: 0, opacity: 0, weight: 1,
          pane: "ewbDotPane",
        });
        mk.bindTooltip(makeTip("Init: " + label, d.times[idx],
                               d.slp[idx], d.wind[idx]),
                       { sticky: true });
        mk.on("mouseover", (function (lbl, i) {
          return function () { highlightChartPoint(lbl, i); };
        }(label, idx)));
        mk.on("mouseout", clearChartHighlight);
        mk.addTo(group);
      });
    });

    /* observed dots — always shown at low opacity */
    obsDotLayer = L.layerGroup().addTo(leafMap);
    EWB_OBS.lats.forEach(function (lat, idx) {
      var lon = EWB_OBS.lons[idx];
      if (lat == null || lon == null) return;
      var mk = L.circleMarker([lat, lon], {
        radius: 4, color: "#333", fillColor: "#333",
        fillOpacity: 0.45, opacity: 0.65, weight: 1,
        pane: "ewbDotPane",
      });
      mk.bindTooltip(makeTip("IBTrACS", EWB_OBS.times[idx],
                             EWB_OBS.slp[idx], EWB_OBS.wind[idx]),
                     { sticky: true });
      mk.on("mouseover", (function (i) {
        return function () { highlightObsPoint(i); };
      }(idx)));
      mk.on("mouseout", clearChartHighlight);
      mk.addTo(obsDotLayer);
    });
  }

  /* ── dot visibility ──────────────────────────────────────────────── */

  function showDots(label) {
    Object.keys(dotLayers).forEach(function (lbl) {
      dotLayers[lbl].eachLayer(function (mk) {
        mk.setStyle({ fillOpacity: 0, opacity: 0 });
        mk.setRadius(5);
      });
    });
    if (label && dotLayers[label]) {
      dotLayers[label].eachLayer(function (mk) {
        mk.setStyle({ fillOpacity: 0.9, opacity: 1 });
        mk.setRadius(7);
      });
    }
  }

  /* ── chart rendering ─────────────────────────────────────────────── */

  var CHART_OPTS = {
    responsive: true,
    animation: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: {
        position: "bottom",
        labels: { boxWidth: 12, font: { size: 11 } },
      },
    },
    scales: {
      x: {
        ticks: { maxTicksLimit: 5, maxRotation: 30,
                 font: { size: 10 } },
      },
      y: {
        position: "left",
        title: { display: true, text: "Wind (m/s)",
                 font: { size: 10 } },
        ticks: { font: { size: 10 } },
      },
      y2: {
        position: "right",
        title: { display: true, text: "SLP (hPa)",
                 font: { size: 10 } },
        ticks: { font: { size: 10 } },
        grid: { drawOnChartArea: false },
      },
    },
  };

  function buildDatasets(times, wind, slp, colour) {
    var slpHpa = slp.map(function (v) {
      return (v != null) ? v / 100 : null;
    });
    return {
      labels: times,
      datasets: [
        {
          label: "Wind (m/s)", data: wind,
          borderColor: "#2196F3",
          backgroundColor: "rgba(33,150,243,0.08)",
          yAxisID: "y", tension: 0.3, pointRadius: 3,
          borderWidth: 2, fill: false,
        },
        {
          label: "SLP (hPa)", data: slpHpa,
          borderColor: colour || "#555",
          backgroundColor: "rgba(80,80,80,0.06)",
          yAxisID: "y2", tension: 0.3, pointRadius: 3,
          borderWidth: 2, fill: false,
        },
      ],
    };
  }

  function openChart(title, times, wind, slp, colour) {
    document.getElementById("ewb-panel-title").textContent = title;
    document.getElementById("ewb-panel").style.display = "block";
    if (ewbChart) { ewbChart.destroy(); ewbChart = null; }
    ewbChart = new Chart(
      document.getElementById("ewb-chart"),
      { type: "line",
        data: buildDatasets(times, wind, slp, colour),
        options: CHART_OPTS }
    );
  }

  /* ── track selection ─────────────────────────────────────────────── */

  function selectTrack(label) {
    activeLabel = label;
    showDots(label);
    var d = EWB_TRACKS[label];
    openChart("Init: " + label, d.times, d.wind, d.slp, d.color);
  }

  function selectObsTrack() {
    activeLabel = "__obs__";
    showDots(null);
    openChart("IBTrACS Observed",
              EWB_OBS.times, EWB_OBS.wind, EWB_OBS.slp, "#333");
  }

  /* ── chart crosshair on dot hover ────────────────────────────────── */

  function highlightChartPoint(label, idx) {
    if (!ewbChart || label !== activeLabel) return;
    ewbChart.tooltip.setActiveElements(
      [{ datasetIndex: 0, index: idx }, { datasetIndex: 1, index: idx }],
      { x: 0, y: 0 }
    );
    ewbChart.update("none");
  }

  function highlightObsPoint(idx) {
    if (!ewbChart || activeLabel !== "__obs__") return;
    ewbChart.tooltip.setActiveElements(
      [{ datasetIndex: 0, index: idx }, { datasetIndex: 1, index: idx }],
      { x: 0, y: 0 }
    );
    ewbChart.update("none");
  }

  function clearChartHighlight() {
    if (!ewbChart) return;
    ewbChart.tooltip.setActiveElements([], {});
    ewbChart.update("none");
  }

  /* ── wiring GeoJson click events ─────────────────────────────────── */

  function wireClicks(leafMap) {
    leafMap.eachLayer(function (layer) {
      if (typeof layer.eachLayer !== "function") return;
      layer.eachLayer(function (sub) {
        if (!sub.feature || !sub.feature.properties) return;
        var props = sub.feature.properties;
        if (props.ewb_init) {
          var label = props.ewb_init;
          sub.on("click", function () { selectTrack(label); });
        } else if (props.ewb_obs) {
          sub.on("click", function () { selectObsTrack(); });
        }
      });
    });
  }

  /* ── close button ────────────────────────────────────────────────── */

  function wireClose() {
    var btn = document.getElementById("ewb-panel-close");
    if (btn) {
      btn.addEventListener("click", function () {
        document.getElementById("ewb-panel").style.display = "none";
        showDots(null);
        activeLabel = null;
      });
    }
  }

  /* ── init (wait for map) ─────────────────────────────────────────── */

  function init() {
    var leafMap = getMap();
    if (!leafMap) { setTimeout(init, 200); return; }
    createDotLayers(leafMap);
    wireClicks(leafMap);
    wireClose();
  }

  if (document.readyState === "complete") {
    init();
  } else {
    window.addEventListener("load", init);
  }
}());
</script>"""
    m.get_root().html.add_child(folium.Element(js))

    # ── 9. Legend ─────────────────────────────────────────────────────────
    legend_items = [
        '<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">'
        '<svg width="28" height="6"><line x1="0" y1="3" x2="28" y2="3" '
        'stroke="black" stroke-width="3"/></svg>'
        '<span>IBTrACS observed</span></div>',
    ]
    if "obs_landfall_lat" in ds:
        legend_items.append(
            '<div style="display:flex;align-items:center;gap:6px;'
            'margin-bottom:4px">'
            '<svg width="16" height="16"><circle cx="8" cy="8" r="7" '
            'fill="black"/></svg>'
            '<span>Observed landfall</span></div>'
        )
    if tracks_json:
        for label, td in tracks_json.items():
            c = td["color"]
            legend_items.append(
                f'<div style="display:flex;align-items:center;gap:6px;'
                f'margin-bottom:2px">'
                f'<svg width="28" height="6"><line x1="0" y1="3" x2="28" y2="3"'
                f' stroke="{c}" stroke-width="2.5"/></svg>'
                f'<span style="font-size:11px">{label}</span></div>'
            )
    if "fc_landfall_lat" in ds:
        legend_items.append(
            '<div style="display:flex;align-items:center;gap:6px;margin-top:2px">'
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
