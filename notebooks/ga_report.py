#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["google-analytics-data", "pandas", "plotly"]
# ///
"""Build an HTML usage report for muscriptor.kyutai.org out of GA4.

    export GOOGLE_APPLICATION_CREDENTIALS=~/.config/muscriptor-ga.json
    ./notebooks/ga_report.py --days 30 --out report.html

Set GA_PROPERTY_ID_MUSCRIPTOR to report on a different property.

Custom params must be registered in GA4 admin, and registration is not
retroactive: events from before it come back as "(not set)" and are dropped
here. Text params (format, file_type, instruments, detected_instruments,
is_example, status) go in as custom *dimensions*; numeric ones
(audio_duration_s, transcribe_time_s) as custom *metrics*, which GA4 sums — the
averages here are that sum over eventCount. A section whose param is missing
reports that instead of rendering.

`--self-check` runs the parsing asserts without touching the network.
"""

import argparse
import collections
import os
import sys
import webbrowser

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from google.analytics.data_v1beta import BetaAnalyticsDataClient
from google.analytics.data_v1beta.types import (
    DateRange,
    Dimension,
    Filter,
    FilterExpression,
    Metric,
    RunReportRequest,
)

# GA4 truncates event params at 100 characters. `instruments` is a sorted
# comma-joined list, so long selections lose their tail — and because the sort
# is alphabetical, the loss is biased towards the end of the alphabet.
TRUNCATION_MARK = "…"

MISSING = {"(not set)", ""}

# The muscriptor.kyutai.org GA4 property. Not a credential — reporting still
# needs GOOGLE_APPLICATION_CREDENTIALS for a service account with Viewer on it.
PROPERTY_ID = "545887727"


def report(client, prop, days, dimensions, event_name=None, metrics=("eventCount",)):
    """One GA4 report, optionally filtered to a single event, as a DataFrame."""
    req = RunReportRequest(
        property=f"properties/{prop}",
        date_ranges=[DateRange(start_date=f"{days}daysAgo", end_date="yesterday")],
        dimensions=[Dimension(name=d) for d in dimensions],
        metrics=[Metric(name=m) for m in metrics],
        limit=100000,
    )
    if event_name:
        req.dimension_filter = FilterExpression(
            filter=Filter(
                field_name="eventName",
                string_filter=Filter.StringFilter(value=event_name),
            )
        )
    resp = client.run_report(req)
    rows = [
        [v.value for v in r.dimension_values]
        # Custom metrics come back as decimals, standard ones as integers.
        + [float(v.value) for v in r.metric_values]
        for r in resp.rows
    ]
    df = pd.DataFrame(rows, columns=list(dimensions) + list(metrics))
    return df.rename(columns=lambda c: c.replace("customEvent:", ""))


def need(tables, name):
    """A fetched table, or an exception carrying why the fetch failed."""
    df = tables[name]
    if df.empty and df.attrs.get("error"):
        raise RuntimeError(df.attrs["error"])
    return df


def explode_instruments(df, column="instruments"):
    """Count how often each instrument appears across comma-joined selections.

    Rows cut off by the 100-character param limit end in a partial name, which
    is dropped rather than counted as its own instrument.
    """
    counts = collections.Counter()
    for value, n in zip(df[column], df["eventCount"]):
        value = str(value)
        if value in MISSING or value == "(none)":
            continue
        tokens = value.split(",")
        if value.endswith(TRUNCATION_MARK):
            tokens = tokens[:-1]
        for token in tokens:
            token = token.strip().rstrip(TRUNCATION_MARK)
            if token:
                counts[token] += n
    return (
        pd.Series(counts, dtype="int64")
        .rename_axis("instrument")
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )


def bar(df, x, y, title, subtitle):
    """Horizontal bar chart, biggest at the top."""
    fig = px.bar(df.sort_values(x), x=x, y=y, orientation="h", text=x)
    fig.update_traces(marker_color="#2a78d6", textposition="outside", cliponaxis=False)
    fig.update_layout(
        title=dict(text=title, subtitle=dict(text=subtitle)),
        height=max(320, 26 * len(df) + 140),
        xaxis_title=None,
        yaxis_title=None,
        margin=dict(l=10, r=60, t=90, b=40),
    )
    return fig


def fig_downloads(tables):
    df = need(tables, "downloads")
    df = df[~df["format"].isin(MISSING)].copy()
    # Only sheets_file splits further; for every other format file_type is empty.
    df["label"] = df.apply(
        lambda r: (
            f"{r['format']} ({r['file_type']})"
            if r["format"] == "sheets_file" and r["file_type"] not in MISSING
            else r["format"]
        ),
        axis=1,
    )
    df = df.groupby("label", as_index=False).eventCount.sum()
    return bar(
        df,
        "eventCount",
        "label",
        "What people download",
        "`sheets` is the dialog opening, so it overlaps the sheets_* rows below it.",
    )


def fig_instruments(tables):
    counts = explode_instruments(need(tables, "instruments"))
    return bar(
        counts,
        "count",
        "instrument",
        "What people pick in the instruments box",
        "Undercounts late-alphabet instruments: long selections are truncated at 100 chars.",
    )


def fig_picked_vs_detected(tables):
    """Where the model's own instrument detection disagrees with the user."""
    df = need(tables, "detected")
    picked = explode_instruments(df, "instruments").set_index("instrument")["count"]
    detected = explode_instruments(df, "detected_instruments").set_index("instrument")[
        "count"
    ]
    both = pd.concat(
        [picked.rename("picked"), detected.rename("detected")], axis=1
    ).fillna(0)
    both = both.sort_values("detected", ascending=True).reset_index()
    fig = go.Figure(
        [
            go.Bar(
                y=both.instrument,
                x=both.picked,
                name="picked by user",
                orientation="h",
                marker_color="#2a78d6",
            ),
            go.Bar(
                y=both.instrument,
                x=both.detected,
                name="detected by model",
                orientation="h",
                marker_color="#eb6834",
            ),
        ]
    )
    fig.update_layout(
        barmode="group",
        title=dict(
            text="Picked by the user vs detected by the model",
            subtitle=dict(
                text="Both counts share the same truncation bias, so read the gap, not the level."
            ),
        ),
        height=max(400, 34 * len(both) + 140),
        margin=dict(l=10, r=60, t=90, b=40),
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
    )
    return fig


def fig_funnel(tables):
    counts = need(tables, "events").set_index("eventName")["totalUsers"]
    steps = [
        ("Started a transcription", "transcription_start"),
        ("Finished it", "transcription_complete"),
        ("Downloaded something", "download"),
    ]
    fig = go.Figure(
        go.Funnel(
            y=[label for label, _ in steps],
            x=[int(counts.get(event, 0)) for _, event in steps],
            marker_color="#2a78d6",
        )
    )
    fig.update_layout(
        title=dict(
            text="From transcribe to download",
            subtitle=dict(
                text="Users per event, not a per-user path: someone who downloads "
                "without finishing still counts in both. A true funnel needs the "
                "BigQuery export."
            ),
        ),
        height=380,
        margin=dict(l=10, r=10, t=110, b=40),
    )
    return fig


HEALTH_EVENTS = {
    "transcription_start": "#2a78d6",
    "transcription_complete": "#1baf7a",
    "transcription_server_busy": "#eda100",
    "transcription_error": "#e34948",
}


def fig_daily(tables):
    """Daily health monitor: volume on top, the two failure modes underneath."""
    df = need(tables, "daily")
    df = df[df.eventName.isin(HEALTH_EVENTS)].copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    fig = px.line(
        df.sort_values("date"),
        x="date",
        y="eventCount",
        color="eventName",
        markers=True,
        color_discrete_map=HEALTH_EVENTS,
        category_orders={"eventName": list(HEALTH_EVENTS)},
    )
    fig.update_traces(line_width=2)
    fig.update_layout(
        title=dict(
            text="Activity and health per day",
            subtitle=dict(
                text="A busy server retries, so its line counts retries, not "
                "distinct waits."
            ),
        ),
        height=440,
        xaxis_title=None,
        yaxis_title=None,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.02, yanchor="bottom", title=None),
        margin=dict(l=10, r=10, t=110, b=40),
    )
    return fig


def fig_example(tables):
    """Example track vs a real upload.

    `is_example` is authoritative but only exists from the day it was registered
    as a custom dimension. The example filename carries no extension, so
    file_type == "unknown" is a proxy that works over the whole history — it
    also catches the rare genuine upload with no extension.
    """
    df = need(tables, "starts")
    known = df[~df.is_example.isin(MISSING)]
    if len(known):
        counts = known.groupby("is_example", as_index=False).eventCount.sum()
        counts["label"] = counts.is_example.map(
            lambda v: "Example track" if v == "true" else "Own upload"
        )
        subtitle = "From the is_example param."
    else:
        by_type = df.groupby("file_type", as_index=False).eventCount.sum()
        example = by_type[by_type.file_type == "unknown"].eventCount.sum()
        counts = pd.DataFrame(
            {
                "label": ["Example track", "Own upload"],
                "eventCount": [example, by_type.eventCount.sum() - example],
            }
        )
        subtitle = (
            "is_example is not registered — inferred from the missing file extension."
        )

    return bar(counts, "eventCount", "label", "Example track vs own upload", subtitle)


def fig_file_types(tables):
    df = need(tables, "starts")
    df = df[~df.file_type.isin(MISSING)]
    df = df.groupby("file_type", as_index=False).eventCount.sum()
    return bar(
        df,
        "eventCount",
        "file_type",
        "Uploaded file types",
        '"unknown" is the example track, which has no extension.',
    )


def fig_error_kinds(tables):
    df = need(tables, "errors").copy()
    df["status"] = df.status.map(
        lambda s: "no HTTP status (network / client)" if s in MISSING else f"HTTP {s}"
    )
    df = df.groupby("status", as_index=False).eventCount.sum()
    return bar(
        df,
        "eventCount",
        "status",
        "What the errors were",
        "No HTTP status means the request never got a response: a dropped "
        "connection, or the user leaving mid-transcription.",
    )


def fig_error_messages(tables):
    df = need(tables, "error_messages").copy()
    df["message"] = df.message.map(lambda s: "(none)" if s in MISSING else s)
    df = df.groupby("message", as_index=False).eventCount.sum()
    df = df.sort_values("eventCount", ascending=False).head(20)
    return bar(
        df,
        "eventCount",
        "message",
        "Error messages",
        "Raw Error.message, truncated at 100 chars by GA4. Top 20.",
    )


FIGURES = [
    fig_funnel,
    fig_downloads,
    fig_example,
    fig_file_types,
    fig_daily,
    fig_error_kinds,
    fig_error_messages,
    fig_instruments,
    fig_picked_vs_detected,
]


def mmss(seconds):
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def stat_tiles(tables):
    """Headline numbers, each skipped rather than faked when its data is missing."""
    tiles = []

    def add(value, label):
        tiles.append(
            f'<div class="tile"><div class="n">{value}</div>'
            f'<div class="l">{label}</div></div>'
        )

    try:
        done = need(tables, "completed").iloc[0]
        if done.eventCount:
            add(mmss(done.audio_duration_s / done.eventCount), "Average song length")
            add(
                mmss(done.transcribe_time_s / done.eventCount),
                "Average time to transcribe",
            )
    except Exception as e:
        tiles.append(
            f'<div class="tile warn">Average song length needs '
            f"<code>audio_duration_s</code> and <code>transcribe_time_s</code> "
            f"registered as custom <em>metrics</em> in GA4 — {e}</div>"
        )

    try:
        counts = need(tables, "events").set_index("eventName")["eventCount"]
        started = counts.get("transcription_start", 0)
        if started:
            add(f"{int(started):,}", "Transcriptions started")
            add(
                f"{100 * counts.get('transcription_complete', 0) / started:.0f}%",
                "Completed after starting",
            )
            add(
                f"{100 * counts.get('transcription_error', 0) / started:.1f}%",
                "Ended in an error",
            )
            add(
                f"{100 * counts.get('transcription_server_busy', 0) / started:.1f}%",
                "Hit a busy server (retries counted)",
            )
    except Exception as e:
        tiles.append(f'<div class="tile warn">Event counts unavailable — {e}</div>')

    return f'<div class="tiles">{"".join(tiles)}</div>'


PAGE = """<!doctype html>
<meta charset="utf-8">
<title>Muscriptor usage</title>
<style>
  body {{ font: 15px/1.5 ui-sans-serif, system-ui, sans-serif; max-width: 900px;
         margin: 0 auto; padding: 40px 20px 80px; color: #0b0b0b; }}
  h1 {{ margin-bottom: 4px; }}
  p.sub {{ color: #52514e; margin-top: 0; }}
  .chart {{ margin: 36px 0; }}
  .tiles {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 24px 0 8px; }}
  .tile {{ flex: 1 1 150px; background: #f2f2ef; border: 1px solid #e2e2dd;
          border-radius: 8px; padding: 14px 16px; }}
  .tile .n {{ font-size: 26px; font-weight: 600; font-variant-numeric: tabular-nums; }}
  .tile .l {{ font-size: 12px; color: #52514e; margin-top: 2px; }}
  .tile.warn {{ flex-basis: 100%; font-size: 12px; color: #52514e; }}
</style>
<h1>Muscriptor usage</h1>
<p class="sub">Last {days} full days (through yesterday), generated {stamp}.</p>
{stats}
{charts}
"""


def build(tables, days, stamp):
    charts = []
    for i, make in enumerate(FIGURES):
        try:
            fig = make(tables)
        except (
            Exception
        ) as e:  # a param nobody registered yet shouldn't kill the report
            charts.append(f"<p><em>{make.__name__} failed: {e}</em></p>")
            continue
        charts.append(
            '<div class="chart">'
            + pio.to_html(
                fig, full_html=False, include_plotlyjs="cdn" if i == 0 else False
            )
            + "</div>"
        )
    return PAGE.format(
        days=days, stamp=stamp, stats=stat_tiles(tables), charts="\n".join(charts)
    )


# name -> report() kwargs. Anything whose params are not registered in GA4 fails
# its own request; the sections that need it say so and the rest still render.
TABLES = {
    "downloads": dict(
        # Add "customEvent:sheet_kind" once it is registered in GA4 admin
        # and the frontend change shipped.
        dimensions=["customEvent:format", "customEvent:file_type"],
        event_name="download",
    ),
    "instruments": dict(
        dimensions=["customEvent:instruments"], event_name="transcription_start"
    ),
    "starts": dict(
        dimensions=["customEvent:is_example", "customEvent:file_type"],
        event_name="transcription_start",
    ),
    "detected": dict(
        dimensions=["customEvent:instruments", "customEvent:detected_instruments"],
        event_name="transcription_complete",
    ),
    "events": dict(dimensions=["eventName"], metrics=("eventCount", "totalUsers")),
    "daily": dict(dimensions=["date", "eventName"]),
    # Custom metrics, so these are sums: divide by eventCount for the average.
    "completed": dict(
        dimensions=[],
        event_name="transcription_complete",
        metrics=(
            "eventCount",
            "customEvent:audio_duration_s",
            "customEvent:transcribe_time_s",
        ),
    ),
    "errors": dict(dimensions=["customEvent:status"], event_name="transcription_error"),
    # Separate request so an unregistered `message` can't take `status` with it.
    "error_messages": dict(
        dimensions=["customEvent:message"], event_name="transcription_error"
    ),
}


def fetch(client, prop, days):
    tables = {}
    for name, kwargs in TABLES.items():
        try:
            tables[name] = report(client, prop, days, **kwargs)
        except Exception as e:
            df = pd.DataFrame()
            df.attrs["error"] = str(e).strip().splitlines()[0]
            tables[name] = df
            print(f"warning: {name}: {df.attrs['error']}", file=sys.stderr)
    return tables


def self_check():
    df = pd.DataFrame(
        {
            "instruments": [
                "drums,voice",
                "drums",
                "(none)",
                "(not set)",
                "acoustic_piano,cello,vio…",
            ],
            "eventCount": [10, 5, 99, 99, 3],
        }
    )
    got = explode_instruments(df).set_index("instrument")["count"].to_dict()
    assert got == {"drums": 15, "voice": 10, "acoustic_piano": 3, "cello": 3}, got
    print("self-check ok")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--out", default="ga_report.html")
    ap.add_argument("--csv-dir", help="also dump the raw GA4 tables here")
    ap.add_argument("--open", action="store_true", help="open the report when done")
    ap.add_argument(
        "--self-check", action="store_true", help="run parsing asserts only"
    )
    args = ap.parse_args()

    if args.self_check:
        return self_check()

    prop = os.environ.get("GA_PROPERTY_ID_MUSCRIPTOR", PROPERTY_ID)
    tables = fetch(BetaAnalyticsDataClient(), prop, args.days)
    if args.csv_dir:
        os.makedirs(args.csv_dir, exist_ok=True)
        for name, df in tables.items():
            df.to_csv(os.path.join(args.csv_dir, f"{name}.csv"), index=False)

    stamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    with open(args.out, "w") as f:
        f.write(build(tables, args.days, stamp))
    print(f"wrote {args.out}")
    if args.open:
        webbrowser.open(f"file://{os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
