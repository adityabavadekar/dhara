"""Command line entry point: `dhara scenes ...` and `dhara run ...`."""

import json
from typing import Annotated

import typer

from dhara.geo import AOI
from dhara.timeutil import parse_event

app = typer.Typer(no_args_is_help=True, help="Dhara flood damage pipeline (educational prototype).")


def _aoi(bbox: str | None, geojson: str | None) -> AOI:
    if geojson:
        with open(geojson) as f:
            g = json.load(f)
        g = g.get("geometry", g) if g.get("type") == "Feature" else g
        if g.get("type") == "FeatureCollection":
            g = g["features"][0]["geometry"]
        return AOI(geojson=g)
    if bbox:
        minx, miny, maxx, maxy = (float(v) for v in bbox.split(","))
        return AOI.from_bbox(minx, miny, maxx, maxy)
    raise typer.BadParameter("give --bbox or --geojson")


BBox = Annotated[str | None, typer.Option(help="minlon,minlat,maxlon,maxlat (EPSG:4326)")]
GeoJSON = Annotated[str | None, typer.Option(help="Path to a GeoJSON polygon (EPSG:4326)")]
Event = Annotated[
    str,
    typer.Option(
        "--event",
        help="ISO 8601 event date (2026-08-26) or datetime with offset (2026-08-26T08:40:00+05:45)",
    ),
]


@app.command()
def scenes(
    event: Event,
    bbox: BBox = None,
    geojson: GeoJSON = None,
) -> None:
    """List same-orbit Sentinel-1 pre/post pairs for an AOI and event date."""
    from dhara.jobs.scene_selection import find_s1_pairs

    pairs, notes = find_s1_pairs(_aoi(bbox, geojson), parse_event(event))
    for p in pairs:
        typer.echo(
            f"R{p.relative_orbit:03d} {p.orbit_state:<10} pre {p.pre.key}  post {p.post.key}  "
            f"coverage {min(p.pre.aoi_coverage, p.post.aoi_coverage):.0%}"
        )
    for n in notes:
        typer.echo(f"note: {n}")


@app.command()
def run(
    event: Event,
    bbox: BBox = None,
    geojson: GeoJSON = None,
    orbit: Annotated[int | None, typer.Option(help="Force a relative orbit")] = None,
    s2: Annotated[bool, typer.Option(help="Use Sentinel-2 when clear")] = True,
    ml: Annotated[bool, typer.Option(help="Use the segmentation model")] = True,
    s2_max_post_days: Annotated[
        int, typer.Option(help="Latest Sentinel-2 'after' image, days after the event")
    ] = 20,
    s1_source: Annotated[
        str, typer.Option(help="Sentinel-1 processing: grd (own chain) or sentinelhub")
    ] = "grd",
) -> None:
    """Run the pipeline end-to-end and write outputs to data/jobs/<job_id>/."""
    from dhara.jobs.pipeline import JobSpec, run_job

    if s1_source not in ("grd", "sentinelhub"):
        raise typer.BadParameter("--s1-source must be grd or sentinelhub")
    spec = JobSpec(
        aoi=_aoi(bbox, geojson),
        event=parse_event(event),
        relative_orbit=orbit,
        use_s2=s2,
        use_ml=ml,
        s2_max_post_days=s2_max_post_days,
        s1_source="sentinelhub" if s1_source == "sentinelhub" else "grd",
    )
    res = run_job(spec)
    typer.echo(f"job {res.job_id} → {res.job_dir}")
    typer.echo(json.dumps(res.summary, indent=2))


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Bind host")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Bind port")] = 8000,
    reload: Annotated[bool, typer.Option(help="Enable hot reload")] = False,
) -> None:
    """Start the Dhara FastAPI backend server."""
    import uvicorn

    typer.echo(f"Starting Dhara API on http://{host}:{port}")
    uvicorn.run("dhara.api.app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
