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
) -> None:
    """Run the pipeline end-to-end and write outputs to data/jobs/<job_id>/."""
    from dhara.jobs.pipeline import JobSpec, run_job

    spec = JobSpec(
        aoi=_aoi(bbox, geojson),
        event=parse_event(event),
        relative_orbit=orbit,
    )
    res = run_job(spec)
    typer.echo(f"job {res.job_id} → {res.job_dir}")
    typer.echo(json.dumps(res.summary, indent=2))


if __name__ == "__main__":
    app()
