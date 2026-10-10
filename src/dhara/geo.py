"""AOI model and CRS helpers."""

import hashlib
import json
from typing import Any

import shapely
from pydantic import BaseModel, field_validator
from pyproj import CRS, Geod, Transformer
from shapely import box, from_geojson, to_geojson
from shapely.geometry.base import BaseGeometry

from dhara.config import get_settings

WGS84 = CRS.from_epsg(4326)
_GEOD = Geod(ellps="WGS84")


class AOIError(ValueError):
    pass


class AOI(BaseModel):
    """Area of interest. CRS: EPSG:4326 (lon/lat). Stored as GeoJSON geometry."""

    geojson: dict[str, Any]

    @field_validator("geojson")
    @classmethod
    def _valid(cls, v: dict[str, Any]) -> dict[str, Any]:
        geom = from_geojson(json.dumps(v))
        if geom.geom_type not in ("Polygon", "MultiPolygon"):
            raise AOIError(f"AOI must be a Polygon/MultiPolygon, got {geom.geom_type}")
        if not geom.is_valid or geom.is_empty:
            raise AOIError("AOI geometry is invalid or empty")
        minx, miny, maxx, maxy = geom.bounds
        if not (-180 <= minx <= maxx <= 180 and -90 <= miny <= maxy <= 90):
            raise AOIError("AOI must be in EPSG:4326 lon/lat degrees")
        return v

    @classmethod
    def from_bbox(cls, minx: float, miny: float, maxx: float, maxy: float) -> "AOI":
        """bbox in EPSG:4326 (lon/lat)."""
        return cls(geojson=json.loads(to_geojson(box(minx, miny, maxx, maxy))))

    @property
    def geometry(self) -> BaseGeometry:
        """Shapely geometry, EPSG:4326."""
        return from_geojson(json.dumps(self.geojson))

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """(minx, miny, maxx, maxy) in EPSG:4326."""
        return self.geometry.bounds

    @property
    def area_km2(self) -> float:
        """Geodesic area on the WGS84 ellipsoid, km²."""
        area, _ = _GEOD.geometry_area_perimeter(self.geometry)
        return abs(area) / 1e6

    @property
    def hash(self) -> str:
        """Stable short hash of the normalised geometry (cache key component)."""
        wkb = self.geometry.normalize().wkb
        return hashlib.sha256(wkb).hexdigest()[:16]

    def utm_crs(self) -> CRS:
        """Local UTM zone (WGS84) for analysis, chosen from the AOI centroid."""
        c = self.geometry.centroid
        zone = int((c.x + 180) // 6) + 1
        return CRS.from_epsg((32600 if c.y >= 0 else 32700) + zone)

    def check_size(self) -> None:
        cap = get_settings().dhara_max_aoi_km2
        if self.area_km2 > cap:
            raise AOIError(
                f"AOI is {self.area_km2:.0f} km², above the live-run cap of {cap:.0f} km². "
                "Draw a smaller area."
            )


def reproject(geom: BaseGeometry, src: CRS | str, dst: CRS | str) -> BaseGeometry:
    """Reproject a shapely geometry from `src` to `dst` (always_xy axis order)."""
    tr = Transformer.from_crs(src, dst, always_xy=True)
    return shapely.transform(
        geom, lambda xy: list(zip(*tr.transform(xy[:, 0], xy[:, 1]), strict=True))
    )
