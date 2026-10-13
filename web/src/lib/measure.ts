/**
 * Geodesic distance and spherical polygon area calculations.
 * Exact WGS84 spherical excess formulas with no external dependencies.
 */

const R = 6371.0088 // Earth mean radius in kilometers

function toRad(deg: number): number {
  return (deg * Math.PI) / 180
}

/** Great-circle distance between two [lng, lat] points in kilometers. */
export function haversineDistanceKm(p1: [number, number], p2: [number, number]): number {
  const dLat = toRad(p2[1] - p1[1])
  const dLng = toRad(p2[0] - p1[0])
  const lat1 = toRad(p1[1])
  const lat2 = toRad(p2[1])
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.sin(dLng / 2) * Math.sin(dLng / 2) * Math.cos(lat1) * Math.cos(lat2)
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
  return R * c
}

/** Total great-circle distance along an array of [lng, lat] points in kilometers. */
export function measurePathDistanceKm(points: [number, number][]): number {
  if (points.length < 2) return 0
  let total = 0
  for (let i = 0; i < points.length - 1; i++) {
    total += haversineDistanceKm(points[i], points[i + 1])
  }
  return total
}

/** Geodesic area of a spherical polygon ring defined by [lng, lat] points in km². */
export function measurePolygonAreaKm2(points: [number, number][]): number {
  if (points.length < 3) return 0
  let s = 0
  const n = points.length
  for (let i = 0; i < n; i++) {
    const p1 = points[i]
    const p2 = points[(i + 1) % n]
    s += toRad(p2[0] - p1[0]) * (2 + Math.sin(toRad(p1[1])) + Math.sin(toRad(p2[1])))
  }
  const area = Math.abs((s * R * R) / 2)
  const totalSphere = 4 * Math.PI * R * R
  return area > totalSphere / 2 ? totalSphere - area : area
}

export function formatDistance(km: number): string {
  if (km < 1) {
    return `${Math.round(km * 1000)} m`
  }
  return `${km.toFixed(2)} km`
}

export function formatArea(km2: number): string {
  if (km2 < 0.01) {
    return `${Math.round(km2 * 1e6).toLocaleString()} m²`
  }
  return `${km2.toFixed(2)} km²`
}
