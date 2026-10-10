"""Time handling. Every timestamp is ISO 8601 and timezone-aware.

- Inputs accept an ISO 8601 date (`2026-08-26`, time of event unknown) or an ISO 8601 datetime
  with an explicit offset (`2026-08-26T08:40:00+05:45`, `2026-08-26T02:55:00Z`).
  A naive datetime (no offset) is rejected.
- Internally, and in every output, datetimes are UTC and written as `YYYY-MM-DDTHH:MM:SSZ`.
"""

from datetime import UTC, date, datetime
from typing import Annotated

from pydantic import AwareDatetime, BeforeValidator


class TimeError(ValueError):
    pass


def parse_event(value: str | date | datetime) -> date | datetime:
    """Parse an event time: a date, or an aware datetime (normalised to UTC)."""
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, date):
        return value
    else:
        s = value.strip()
        if len(s) == 10:
            return date.fromisoformat(s)
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError as e:
            raise TimeError(f"not ISO 8601: {value!r}") from e
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise TimeError(
            f"datetime {value!r} has no timezone; use e.g. 2026-08-26T08:40:00+05:45 or ...Z"
        )
    return dt.astimezone(UTC)


# Pydantic field type: `date` or UTC-normalised aware datetime.
EventTime = Annotated[date | AwareDatetime, BeforeValidator(parse_event)]


def event_day_utc(ev: date | datetime) -> date:
    """Calendar day (UTC) of the event."""
    return ev.astimezone(UTC).date() if isinstance(ev, datetime) else ev


def iso_utc(dt: datetime) -> str:
    """Aware datetime → `YYYY-MM-DDTHH:MM:SSZ`. Naive datetimes are refused."""
    if dt.tzinfo is None:
        raise TimeError(f"naive datetime {dt!r}")
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def now_utc() -> str:
    return iso_utc(datetime.now(UTC))
