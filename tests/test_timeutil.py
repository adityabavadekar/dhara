from datetime import UTC, date, datetime

import pytest
from pydantic import BaseModel, ValidationError

from dhara.adapters.ohsome import snapshot_date
from dhara.timeutil import EventTime, TimeError, event_day_utc, iso_utc, parse_event


def test_date_and_offsets():
    assert parse_event("2026-08-26") == date(2026, 8, 26)
    # 08:40 Nepal time (+05:45) = 02:55Z
    dt = parse_event("2026-08-26T08:40:00+05:45")
    assert isinstance(dt, datetime)
    assert dt == datetime(2026, 8, 26, 2, 55, tzinfo=UTC)
    assert iso_utc(dt) == "2026-08-26T02:55:00Z"
    assert parse_event("2026-08-26T02:55:00Z") == dt


def test_naive_rejected():
    with pytest.raises(TimeError):
        parse_event("2026-08-26T08:40:00")
    with pytest.raises(TimeError):
        iso_utc(datetime(2026, 8, 26))


def test_utc_day_rollover():
    # 01:00 at +05:45 on the 27th is still the 26th in UTC.
    ev = parse_event("2026-08-27T01:00:00+05:45")
    assert event_day_utc(ev) == date(2026, 8, 26)
    assert snapshot_date(ev) == date(2026, 7, 27)


def test_pydantic_field():
    class M(BaseModel):
        event: EventTime

    assert M(event="2026-08-26").event == date(2026, 8, 26)
    ev = M(event="2026-08-26T08:40+05:45").event
    assert isinstance(ev, datetime) and ev.tzinfo is not None
    with pytest.raises(ValidationError):
        M(event="2026-08-26T08:40")
