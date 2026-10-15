"""Tests for the grounded assistant: typed job queries, off-topic refusal, template answers."""

from pathlib import Path

import pytest

from dhara.assistant.service import (
    OFF_TOPIC_FALLBACK,
    _execute_gis_tool,
    _is_obviously_off_topic,
    ask_assistant,
)


@pytest.fixture(autouse=True)
def _no_llm(monkeypatch):
    """Tests never call a language model API."""
    from dhara.config import get_settings

    monkeypatch.setenv("ASSISTANT_LLM", "off")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _no_language_model(monkeypatch):
    """Deterministic answers: the template path only, no network call to a model provider."""
    from dhara.config import get_settings

    monkeypatch.setenv("ASSISTANT_LLM", "off")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def sample_job_dir():
    d = Path("data/jobs/1ca3bdd9fba8")
    if d.exists():
        return d
    pytest.skip("Sample job 1ca3bdd9fba8 not found in data/jobs")


def test_gis_tools_execution(sample_job_dir):
    # 1. query_kpis
    kpis = _execute_gis_tool(sample_job_dir, "query_kpis", {})
    assert kpis["status"] == "success"
    assert "kpis" in kpis
    assert "settlements_cut_off" in kpis["kpis"] or "aoi_area_km2" in kpis["kpis"]

    # 2. query_settlements
    settlements = _execute_gis_tool(sample_job_dir, "query_settlements", {"limit": 5})
    assert settlements["status"] == "success"
    assert "settlements" in settlements
    assert len(settlements["settlements"]) <= 5

    # 3. query_bridges
    bridges = _execute_gis_tool(sample_job_dir, "query_bridges", {"limit": 5})
    assert bridges["status"] == "success"
    assert "bridges" in bridges

    # 4. query_health_facilities
    hospitals = _execute_gis_tool(sample_job_dir, "query_health_facilities", {"limit": 5})
    assert hospitals["status"] == "success"
    assert "facilities" in hospitals

    # 5. query_restoration_priorities
    repairs = _execute_gis_tool(sample_job_dir, "query_restoration_priorities", {"limit": 3})
    assert repairs["status"] == "success"
    assert "priorities" in repairs

    # 6. spatial_nearest_features
    near = _execute_gis_tool(
        sample_job_dir,
        "spatial_nearest_features",
        {"lat": 28.1, "lon": 85.3, "limit": 3},
    )
    assert near["status"] == "success"
    assert "nearest" in near


def test_gis_tool_reports_full_status_counts_when_examples_are_limited(sample_job_dir):
    result = _execute_gis_tool(sample_job_dir, "query_settlements", {"status": "all", "limit": 2})
    assert result["count"] > result["returned_count"] == 2
    assert result["truncated"] is True
    assert sum(result["status_counts"].values()) == result["count"]


def test_off_topic_detector():
    assert _is_obviously_off_topic("write a poem about cats")
    assert _is_obviously_off_topic("Who is Batman?")
    assert _is_obviously_off_topic("What is the capital of France?")
    assert _is_obviously_off_topic("how to bake sourdough bread")
    assert _is_obviously_off_topic("tell me a joke")
    assert _is_obviously_off_topic("are you ai")
    assert _is_obviously_off_topic("Are you an AI?")
    assert _is_obviously_off_topic("what are you")
    assert _is_obviously_off_topic("who made you")

    # Disaster questions must NOT be flagged as off-topic
    assert not _is_obviously_off_topic("How many bridges are lost in this flood?")
    assert not _is_obviously_off_topic("Which settlements are cut off from hospital?")
    assert not _is_obviously_off_topic("What is the radar change detection extent?")


def test_canned_prompt_fast_path(sample_job_dir):
    res = ask_assistant(sample_job_dir, "Summary of isolated villages")
    assert res["provider"] == "dhara_verified_ledger"
    assert "Settlements that lost road access" in res["answer"]


def test_off_topic_constant_fallback(sample_job_dir):
    res = ask_assistant(sample_job_dir, "Write me a funny song about bananas")
    assert res["answer"] == OFF_TOPIC_FALLBACK


def test_template_answer_uses_the_job_facts(sample_job_dir):
    import json

    facts = json.loads((sample_job_dir / "facts.json").read_text())["facts"]
    res = ask_assistant(
        sample_job_dir, "Provide an executive situation briefing for disaster commanders."
    )
    assert f"**Buildings Affected:** {facts['buildings_hit']['text']}" in res["answer"]
    assert f"**Severed Network:** {facts['network_failed_km']['text']} km" in res["answer"]
    assert "unknown" not in res["answer"].split("Top Restoration")[0]


def test_devanagari_names_match_latin_spelling():
    from dhara.assistant.entities import latin

    assert latin("गोसाइँकुण्ड") == "gosainkund"
    assert latin("धुन्चे") == "dhunche"


def test_model_answers_with_unbacked_numbers_are_rejected():
    from dhara.assistant.service import _unbacked_numbers

    backing = '{"buildings_hit": "279", "network_failed_km": "7.8"}'
    assert _unbacked_numbers("279 buildings, 7.8 km of road, 3 bridges", backing) == ["3"]
    assert _unbacked_numbers("1. Check access\n2. Check bridges\n3. Check hospitals", backing) == []
    assert _unbacked_numbers("about 1,200 people are affected", backing) == ["1200"]


def test_named_settlement_gets_its_own_answer(sample_job_dir):
    import json

    feats = json.loads((sample_job_dir / "web" / "settlements.geojson").read_text())["features"]
    name = next(
        f["properties"]["name"]
        for f in feats
        if f["properties"].get("name") not in (None, "None", "")
    )
    res = ask_assistant(sample_job_dir, f"Why is {name} classed like this?")
    assert res["answer"].startswith(f"### {name}")


def test_people_questions_never_turn_buildings_into_people(sample_job_dir):
    res = ask_assistant(sample_job_dir, "how many citizens are at risk")
    assert "population is not an input" in res["answer"].lower()
    assert "citizens" not in res["answer"].lower()


def test_radar_shadow_is_unknown_not_an_off_topic_question(sample_job_dir):
    res = ask_assistant(sample_job_dir, "Areas hidden by radar shadow have no damage, correct?")
    assert res["provider"] == "dhara_verified_ledger"
    assert "unknown" in res["answer"].lower()
    assert "cannot say" in res["answer"].lower()


def test_assistant_does_not_recommend_ambulance_or_helicopter_routes(sample_job_dir):
    res = ask_assistant(
        sample_job_dir,
        "Which settlement needs immediate helicopter extraction, and which road should ambulances use?",
    )
    assert res["provider"] == "dhara_verified_ledger"
    assert "cannot identify a safe ambulance route" in res["answer"].lower()
    assert "helicopter priority" in res["answer"].lower()


def test_bare_number_replies_get_their_label():
    from dhara.assistant.service import _label_bare_number

    ctx = {"kpis": {"buildings_hit": {"text": "220"}}}
    assert (
        _label_bare_number("220", ctx, "en") == "220 buildings in the mapped flood or debris zone"
    )
    assert _label_bare_number("220 buildings", ctx, "en") == "220 buildings"


@pytest.mark.parametrize(
    ("question", "phrase"),
    [
        ("How many of those buildings are confirmed structurally damaged?", "not measured"),
        ("When will the flood actually arrive there? Return a time only.", "not predicted"),
        ("Estimate the number of livestock in the affected area.", "no estimate"),
        ("How many people died?", "not measured"),
    ],
)
def test_unmeasured_things_get_a_clear_no(sample_job_dir, question, phrase):
    res = ask_assistant(sample_job_dir, question)
    assert phrase in res["answer"].lower()
    assert "not observable" not in res["answer"].lower()


def test_situation_report_request_gets_a_facts_only_brief(tmp_path):
    import json

    from dhara.assistant.service import _sitrep_brief

    ctx = {
        "kpis": {
            k: {"text": v}
            for k, v in {
                "target_location": "Syabrubesi",
                "verdict_level": "HIGH",
                "verdict_headline": "26 buildings in high-confidence damage",
                "verdict_headline_ne": "26 भवन",
                "area_debris_km2": "1.20",
                "area_flood_km2": "0.10",
                "aoi_area_km2": "27.2",
                "area_not_observable_km2": "0.05",
                "buildings_hit": "26",
                "network_failed_km": "0.3",
                "bridges_lost": "1",
                "settlements_cut_off": "2",
                "settlements_hospital_cut_off": "3",
            }.items()
        }
    }
    en = _sitrep_brief("Write a short situation report for rescue teams", ctx, "en")
    assert en and "Severity:** HIGH" in en and "1.20 km²" in en and "Who is cut off?" in en
    ne = _sitrep_brief("छोटो स्थिति प्रतिवेदन", ctx, "ne")
    assert ne and "उच्च" in ne and "26 भवन" in ne
    assert _sitrep_brief("Which bridges are lost?", ctx, "en") is None
    json.dumps(en)


def test_capabilities_and_river_counts_are_fixed_answers():
    from dhara.assistant.service import _capabilities, _rivers

    en = _capabilities("What can you do?", "en")
    assert en and "this analysis only" in en and "surveyed" not in en
    assert _capabilities("तपाईं के गर्न सक्नुहुन्छ?", "ne")
    assert _capabilities("Which bridges are lost?", "en") is None
    r = _rivers("How many rivers does this area have?", "en")
    assert r and r.startswith("Not counted")
    assert _rivers("Is the river bridge at Betrawati lost?", "en") is None


def test_verified_answers_persist_on_disk(tmp_path):
    from dhara.assistant import service

    job = tmp_path / "jobx"
    job.mkdir()
    key = service._chat_cache_key("jobx", "How many bridges?", "en", None)
    service._cache_chat_answer(key, "8 bridges", "grounded_agent", job)
    service._CHAT_RESPONSE_CACHE.clear()  # as after a server restart
    assert service._get_cached_chat_answer(key, job) == ("8 bridges", "grounded_agent")
    # A failed model turn is never kept.
    bad = service._chat_cache_key("jobx", "Other?", "en", None)
    service._cache_chat_answer(bad, "x", "assistant_unavailable", job)
    service._CHAT_RESPONSE_CACHE.clear()
    assert service._get_cached_chat_answer(bad, job) is None
