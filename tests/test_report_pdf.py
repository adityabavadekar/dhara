from dhara.sitrep.pdf import ascii_filename, report_filename

FACTS = {
    "event": "2026-08-26T02:55:00Z",
    "target_location": "पाल्ते (27.92°N, 85.25°E)",
    "target_lat": "27.92",
    "target_lon": "85.25",
}


def test_name_keeps_devanagari_vowel_signs():
    assert (
        report_filename(FACTS, "68d2b5869583", "en") == "Dhara_SitRep_2026-08-26_पाल्ते_68d2b5869583"
    )


def test_nepali_report_is_marked():
    assert report_filename(FACTS, "j1", "ne").endswith("_j1_NE")


def test_punctuation_becomes_single_dash():
    f = {**FACTS, "target_location": "Bisang  Bazaar / Ward 7 (x)"}
    assert report_filename(f, "j1", "en") == "Dhara_SitRep_2026-08-26_Bisang-Bazaar-Ward-7_j1"


def test_ascii_fallback_uses_coordinates():
    assert ascii_filename(FACTS, "j1", "en") == "Dhara_SitRep_2026-08-26_27.92N-85.25E_j1"
    assert ascii_filename(FACTS, "j1", "en").isascii()


def test_missing_place_and_coordinates():
    assert (
        report_filename({"event": "2026-08-26T00:00:00Z"}, "j1", "en")
        == "Dhara_SitRep_2026-08-26_j1"
    )
