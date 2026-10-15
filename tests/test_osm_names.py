from dhara.osm_names import english_name, nepali_name


def test_english_views_prefer_english_and_latin_names():
    assert english_name({"name": "چک 13 گ ب", "name:en": "Chak 13 GB"}) == "Chak 13 GB"
    assert english_name({"name": "पाल्ते", "name:en": "Paatle"}) == "Paatle"
    assert english_name({"name": "Kartarpura"}) == "Kartarpura"
    assert english_name({"name": "पाल्ते"}) == "पाल्ते"  # Devanagari kept when nothing else exists


def test_other_scripts_are_romanised_and_nepali_views_prefer_name_ne():
    assert english_name({"name": "吉隆镇"}) == "Jilongzhen"
    assert english_name({"name": "بلوچاں"}).isascii()
    assert nepali_name({"name": "Paatle", "name:ne": "पाल्ते"}) == "पाल्ते"
    assert (
        english_name({"name": "x", "bridge:name": "Trishuli Bridge"}, "bridge:name")
        == "Trishuli Bridge"
    )
