from dhara.sitrep.facts import fmt_num


def test_small_positive_values_do_not_read_as_zero():
    assert fmt_num(0.003, 2) == "<0.01"
    assert fmt_num(0.0, 2) == "0.00"
    assert fmt_num(0.005, 2) == "0.01"
    assert fmt_num(1234.5, 2) == "1,234.50"
    assert fmt_num(3, 0) == "3"
