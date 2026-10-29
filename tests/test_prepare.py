from prepare import clean, lower_tr, normalize, section, without

PAGE = (
    '<noinclude><pagequality level="3" user="x" /></noinclude>'
    '<section begin="A"/>end of one story<section end="A"/>'
    "<section begin=\"B\"/><font color=\"#aa634a\">'''hânım hey'''</font>\nstart of the next<section end=\"B\"/>"
    "<noinclude></noinclude>"
)


def test_clean_strips_markup_and_keeps_rubrics():
    text = clean(section(PAGE, "B"))
    assert text == "hânım hey\nstart of the next"


def test_section_missing_name_gives_whole_page():
    assert section(PAGE, None) == PAGE
    assert section(PAGE, "nope") == PAGE


def test_without_drops_the_neighbours_section():
    text = clean(without(PAGE, "B"))
    assert text == "end of one story"


def test_clean_templates_links_soft_hyphens():
    assert clean("{{ortalı ve hizalı}}be­yân [[a|bağ]] [[düz]]") == "beyân bağ düz"


def test_turkish_lower_case():
    assert lower_tr("Iğdır İstanbul") == "ığdır istanbul"


def test_normalize_drops_stray_digits():
    assert normalize("Vez3iri") == "veziri"
