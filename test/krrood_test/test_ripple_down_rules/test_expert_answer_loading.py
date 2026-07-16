from krrood.ripple_down_rules.experts import Human


def make_answer_source(return_expression: str) -> str:
    """
    :return: The source of a minimal recorded-answer function returning
        ``return_expression``, in the shape :meth:`Human.save_answers` writes.
    """
    return f"def _get_value(case):\n    return {return_expression}\n"


def test_load_answers_recovers_every_saved_answer(tmp_path):
    path = str(tmp_path / "answers")
    expert = Human()
    expert.all_expert_answers = [
        ({}, make_answer_source("case.legs == 4")),
        ({}, make_answer_source("case.legs == 2")),
        ({}, make_answer_source("case.legs == 8")),
    ]
    expert.save_answers(path)

    loaded = Human(use_loaded_answers=True)
    loaded.load_answers(path)

    assert len(loaded.all_expert_answers) == 3


def test_load_answers_recovers_the_last_answer_even_without_trailing_blank_lines(
    tmp_path,
):
    """
    Regression test: a formatter (e.g. black/docformatter) trimming the trailing blank
    lines a saved answers file ends with must not silently drop the last recorded answer
    on load.
    """
    path = str(tmp_path / "answers")
    expert = Human()
    expert.all_expert_answers = [
        ({}, make_answer_source("case.legs == 4")),
        ({}, make_answer_source("case.legs == 2")),
    ]
    expert.save_answers(path)

    with open(path + ".py", "r") as f:
        content = f.read()
    with open(path + ".py", "w") as f:
        f.write(content.rstrip("\n") + "\n")

    loaded = Human(use_loaded_answers=True)
    loaded.load_answers(path)

    assert len(loaded.all_expert_answers) == 2
