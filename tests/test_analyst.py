from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.agents.analyst import AnalystAssessment


def test_analyst_assessment_valid():
    a = AnalystAssessment(
        needs_more_research=True,
        gaps=["missing transport details"],
        re_research_instructions="search for transport protocol",
        synthesis="",
    )
    assert a.needs_more_research is True
    assert len(a.gaps) == 1


def test_analyst_assessment_defaults():
    a = AnalystAssessment(needs_more_research=False)
    assert a.gaps == []
    assert a.re_research_instructions == ""
    assert a.synthesis == ""


def test_analyst_assessment_requires_needs_more_research():
    with pytest.raises(ValidationError):
        AnalystAssessment()


def test_analyst_assessment_model_dump():
    a = AnalystAssessment(
        needs_more_research=False,
        gaps=[],
        synthesis="All research complete.",
    )
    d = a.model_dump()
    assert d["needs_more_research"] is False
    assert d["synthesis"] == "All research complete."
    assert isinstance(d["gaps"], list)
