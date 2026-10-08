"""Tests for the SectionResult failure-isolation pattern."""

from brief.models import SectionResult, safe_section


def test_success_is_ok() -> None:
    result = SectionResult.success({"price": 1.0}, source="test")
    assert result.ok
    assert result.as_of.tzinfo is not None


def test_failure_is_not_ok() -> None:
    result: SectionResult[dict] = SectionResult.failure("boom", source="test")
    assert not result.ok
    assert result.error == "boom"


def test_safe_section_converts_exceptions() -> None:
    @safe_section(source="test")
    def broken() -> SectionResult[int]:
        raise ValueError("bad data")

    result = broken()
    assert not result.ok
    assert "ValueError" in result.error
