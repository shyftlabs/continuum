"""Unit tests for the structured-output helper (pure, no LLM calls)."""

from __future__ import annotations

from pydantic import BaseModel

from continuum.llm.structured_output import (
    coerce_and_validate,
    schema_prompt,
    to_openai_response_format,
)


class Review(BaseModel):
    sentiment: str
    score: float
    summary: str


class TestCoerceAndValidate:
    def test_clean_json(self):
        obj, err = coerce_and_validate(
            '{"sentiment": "mixed", "score": 0.75, "summary": "ok"}', Review
        )
        assert err is None
        assert isinstance(obj, Review)
        assert obj.score == 0.75

    def test_markdown_fenced_json(self):
        obj, err = coerce_and_validate(
            '```json\n{"sentiment": "pos", "score": 0.9, "summary": "great"}\n```', Review
        )
        assert err is None
        assert isinstance(obj, Review)

    def test_json_embedded_in_prose(self):
        obj, err = coerce_and_validate(
            'Here is the result: {"sentiment": "neg", "score": 0.1, "summary": "bad"} done.',
            Review,
        )
        assert err is None
        assert obj.sentiment == "neg"

    def test_unwraps_schema_name_wrapper(self):
        obj, err = coerce_and_validate(
            '{"Review": {"sentiment": "neg", "score": 0.1, "summary": "bad"}}', Review
        )
        assert err is None
        assert isinstance(obj, Review)

    def test_wrong_shape_returns_error_not_raise(self):
        obj, err = coerce_and_validate('{"review_summary": {"overall": "fantastic"}}', Review)
        assert obj is None
        assert err is not None
        assert "did not match schema" in err

    def test_not_json_returns_error(self):
        obj, err = coerce_and_validate("# Review\n**Sentiment:** mixed", Review)
        assert obj is None
        assert err is not None
        assert "not valid JSON" in err

    def test_empty_content(self):
        obj, err = coerce_and_validate("", Review)
        assert obj is None
        assert err == "empty content"

    def test_none_content(self):
        obj, err = coerce_and_validate(None, Review)
        assert obj is None
        assert err == "empty content"


class TestSchemaPrompt:
    def test_lists_all_field_names(self):
        prompt = schema_prompt(Review)
        for field in ("sentiment", "score", "summary"):
            assert field in prompt

    def test_instructs_json_only(self):
        prompt = schema_prompt(Review)
        assert "JSON" in prompt


class TestToOpenAIResponseFormat:
    def test_builds_json_schema_envelope(self):
        rf = to_openai_response_format(Review)
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "Review"
        assert "schema" in rf["json_schema"]
