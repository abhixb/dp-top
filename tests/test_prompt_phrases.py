"""Tests for PromptPhraseKey enum and prompt-phrase validation in BaseModelClient."""

import pytest

from topreward.utils.constants import PromptPhraseKey


def _make_valid_phrases() -> dict:
    """Build a dict that satisfies every required PromptPhraseKey."""
    return {
        PromptPhraseKey.INITIAL_SCENE_LABEL.value: "Initial Scene:",
        PromptPhraseKey.INITIAL_SCENE_COMPLETION.value: "Completion: 0%",
        PromptPhraseKey.CONTEXT_FRAME_LABEL_TEMPLATE.value: "Frame {i}:",
        PromptPhraseKey.CONTEXT_FRAME_COMPLETION_TEMPLATE.value: "Completion: {p}%",
        PromptPhraseKey.EVAL_FRAME_LABEL_TEMPLATE.value: "Eval Frame {i}:",
        PromptPhraseKey.EVAL_TASK_COMPLETION_INSTRUCTION.value: ["Predict completion for {instruction}."],
    }


def _get_validator():
    """Return the _validate_and_normalize_prompt_phrases method as a standalone callable."""
    from topreward.clients.base import BaseModelClient

    class _StubClient(BaseModelClient):
        def _generate_from_events(self, events, temperature=0.0):
            return ""

    instance = _StubClient.__new__(_StubClient)
    return instance._validate_and_normalize_prompt_phrases


def test_all_required_keys_present():
    """A complete dict with all required keys passes validation without raising."""
    validate = _get_validator()
    phrases = _make_valid_phrases()
    result = validate(phrases)
    # Should return a normalized dict with all keys present
    for key in PromptPhraseKey:
        assert key.value in result


def test_missing_required_key_raises():
    """An empty dict raises ValueError because all required keys are missing."""
    validate = _get_validator()
    with pytest.raises(ValueError):
        validate({})


def test_single_missing_key_raises():
    """Omitting one required key raises ValueError mentioning that key."""
    validate = _get_validator()
    phrases = _make_valid_phrases()
    del phrases[PromptPhraseKey.EVAL_TASK_COMPLETION_INSTRUCTION.value]
    with pytest.raises(ValueError):
        validate(phrases)


def test_all_enum_values_are_strings():
    """Every PromptPhraseKey enum value is a non-empty string."""
    for key in PromptPhraseKey:
        assert isinstance(key.value, str)
        assert len(key.value) > 0


def test_enum_has_all_required_keys():
    """PromptPhraseKey exposes the six expected keys."""
    expected = {
        "initial_scene_label",
        "initial_scene_completion",
        "context_frame_label_template",
        "context_frame_completion_template",
        "eval_frame_label_template",
        "eval_task_completion_instruction",
    }
    actual = {k.value for k in PromptPhraseKey}
    assert actual == expected


def test_extra_keys_are_ignored():
    """Extra keys in the input dict are accepted (not raised as errors)."""
    validate = _get_validator()
    phrases = _make_valid_phrases()
    phrases["unknown_extra_key"] = "some value"
    result = validate(phrases)
    # Extra key should not appear in the normalized output
    assert "unknown_extra_key" not in result
