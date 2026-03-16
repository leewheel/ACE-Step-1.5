"""Unit tests for lyrics script requirements in ``LLMHandler`` prompts."""

import unittest
from unittest.mock import MagicMock

try:
    from acestep.llm_inference import LLMHandler

    _IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover – dependency guard
    LLMHandler = None
    _IMPORT_ERROR = exc


@unittest.skipIf(LLMHandler is None, f"llm_inference import unavailable: {_IMPORT_ERROR}")
class TestUnderstandingPromptLyricsScript(unittest.TestCase):
    """Understanding prompt should discourage romanization for Chinese lyrics."""

    def test_instruction_requests_hanzi_not_romanization(self):
        """System instruction should ask for Chinese characters, not pinyin."""
        handler = LLMHandler()
        handler.llm_tokenizer = MagicMock()
        handler.llm_tokenizer.apply_chat_template = MagicMock(return_value="PROMPT")

        rendered = handler.build_formatted_prompt_for_understanding("<|audio_code_1|>")
        self.assertEqual(rendered, "PROMPT")

        (messages,), kwargs = handler.llm_tokenizer.apply_chat_template.call_args
        self.assertFalse(kwargs.get("tokenize", True))
        self.assertTrue(kwargs.get("add_generation_prompt", False))

        system_content = messages[0]["content"]
        self.assertIn("Chinese characters", system_content)
        self.assertIn("not pinyin/romanization", system_content)
        self.assertIn("[zh]", system_content)

