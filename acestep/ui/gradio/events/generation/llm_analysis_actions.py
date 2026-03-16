"""Audio-code analysis and transcription actions for generation handlers.

This module contains source-audio analysis and audio-code transcription
entry points used by the Gradio generation UI.
"""

import re
import os

import gradio as gr

from acestep.inference import understand_music
from acestep.ui.gradio.i18n import t

from .validation import _contains_audio_code_tokens, clamp_duration_to_gpu_limit


def _is_unreliable_zh_lyrics(language: str, lyrics: str) -> bool:
    """Return True when analyzed Chinese lyrics look like synthetic numbered pinyin."""
    if not lyrics or not language:
        return False
    if language.strip().lower() not in {"zh", "yue"}:
        return False

    lowered = lyrics.lower()
    numbered_tokens = re.findall(r"\b[a-züv]+[1-5]\b", lowered)
    if len(numbered_tokens) < 6:
        return False
    return "[zh]" in lowered or len(numbered_tokens) >= 10


def _sanitize_analysis_lyrics(language: str, lyrics: str, status_message: str) -> tuple[str, str]:
    """Clear unreliable analyzed lyrics and append a warning message to status."""
    if not _is_unreliable_zh_lyrics(language=language, lyrics=lyrics):
        return lyrics, status_message

    warning = "⚠ 检测到疑似拼音占位歌词，已清空歌词，请手动填写或改用转写流程。"
    if status_message:
        return "", f"{status_message}\n{warning}"
    return "", warning


def _transcribe_lyrics_with_whisper(src_audio: str, language: str) -> tuple[str, str]:
    """Transcribe lyrics from source audio via Whisper API when fallback is available."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return "", "ℹ 未启用Whisper歌词回退（缺少 OPENAI_API_KEY）。"

    try:
        from scripts.lora_data_prepare.whisper_transcription import (
            transcribe_whisper,
            words_to_lyrics,
        )
    except Exception:
        return "", "ℹ Whisper歌词回退不可用（缺少转写依赖）。"

    target_language = (language or "").strip().lower()
    if target_language == "yue":
        target_language = "zh"
    if not target_language:
        target_language = None

    api_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1").strip()
    model = os.getenv("OPENAI_WHISPER_MODEL", "whisper-1").strip()
    try:
        words = transcribe_whisper(
            audio_path=src_audio,
            api_key=api_key,
            api_url=api_url,
            model=model,
            language=target_language,
        )
        lyrics = words_to_lyrics(words).strip()
    except Exception as exc:
        return "", f"ℹ Whisper歌词回退失败：{exc}"

    if not lyrics:
        return "", "ℹ Whisper歌词回退未返回有效文本。"
    return lyrics, "✅ 已使用Whisper回退转写歌词。"


def analyze_src_audio(
    dit_handler,
    llm_handler,
    src_audio,
    constrained_decoding_debug: bool = False,
):
    """Analyze source audio and optionally transcribe generated audio codes.

    Args:
        dit_handler: DiT handler instance.
        llm_handler: LLM handler instance.
        src_audio: Path to source audio file.
        constrained_decoding_debug: Whether constrained-decoding debug logs are enabled.

    Returns:
        Tuple of ``(audio_codes, status, caption, lyrics, bpm, duration,
        keyscale, language, timesignature, is_format_caption)``.
    """
    error_tuple = ("", "", "", "", None, None, "", "", "", False)

    if not src_audio:
        gr.Warning(t("messages.no_source_audio"))
        return error_tuple

    if dit_handler.model is None:
        gr.Warning(t("messages.model_not_initialized"))
        return error_tuple

    try:
        codes_string = dit_handler.convert_src_audio_to_codes(src_audio)
    except Exception as exc:
        gr.Warning(t("messages.audio_conversion_failed", error=str(exc)))
        return error_tuple

    if not codes_string or not _contains_audio_code_tokens(codes_string):
        gr.Warning(t("messages.no_audio_codes_generated"))
        return (
            codes_string or "",
            t("messages.no_audio_codes_generated"),
            "",
            "",
            None,
            None,
            "",
            "",
            "",
            False,
        )

    if not llm_handler.llm_initialized:
        return (
            codes_string,
            t("messages.codes_ready_no_lm"),
            "",
            "",
            None,
            None,
            "",
            "",
            "",
            False,
        )

    result = understand_music(
        llm_handler=llm_handler,
        audio_codes=codes_string,
        use_constrained_decoding=True,
        constrained_decoding_debug=constrained_decoding_debug,
    )

    if not result.success:
        return (
            codes_string,
            result.status_message,
            "",
            "",
            None,
            None,
            "",
            "",
            "",
            False,
        )

    sanitized_lyrics, sanitized_status = _sanitize_analysis_lyrics(
        language=result.language,
        lyrics=result.lyrics,
        status_message=result.status_message,
    )
    if _is_unreliable_zh_lyrics(language=result.language, lyrics=result.lyrics):
        fallback_lyrics, fallback_status = _transcribe_lyrics_with_whisper(
            src_audio=src_audio,
            language=result.language,
        )
        if fallback_lyrics:
            sanitized_lyrics = fallback_lyrics
        if fallback_status:
            sanitized_status = (
                f"{sanitized_status}\n{fallback_status}" if sanitized_status else fallback_status
            )
    clamped_duration = clamp_duration_to_gpu_limit(result.duration, llm_handler)
    return (
        codes_string,
        sanitized_status,
        result.caption,
        sanitized_lyrics,
        result.bpm,
        clamped_duration,
        result.keyscale,
        result.language,
        result.timesignature,
        True,
    )


def transcribe_audio_codes(llm_handler, audio_code_string, constrained_decoding_debug: bool):
    """Transcribe serialized audio codes into metadata fields via the LLM.

    Args:
        llm_handler: LLM handler instance.
        audio_code_string: Serialized audio-code tokens.
        constrained_decoding_debug: Whether constrained-decoding debug logs are enabled.

    Returns:
        Tuple of ``(status, caption, lyrics, bpm, duration, keyscale,
        language, timesignature, is_format_caption)``.
    """
    result = understand_music(
        llm_handler=llm_handler,
        audio_codes=audio_code_string,
        use_constrained_decoding=True,
        constrained_decoding_debug=constrained_decoding_debug,
    )

    if not result.success:
        if result.error == "LLM not initialized":
            return t("messages.lm_not_initialized"), "", "", None, None, "", "", "", False
        return result.status_message, "", "", None, None, "", "", "", False

    sanitized_lyrics, sanitized_status = _sanitize_analysis_lyrics(
        language=result.language,
        lyrics=result.lyrics,
        status_message=result.status_message,
    )
    clamped_duration = clamp_duration_to_gpu_limit(result.duration, llm_handler)
    return (
        sanitized_status,
        result.caption,
        sanitized_lyrics,
        result.bpm,
        clamped_duration,
        result.keyscale,
        result.language,
        result.timesignature,
        True,
    )
