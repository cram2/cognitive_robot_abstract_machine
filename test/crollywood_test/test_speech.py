"""
Tests for :mod:`crollywood.speech`: sound written and read as a wave file, speech
trimmed and levelled, and the Kokoro speech model's lines kept on disk.
"""

from __future__ import annotations

import sys
import types
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from crollywood.cache import SceneCache
from crollywood.narration import NarratedLine
from crollywood.speech import (
    KokoroLanguage,
    KokoroTextToSpeech,
    KokoroVoice,
    SoundChannel,
    SoundLevel,
)

from .voices import EvenlyPacedTextToSpeech, ToneSpeechModel

RATE = 24_000
SILENCE_LEVEL = SoundLevel(decibels=-40.0)
BREATH_BEFORE = timedelta(seconds=0.03)
BREATH_AFTER = timedelta(seconds=0.12)
PEAK = SoundLevel(decibels=-1.0)


def test_speech_lasts_its_samples_over_its_rate() -> None:
    samples = 2400
    assert SoundChannel(
        np.zeros(samples, dtype=np.float32), RATE
    ).duration == timedelta(seconds=samples / RATE)


def test_sound_written_to_a_wave_file_reads_back_the_same(tmp_path: Path) -> None:
    samples = np.linspace(-1.0, 1.0, 480, dtype=np.float32)
    path = SoundChannel(samples, RATE).written_to(tmp_path / "line.wav")
    read = SoundChannel.read_from(path)
    assert isinstance(read, SoundChannel)
    assert read.sample_rate == RATE
    assert read.samples == pytest.approx(samples, abs=1e-4)


def test_speech_is_trimmed_to_its_sound_keeping_a_breath_before_and_after() -> None:
    rate = 1000
    sound_length = 1
    samples = np.zeros(3 * rate, dtype=np.float32)
    samples[rate : (1 + sound_length) * rate] = 0.5
    trimmed = SoundChannel(samples, rate).trimmed(
        SILENCE_LEVEL, BREATH_BEFORE, BREATH_AFTER
    )
    assert isinstance(trimmed, SoundChannel)
    assert (
        trimmed.duration
        == timedelta(seconds=sound_length) + BREATH_BEFORE + BREATH_AFTER
    )


def test_speech_with_no_sound_is_kept_whole_when_trimmed() -> None:
    silence = SoundChannel(np.zeros(100, dtype=np.float32), 100)
    assert silence.trimmed(SILENCE_LEVEL, BREATH_BEFORE, BREATH_AFTER) is silence


def test_speech_is_levelled_so_its_loudest_sample_reaches_the_peak() -> None:
    samples = np.array([0.1, -0.25, 0.2], dtype=np.float32)
    levelled = SoundChannel(samples, 100).levelled(PEAK)
    loudest = 0.25
    assert levelled.samples == pytest.approx(samples * PEAK.amplitude / loudest)


def test_silence_is_left_as_it_is_when_levelled() -> None:
    silence = SoundChannel(np.zeros(10, dtype=np.float32), 100)
    assert silence.levelled(PEAK) is silence


def test_a_level_of_minus_twenty_decibels_is_a_tenth_of_full_scale() -> None:
    assert SoundLevel(decibels=-20.0).amplitude == pytest.approx(0.1)


def test_a_level_of_zero_decibels_is_full_scale() -> None:
    assert SoundLevel(decibels=0.0).amplitude == 1.0


def test_a_line_at_half_pace_takes_twice_as_long_to_say() -> None:
    voice = EvenlyPacedTextToSpeech()
    half_pace = 0.5
    at_its_pace = NarratedLine("one two three").spoken_by(voice)
    slowed = NarratedLine("one two three", pace=half_pace).spoken_by(voice)
    assert slowed.duration == at_its_pace.duration / half_pace


def test_the_kokoro_voice_says_a_line_at_its_pace_trimmed_and_levelled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    speech_model = types.ModuleType("kokoro_onnx")
    speech_model.Kokoro = ToneSpeechModel
    monkeypatch.setitem(sys.modules, "kokoro_onnx", speech_model)
    voice = KokoroTextToSpeech(speed=1.0, cache=SceneCache("narration", root=tmp_path))
    half_pace = 0.5
    slowed = NarratedLine("one two", pace=half_pace).spoken_by(voice)
    at_its_pace = NarratedLine("one two").spoken_by(voice)
    breath = voice.breath_before + voice.breath_after
    tone = ToneSpeechModel.time_per_word * 2
    tolerance = timedelta(seconds=1 / ToneSpeechModel.sample_rate)
    assert abs(slowed.duration - (tone / half_pace + breath)) <= tolerance
    assert abs(at_its_pace.duration - (tone + breath)) <= tolerance
    assert np.abs(slowed.samples).max() == pytest.approx(voice.peak.amplitude, abs=1e-4)


def test_a_line_said_twice_is_synthesized_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    created = []

    class CountingModel(ToneSpeechModel):
        def create(self, text, voice, speed, lang):
            created.append(text)
            return super().create(text, voice, speed, lang)

    speech_model = types.ModuleType("kokoro_onnx")
    speech_model.Kokoro = CountingModel
    monkeypatch.setitem(sys.modules, "kokoro_onnx", speech_model)
    voice = KokoroTextToSpeech(cache=SceneCache("narration", root=tmp_path))
    line = NarratedLine("one two")
    line.spoken_by(voice)
    line.spoken_by(voice)
    assert created == [line.said]


def test_the_kokoro_model_is_asked_for_the_chosen_voice_and_language(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    asked = []

    class AskedModel(ToneSpeechModel):
        def create(self, text, voice, speed, lang):
            asked.append((voice, lang))
            return super().create(text, voice, speed, lang)

    speech_model = types.ModuleType("kokoro_onnx")
    speech_model.Kokoro = AskedModel
    monkeypatch.setitem(sys.modules, "kokoro_onnx", speech_model)
    voice = KokoroTextToSpeech(cache=SceneCache("narration", root=tmp_path))
    NarratedLine("one two").spoken_by(voice)
    assert asked == [(KokoroVoice.HEART, KokoroLanguage.AMERICAN_ENGLISH)]
