"""
Speech as sound, and the voices that make it.

The lines are said once each and kept, so the soundtrack and the subtitles come from the
same speech and always agree with each other.
"""

from __future__ import annotations

import hashlib
import wave
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from datetime import timedelta
from enum import StrEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from krrood.ormatic.utils import classproperty

from crollywood.cache import SceneCache

# %% how loud


@dataclass(frozen=True)
class SoundLevel:
    """
    How loud a sample is, in decibels relative to full scale (dBFS): zero is the loudest
    a sample holds, and every quieter level is negative.
    """

    decibels: float
    """
    Decibels relative to full scale.
    """

    @property
    def amplitude(self) -> float:
        """
        The sample value of this level, from zero to one.
        """
        return 10 ** (self.decibels / 20)


# %% sound


@dataclass(frozen=True)
class SoundChannel:
    """
    One channel of sound, kept as samples.
    """

    samples: NDArray[np.float32]
    """
    The sound, one channel, from minus one to one.
    """

    sample_rate: int
    """
    Samples per second, in hertz.
    """

    @classproperty
    def written_sample(cls) -> np.dtype:
        """
        How a sample is written to a wave file: a little-endian sixteen-bit integer,
        which every player reads.
        """
        return np.dtype("<i2")

    @classmethod
    def loudest_sample(cls) -> int:
        """
        :return: The largest value a written sample holds, which a sound at full scale is
            written as.
        """
        return int(np.iinfo(cls.written_sample).max)

    @property
    def duration(self) -> timedelta:
        """
        How long it lasts.
        """
        return timedelta(seconds=len(self.samples) / self.sample_rate)

    def written_to(self, path: Path) -> Path:
        """
        :param path: Where the wave file goes.
        :return: The path.
        """
        scaled = np.clip(self.samples, -1.0, 1.0) * self.loudest_sample()
        with wave.open(str(path), "wb") as written:
            written.setnchannels(1)
            written.setsampwidth(self.written_sample.itemsize)
            written.setframerate(self.sample_rate)
            written.writeframes(np.round(scaled).astype(self.written_sample).tobytes())
        return path

    @classmethod
    def read_from(cls, path: Path) -> SoundChannel:
        """
        :param path: A wave file of one channel, its samples as :meth:`written_to`
            writes them.
        :return: The sound it holds.
        """
        with wave.open(str(path), "rb") as read:
            sample_rate = read.getframerate()
            frames = read.readframes(read.getnframes())
        samples = (
            np.frombuffer(frames, dtype=cls.written_sample).astype(np.float32)
            / cls.loudest_sample()
        )
        return cls(samples, sample_rate)

    def trimmed(
        self,
        silence_level: SoundLevel,
        breath_before: timedelta,
        breath_after: timedelta,
    ) -> SoundChannel:
        """
        :param silence_level: The loudest a sample may be and still count as silence.
        :param breath_before: The quiet kept before the first sound, so its first word
            is not cut.
        :param breath_after: The quiet kept after the last sound: the natural tail of
            its last word.
        :return: The sound without the silence around it; all of it where it is all
            silence.
        """
        loud = np.flatnonzero(np.abs(self.samples) > silence_level.amplitude)
        if loud.size == 0:
            return self
        first = max(0, loud[0] - int(breath_before.total_seconds() * self.sample_rate))
        after_last = min(
            len(self.samples),
            loud[-1] + 1 + int(breath_after.total_seconds() * self.sample_rate),
        )
        return replace(self, samples=self.samples[first:after_last])

    def levelled(self, peak: SoundLevel) -> SoundChannel:
        """
        :param peak: The level the loudest sample is scaled to.
        :return: The sound scaled so its loudest sample reaches the peak; silence as it
            is.
        """
        loudest = float(np.abs(self.samples).max(initial=0.0))
        if loudest == 0.0:
            return self
        return replace(self, samples=self.samples * (peak.amplitude / loudest))


# %% voices


@dataclass
class TextToSpeech(ABC):
    """
    Whatever turns a line into speech.
    """

    @abstractmethod
    def speaks(self, text: str, pace: float = 1.0) -> SoundChannel:
        """
        :param text: What to say.
        :param pace: How fast, as a share of the voice's own speed.
        :return: It, said.
        """


class KokoroVoice(StrEnum):
    """
    The voices of the Kokoro speech model, as the model names them.
    """

    HEART = "af_heart"
    """
    Heart, a woman speaking American English.
    """


class KokoroLanguage(StrEnum):
    """
    The languages the Kokoro speech model reads text in, as the model names them.
    """

    AMERICAN_ENGLISH = "en-us"
    """
    English as spoken in the United States.
    """


@dataclass
class KokoroTextToSpeech(TextToSpeech):
    """
    The Kokoro speech model, run on this machine, one of its voices chosen.

    Every line is kept on disk once said, so a video is rebuilt with the very same
    speech.
    """

    voice: KokoroVoice = KokoroVoice.HEART
    """
    Which of the model's voices speaks.
    """

    speed: float = 1.15
    """
    How fast it speaks, one being the model's own pace.
    """

    language: KokoroLanguage = KokoroLanguage.AMERICAN_ENGLISH
    """
    The language the text is read in.
    """

    model: Path = field(
        default_factory=lambda: SceneCache.configured_root()
        / "voice"
        / "kokoro-v1.0.onnx"
    )
    """
    The model's weights.
    """

    voices: Path = field(
        default_factory=lambda: SceneCache.configured_root()
        / "voice"
        / "voices-v1.0.bin"
    )
    """
    The model's voices.
    """

    cache: SceneCache = field(default_factory=lambda: SceneCache("narration"))
    """
    Where the lines said are kept.
    """

    silence_level: SoundLevel = SoundLevel(decibels=-40.0)
    """
    The loudest a sample may be and still count as silence around a spoken line.
    """

    breath_before: timedelta = timedelta(seconds=0.03)
    """
    The quiet kept before the first sound of a line, so its first word is not cut.
    """

    breath_after: timedelta = timedelta(seconds=0.12)
    """
    The quiet kept after the last sound of a line: the natural tail of its last word.
    """

    peak: SoundLevel = SoundLevel(decibels=-1.0)
    """
    The level of the loudest sample of every spoken line, so every line is said at one
    loudness and none clips.
    """

    def speaks(self, text: str, pace: float = 1.0) -> SoundChannel:
        """
        :param text: What to say.
        :param pace: How fast, as a share of the voice's own speed.
        :return: It, said by the model the first time and read back from the cache
            after.
        """
        speed = self.speed * pace
        kept = self.cache.directory / f"{self._key(text, speed)}.wav"
        if not kept.exists():
            self._synthesize(text, speed).written_to(kept)
        return SoundChannel.read_from(kept)

    def _key(self, text: str, speed: float) -> str:
        """
        :param text: What is said.
        :param speed: The model's speed it is said at.
        :return: What a line said at a speed is kept as; how the speech is trimmed and
            levelled is part of it, so changing that says every line again rather than
            reusing speech cut the old way.
        """
        said = (
            f"{self.voice}|{speed}|{self.language}|{self.silence_level.decibels}|"
            f"{self.breath_before.total_seconds()}|"
            f"{self.breath_after.total_seconds()}|{self.peak.decibels}|{text}"
        )
        return hashlib.sha1(said.encode()).hexdigest()

    def _synthesize(self, text: str, speed: float) -> SoundChannel:
        """
        :param text: What to say.
        :param speed: The model's speed to say it at.
        :return: The line said by the model, trimmed and levelled.
        """
        # the speech model is an optional dependency, the ``narrator`` extra
        from kokoro_onnx import Kokoro

        model = Kokoro(str(self.model), str(self.voices))
        samples, sample_rate = model.create(
            text, voice=self.voice, speed=speed, lang=self.language
        )
        return (
            SoundChannel(samples.astype(np.float32), sample_rate)
            .trimmed(self.silence_level, self.breath_before, self.breath_after)
            .levelled(self.peak)
        )
