"""
Starting a part finding server from the command line, as ``semantic-digital-twin-part-
finding-server``. Needs the ``vision`` extra.

Without options it serves the locator that found the most handles in the handle finder
benchmark: Qwen3-VL-8B boxes masked by SAM 2.1, on the GPU.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from semantic_digital_twin.adapters.part_finding.server import (
    PartFindingCache,
    PartFindingServer,
)
from semantic_digital_twin.adapters.part_finding.vision_language_models import (
    Device,
    QwenBoxDetector,
    QwenModel,
    Quantization,
    Sam2BoxSegmenter,
)
from semantic_digital_twin.part_finding.vision_language import (
    VisionLanguagePartLocator,
)
from semantic_digital_twin.utils import create_cache_dir


class CommandLineOption(StrEnum):
    """
    The options of ``semantic-digital-twin-part-finding-server``.
    """

    MODEL = "--model"
    """
    Sets :attr:`PartFindingCommandLine.model`.
    """

    QUANTIZATION = "--quantization"
    """
    Sets :attr:`PartFindingCommandLine.quantization`.
    """

    DEVICE = "--device"
    """
    Sets :attr:`PartFindingCommandLine.device`.
    """

    PORT = "--port"
    """
    Sets :attr:`PartFindingCommandLine.port`.
    """

    CACHE = "--cache"
    """
    Sets :attr:`PartFindingCommandLine.cache`.
    """


@dataclass
class PartFindingCommandLine:
    """
    The settings of a part finding server started from the command line.
    """

    model: QwenModel = QwenModel.QWEN3_VL_8B
    """
    The size of Qwen3-VL that puts boxes around the parts.
    """

    quantization: Quantization = Quantization.NONE
    """
    How the weights of Qwen3-VL are stored.
    """

    device: Device = Device.CUDA
    """
    Where the models run.
    """

    port: int = 8790
    """
    The port the server listens on.
    """

    cache: Path = field(default_factory=lambda: create_cache_dir("part_finding"))
    """
    Where the server keeps its answers.
    """

    @classmethod
    def run(cls) -> None:
        """
        Read the settings from the command line and answer requests until the process is
        stopped.
        """
        defaults = cls()
        parser = argparse.ArgumentParser(description=PartFindingServer.__doc__)
        parser.add_argument(
            CommandLineOption.MODEL,
            choices=[model.name for model in QwenModel],
            default=defaults.model.name,
        )
        parser.add_argument(
            CommandLineOption.QUANTIZATION,
            choices=list(Quantization),
            default=defaults.quantization,
        )
        parser.add_argument(
            CommandLineOption.DEVICE, choices=list(Device), default=defaults.device
        )
        parser.add_argument(CommandLineOption.PORT, type=int, default=defaults.port)
        parser.add_argument(CommandLineOption.CACHE, type=Path, default=defaults.cache)
        arguments = parser.parse_args()
        settings = cls(
            model=QwenModel[arguments.model],
            quantization=Quantization(arguments.quantization),
            device=Device(arguments.device),
            port=arguments.port,
            cache=arguments.cache,
        )
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
        settings.server().serve_forever()

    def server(self) -> PartFindingServer:
        """
        :return: A server with these settings, listening but not answering yet.
        """
        return PartFindingServer(
            locator=VisionLanguagePartLocator(
                detector=QwenBoxDetector(
                    model=self.model,
                    quantization=self.quantization,
                    device=self.device,
                ),
                segmenter=Sam2BoxSegmenter(device=self.device),
            ),
            port=self.port,
            cache=PartFindingCache(directory=self.cache),
        )
