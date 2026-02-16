import os

import pytest
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher

try:
    # from neem_pycram_interface import PyCRAMNEEMInterface
    # from segmind.segmenters.neem_segmenter import NEEMSegmenter
    raise ImportError
except ImportError:
    PyCRAMNEEMInterface = None
    NEEMSegmenter = None

from segmind.detectors.coarse_event_detectors import PlacingDetector, GeneralPickUpDetector
from unittest import TestCase


# @pytest.mark.skipif(PyCRAMNEEMInterface is None, reason="PyCRAMNEEMInterface not available")
@pytest.mark.skip(reason="PyCRAMNEEMInterface needs to be updated")
class TestNEEMSegmentor(TestCase):
    pni: PyCRAMNEEMInterface
    viz_mark_publisher: VizMarkerPublisher

    @classmethod
    def setUpClass(cls):
        BulletWorld(WorldMode.GUI)
        set_logger_level(LoggerLevel.DEBUG)
        cls.pni = PyCRAMNEEMInterface(f'mysql+pymysql://{os.environ["my_maria_uri"]}')
        cls.viz_mark_publisher = VizMarkerPublisher()

    @classmethod
    def tearDownClass(cls):
        cls.viz_mark_publisher._stop_publishing()
        if World.current_world is not None:
            World.current_world.exit()

    def test_event_detector(self):
        ns = NEEMSegmenter(self.pni, detectors_to_start=[GeneralPickUpDetector, PlacingDetector], annotate_events=True)
        ns.start([17])

    def test_general_pick_up_detector(self):
        ns = NEEMSegmenter(self.pni, detectors_to_start=[GeneralPickUpDetector], annotate_events=True)
        ns.start([17])
