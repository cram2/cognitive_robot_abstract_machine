import datetime

from neem_pycram_interface import PyCRAMNEEMInterface
from typing_extensions import List

from ..episode_player import EpisodePlayer


class NEEMPlayer(EpisodePlayer):

    def __init__(self, pycram_neem_interface: PyCRAMNEEMInterface):
        super().__init__()
        self.pni = pycram_neem_interface

    def query_neems_motion_replay_data(self, sql_neem_ids: List[int]):
        self.pni.query_neems_motion_replay_data(sql_neem_ids=sql_neem_ids)

    @property
    def ready(self):
        return self.pni.replay_environment_initialized

    def get_frame_data_generator(self):
        return self.pni.replay_motions_in_query(real_time=self.use_realtime,
                                                step_time=self.time_between_frames,
                                                continuous_mode=False)

    def _run(self):
        motion_gen = self.pni.replay_motions_in_query(real_time=True,
                                                      step_time=datetime.timedelta(milliseconds=10),
                                                      continuous_mode=False)
        for _ in motion_gen:
            self._wait_if_paused()

    def _pause(self):
        ...

    def _resume(self):
        ...
