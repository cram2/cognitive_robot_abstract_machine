from datetime import timedelta

import pytest
from std_srvs.srv import Trigger

from giskardpy.middleware.ros2.exceptions import ServiceUnavailableError
from giskardpy.middleware.ros2.ros2_interface import call_service

# %% calling services


def test_calling_a_service_that_nobody_offers_raises(init_rospy):
    with pytest.raises(ServiceUnavailableError):
        call_service(
            Trigger,
            "service_that_nobody_offers",
            Trigger.Request(),
            wait_timeout=timedelta(seconds=0.1),
        )
