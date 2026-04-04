import threading
import time

import rclpy
from rclpy.executors import SingleThreadedExecutor

from krrood.ormatic.dao import to_dao
from krrood.ormatic.utils import drop_database
from semantic_digital_twin.adapters.ros.world_fetcher import fetch_world_from_service
from semantic_digital_twin.orm.ormatic_interface import Base
from semantic_digital_twin.orm.utils import semantic_digital_twin_sessionmaker

session = semantic_digital_twin_sessionmaker()()
drop_database(session.bind)
Base.metadata.create_all(session.bind)

rclpy.init()
node = rclpy.create_node("nodey")
executor = SingleThreadedExecutor()
executor.add_node(node)

# Start executor in a separate thread
thread = threading.Thread(target=executor.spin, daemon=True, name="rclpy-executor")
thread.start()
time.sleep(0.1)

hsrb_world = fetch_world_from_service(node)

dao = to_dao(hsrb_world)
session.add(dao)
session.commit()
