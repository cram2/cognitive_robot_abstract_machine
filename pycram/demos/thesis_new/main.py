from demos.thesis_new import run_thesis_demo
from krrood.ormatic.utils import drop_database
from pycram.orm.ormatic_interface import Base
from pycram.orm.utils import pycram_sessionmaker

robots = (
    "stretch",
    "tiago",
    "g1",
    "justin",
    "armar7",
    "pr2",
    "hsrb",
)
actions = ("wipe", "cut", "mix")
# pr2,hsrb,stretch,tiago,g1,justin,armar7
# apartment,kitchen,isr?, suturo, robocup, isr-testbed
if __name__ == "__main__":
    session = pycram_sessionmaker()()
    # drop_database(session.bind)
    Base.metadata.create_all(session.bind)
    session.commit()
    # run_thesis_demo(
    #     "cut",
    #     robot_name="justin",
    #     environment_name="apartment",
    #     container_kind="cucumber",
    # )
    for robot in robots:
        run_thesis_demo(
            "wipe",
            robot_name=robot,
            environment_name="apartment",
        )
    #     )
    # for action in actions:
    #     for robot in robots:
    #         run_thesis_demo(
    #             action,
    #             robot_name=robot,
    #             environment_name="apartment",
    #         )
