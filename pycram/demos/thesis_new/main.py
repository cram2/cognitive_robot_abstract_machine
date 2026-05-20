from demos.thesis_new import run_thesis_demo_isolated
from krrood.ormatic.utils import drop_database
from pycram.orm.ormatic_interface import Base
from pycram.orm.utils import pycram_sessionmaker
import matplotlib

# use this in terminal: MPLBACKEND=Agg python headmaps.py
matplotlib.use("Agg")
robots = (
    "tiago",
    "justin",
    "stretch",
    "hsrb",
    "pr2",
    "armar7",
    "g1",
)

actions = ("mix", "wipe", "cut")
# pr2,hsrb,stretch,tiago,g1,justin,armar7
# apartment,kitchen,isr?, suturo, robocup, isr-testbed
if __name__ == "__main__":
    session = pycram_sessionmaker()()
    drop_database(session.bind)
    Base.metadata.create_all(session.bind)
    session.commit()

    for action in ["cut", "mix", "wipe"]:
        for env in ["kitchen", "apartment", "isr"]:
            for robot in robots:
                run_thesis_demo_isolated(
                    task_name=action,
                    robot_name=robot,
                    environment_name=env,
                )

    # run_thesis_demo(
    #     "cut",
    #     robot_name="justin",
    #     environment_name="apartment",
    #     container_kind="cucumber",z`
    # )
    # for env in ("isr", "kitchen", "apartment"):
    #     for action in actions:
    #         for robot in robots:
    #             run_thesis_demo(
    #                 task_name=action,
    #                 robot_name=robot,
    #                 environment_name=env,
    #             )

    #     )
    # for action in actions:
    #     for robot in robots:
    #         run_thesis_demo(
    #             action,
    #             robot_name=robot,
    #             environment_name="apartment",
    #         )
