from dataclasses import dataclass

from experiments.control_loop_experiments.control_loop_profiler import (
    CONTROL_CYCLE_PHASES,
    ControlLoopProfiler,
    PhaseDefinition,
)

# %% owners whose profiled method is inherited


@dataclass
class ProfiledMethodOwner:
    """
    Defines the method a profiled phase is measured on.
    """

    def measured(self) -> str:
        return "measured"


@dataclass
class InheritingProfiledMethodOwner(ProfiledMethodOwner):
    """
    Takes the profiled method over from its base class instead of defining it.
    """


# %% profiler mechanics


class TestProfilerInstallation:
    """
    The profiler may only change the control loop while it is measuring it.
    """

    def test_profiled_methods_are_restored(self):
        profiler = ControlLoopProfiler(scenario_name="nothing", control_dt=0.05)
        before = {
            definition: definition.owner.__dict__[definition.method_name]
            for definition in CONTROL_CYCLE_PHASES
        }

        with profiler:
            for definition in CONTROL_CYCLE_PHASES:
                assert (
                    definition.owner.__dict__[definition.method_name]
                    is not before[definition]
                )

        for definition in CONTROL_CYCLE_PHASES:
            assert (
                definition.owner.__dict__[definition.method_name] is before[definition]
            )

    def test_inherited_method_is_measured_and_left_inherited(self):
        """
        A phase whose method the owner inherited must install, measure and disappear
        again, so that profiling never turns an inherited method into an own one.
        """
        definition = PhaseDefinition(
            InheritingProfiledMethodOwner, "measured", "control_cycle"
        )
        profiler = ControlLoopProfiler(
            scenario_name="inherited",
            control_dt=0.05,
            phase_definitions=(definition,),
        )

        with profiler:
            assert "measured" in InheritingProfiledMethodOwner.__dict__
            assert InheritingProfiledMethodOwner().measured() == "measured"

        assert "measured" not in InheritingProfiledMethodOwner.__dict__
        assert InheritingProfiledMethodOwner().measured() == "measured"
        assert profiler.profile.control_cycles == 1
