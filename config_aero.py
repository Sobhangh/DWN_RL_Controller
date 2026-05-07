# -----------------------------------------------------------
#  Imports so we can reference the actual class objects.
#  If DockingEnv expects strings instead, just delete the
#  `import … as …` lines and keep the dotted‑string names.
# -----------------------------------------------------------
import math
from SafeRL.saferl.environment.tasks.processor.reward import DistanceExponentialChangeRewardProcessorNew
from saferl.aerospace.models.cwhspacecraft.platforms.cwh import CWHSpacecraft2d
from saferl.environment.tasks.initializers import RandBoundsInitializer
from saferl.aerospace.tasks.docking.initializers import ConstrainedDeputyPolarInitializer, SimpleDockingInitializer
from saferl.environment.models.platforms import AgentController
from saferl.environment.models.geometry import RelativeCircle

from saferl.aerospace.tasks.docking.processors import (
    InDockingStatusProcessor,
    DockingDistanceStatusProcessor,
    DockingThrustDeltaVStatusProcessor,
    DockingVelocityLimit,
    DockingVelocityLimitViolation,
    RelativeVelocityConstraint,
    FailureStatusProcessor,
    SuccessStatusProcessor,
)
from saferl.aerospace.tasks.docking.processors import (
    AccumulatorStatusProcessor,
)

from saferl.environment.tasks.processor.observation import AttributeObservationProcessor
from saferl.environment.tasks.processor.reward import (
    DistanceExponentialChangeRewardProcessor,
    ProportionalRewardProcessor,
    DistanceExponentialChangeRewardProcessorNew,
)
from saferl.aerospace.tasks.docking.processors import (
    FailureRewardProcessor,
    SuccessRewardProcessor,
)

"""
TO CHANGE:
    Reward processor
    Initial distance
    Velocity limits? 0.5
    Docking region radius from 0.5m to 0.25m or 0.35?
"""

# -----------------------------------------------------------
#  Full environment‑configuration dictionary
# -----------------------------------------------------------
env_cfg = {
    "agent": "deputy",
    "step_size": 1,
    "env_objs": [
        {
            "name": "chief",
            "class": CWHSpacecraft2d,
            "config": {
                "init": {
                    "initializer": RandBoundsInitializer,
                    "x": 0,
                    "x_dot": 0,
                    "y": 0,
                    "y_dot": 0,
                }
            },
        },
        {
            "name": "deputy",
            "class": CWHSpacecraft2d,
            "config": {
                "integration_method": "RK45",
                "controller": {
                    "class": AgentController,
                    "actuators": [
                        {
                            "name": "thrust_x",
                            "space": "continuous",
                            "bounds": [-1, 1],
                        },
                        {
                            "name": "thrust_y",
                            "space": "continuous",
                            "bounds": [-1, 1],
                        },
                    ],
                },
                "init": {
                    "initializer": SimpleDockingInitializer,
                    "ref": "chief",
                    "radius": [4.0, 5.0], # changed from og 100,150 for testing
                    "angle": [0, 2 * 3.141592653589793],
                },
            },
        },
        {
            "name": "docking_region",
            "class": RelativeCircle,
            "config": {
                "ref": "chief",
                "x_offset": 0,
                "y_offset": 0,
                "radius": 0.35,
                "init": {"initializer": RandBoundsInitializer},
            },
        },
    ],
    "status": [
        {
            "name": "in_docking",
            "class": InDockingStatusProcessor,
            "config": {"deputy": "deputy", "docking_region": "docking_region"},
        },
        {
            "name": "docking_distance",
            "class": DockingDistanceStatusProcessor,
            "config": {"deputy": "deputy", "docking_region": "docking_region"},
        },
        {
            "name": "delta_v",
            "class": DockingThrustDeltaVStatusProcessor,
            "config": {"target": "deputy"},
        },
        {
            "name": "custom_metrics.delta_v_total",
            "class": AccumulatorStatusProcessor,
            "config": {"status": "delta_v"},
        },
        {
            "name": "max_vel_limit",
            "class": DockingVelocityLimit,
            "config": {
                "target": "deputy",
                "dist_status": "docking_distance",
                "vel_threshold": 0.2,
                "threshold_dist": 0.5,
                "slope": 2,
            },
        },
        {
            "name": "max_vel_violation",
            "class": DockingVelocityLimitViolation,
            "config": {
                "target": "deputy",
                "ref": "chief",
                "vel_limit_status": "max_vel_limit",
            },
        },
        {
            "name": "max_vel_constraint",
            "class": RelativeVelocityConstraint,
            "config": {
                "target": "deputy",
                "ref": "chief",
                "vel_limit_status": "max_vel_limit",
            },
        },
        {
            "name": "failure",
            "class": FailureStatusProcessor,
            "config": {
                "docking_distance": "docking_distance",
                "max_goal_distance": 40000,
                "timeout": 2000,
                "in_docking_status": "in_docking",
                "max_vel_constraint_status": "max_vel_constraint",
            },
        },
        {
            "name": "success",
            "class": SuccessStatusProcessor,
            "config": {
                "in_docking_status": "in_docking",
                "max_vel_constraint_status": "max_vel_constraint",
            },
        },
    ],
    "observation": [
        {
            "name": "obs_x",
            "class": AttributeObservationProcessor,
            "config": {
                "target": "deputy",
                "attr": "x",
                "observation_space_shape": 1,
                "normalization": 100,
            },
        },
        {
            "name": "obs_y",
            "class": AttributeObservationProcessor,
            "config": {
                "target": "deputy",
                "attr": "y",
                "observation_space_shape": 1,
                "normalization": 100,
            },
        },
        {
            "name": "obs_x_dot",
            "class": AttributeObservationProcessor,
            "config": {
                "target": "deputy",
                "attr": "x_dot",
                "observation_space_shape": 1,
                "normalization": 0.5,
            },
        },
        {
            "name": "obs_y_dot",
            "class": AttributeObservationProcessor,
            "config": {
                "target": "deputy",
                "attr": "y_dot",
                "observation_space_shape": 1,
                "normalization": 0.5,
            },
        },
    ],
    "reward": [
        {
            "name": "dist_change_reward",
            "class": DistanceExponentialChangeRewardProcessorNew,
            "config": {"agent": "deputy", "target": "docking_region", "pivot": 100},
        },
        {
            "name": "delta_v",
            "class": ProportionalRewardProcessor,
            "config": {
                "scale": -0.01,
                "bias": 0,
                "proportion_status": "delta_v",
            },
        },
        {
            "name": "max_vel_constraint",
            "class": ProportionalRewardProcessor,
            "config": {
                "scale": -0.01, #-0.01 og increased for testing
                "bias": -0.01,
                "proportion_status": "max_vel_violation",
                "cond_status": "max_vel_constraint",
                "cond_status_invert": True,
                "lower_bound": -5,
                "lower_bound_terminal": "failure",
            },
        },
        {
            "name": "failure_reward",
            "class": FailureRewardProcessor,
            "config": {
                "failure_status": "failure",
                "reward": {
                    "crash": -1,
                    "distance": -1,
                    "timeout": -1,
                    "reward_lower_bound_max_vel_constraint": 0,
                },
            },
        },
        {
            "name": "success_reward",
            "class": SuccessRewardProcessor,
            "config": {
                "reward": 1,
                "success_status": "success",
                "timeout": 2000,
            },
        },
    ],
    "verbose": False,
}
