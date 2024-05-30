import sys
from pathlib import Path
from typing import Dict
from stable_baselines3 import DQN

sys.path.append(str(Path(__file__).parent.parent))

from simulation.simulator import MPCSimulator
from simulation.gang_scheduling.mpc import ReinforcementMPController
from simulation.config.config import (
    GANG_SCHEDULING_SIMULATION_LENGTH,
    GANG_SCHEDULING_WINDOW_SIZE,
    ZOOKEEPER_BARRIER_PATH,
    ZOOKEEPER_CLIENT_ENDPOINT,
)
from simulation.gang_scheduling.reinforcement import SimulatorEnv, A2C_PATH, DQN_PATH
from simulation.gang_scheduling.resource_configurer import ResourceConfigurer
from simulation.shared.workloads import WORKLOADS
from simulation.forecaster.lstm_forecaster import get_actual_dict, get_predictions_dict


def main() -> None:
    predictions: Dict[str, float] = get_predictions_dict(WORKLOADS)
    actual: Dict[str, float] = get_actual_dict(WORKLOADS)
    resource_configurer: ResourceConfigurer = ResourceConfigurer(
        workloads=WORKLOADS, predictions=predictions
    )
    env = SimulatorEnv(
        resource_configurer=resource_configurer,
        window_size=6,
        default_reward=0,
        default_time_step=200,
        workloads=WORKLOADS,
        num_actions=4,
        duration_low=75,
        duration_high=150,
        actual_workload_sizes=actual,
    )
    env.reset()
    model = DQN.load(DQN_PATH)

    reinforcement_mpc: ReinforcementMPController = ReinforcementMPController(
        model=model,
        env=env,
        resource_configurer=resource_configurer,
        simulation_length=GANG_SCHEDULING_SIMULATION_LENGTH,
        window_size=GANG_SCHEDULING_WINDOW_SIZE,
    )

    simulator: MPCSimulator = MPCSimulator(
        mpc=reinforcement_mpc,
        resource_configurer=resource_configurer,
        workloads=WORKLOADS,
        actual=actual,
        zookeeper_client_endpoint=ZOOKEEPER_CLIENT_ENDPOINT,
        zookeeper_barrier_path=ZOOKEEPER_BARRIER_PATH,
        real_simulation=False,
    )
    simulator.simulate()


if __name__ == "__main__":
    main()
