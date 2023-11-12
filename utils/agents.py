from algorithms.DDPG.ddpg import DDPGAgent
from algorithms.DQN.dqn import DQNAgent
from algorithms.DQN.dqn_cnn import DQNAgentCNN
from algorithms.DQN.dqn_noisy import NoisyDQNAgent
from algorithms.DQN.dqn_per import PerDQNAgent
from algorithms.QLearning.q_learning import QLearningAgent
from algorithms.SARSA.sarsa import SarsaAgent


class Agents:
    agents = {
        'qlearning': QLearningAgent,
        'sarsa': SarsaAgent,
        'dqn': DQNAgent,
        'dqncnn': DQNAgentCNN,
        'noisydqn': NoisyDQNAgent,
        'perdqn': PerDQNAgent,
        'ddpg': DDPGAgent,
    }

    @staticmethod
    def get_agent(algorithm_name, **kwargs):
        algorithm_name_lower = algorithm_name.lower()
        if algorithm_name_lower in Agents.agents:
            return Agents.agents[algorithm_name_lower](**kwargs)
        else:
            raise ValueError(f"No agent found for algorithm: {algorithm_name}")
