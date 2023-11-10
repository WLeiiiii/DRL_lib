from algorithms.DQN.dqn import DQNAgent


class Agents:
    agents = {
        'q_learning': 'agents/q_learning.py',
        'sarsa': 'agents/sarsa.py',
        'dqn': DQNAgent,
        'ddpg': 'agents/ddpg_agent_simple_env.py',
    }

    @staticmethod
    def get_agent(algorithm_name, **kwargs):
        algorithm_name_lower = algorithm_name.lower()
        if algorithm_name_lower in Agents.agents:
            return Agents.agents[algorithm_name_lower](**kwargs)
        else:
            raise ValueError(f"No agent found for algorithm: {algorithm_name}")
