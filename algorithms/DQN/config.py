class DQNConfig:
    learning_rate = 1e-3
    gamma = 0.99
    tau = 1e-3
    buffer_size = int(1e5)
    batch_size = 256
    hidden_dim = 128
    update_frequency = 5

    double = True
    dueling = True
    noisy = False
    prior = False

    def __information__(self):
        print(
            "-------- DQN Config --------\n"
            f"learning_rate: {self.learning_rate}\n"
            f"gamma: {self.gamma}\n"
            f"tau: {self.tau}\n"
            f"buffer_size: {self.buffer_size}\n"
            f"batch_size: {self.batch_size}\n"
            f"hidden_dim: {self.hidden_dim}\n"
            f"update_frequency: {self.update_frequency}\n"
            f"double: {self.double}\n"
            f"dueling: {self.dueling}\n"
            f"noisy: {self.noisy}\n"
            f"prior: {self.prior}\n"
            "---------------------------\n"
        )
