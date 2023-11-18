class DQNConfig:
    learning_rate = 1e-3
    gamma = 0.99
    tau = 1e-3
    buffer_size = int(1e5)
    batch_size = 256
    hidden_dim = 128
    update_frequency = 5

    double = False
    dueling = False
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


class C51Config:
    def __init__(self):
        self.learning_rate = 1e-3
        self.gamma = 0.99
        self.tau = 1e-3
        self.buffer_size = int(1e5)
        self.batch_size = 256
        self.hidden_dim = 128
        self.update_frequency = 5

        self.v_min = -10
        self.v_max = 10
        self.num_atoms = 51
        self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        self.z_values = [self.v_min + i * self.delta_z for i in range(self.num_atoms)]

    def __information__(self):
        print(
            "-------- C51 Config --------\n"
            f"learning_rate: {self.learning_rate}\n"
            f"gamma: {self.gamma}\n"
            f"tau: {self.tau}\n"
            f"buffer_size: {self.buffer_size}\n"
            f"batch_size: {self.batch_size}\n"
            f"hidden_dim: {self.hidden_dim}\n"
            f"update_frequency: {self.update_frequency}\n"
            f"v_min: {self.v_min}\n"
            f"v_max: {self.v_max}\n"
            f"num_atoms: {self.num_atoms}\n"
            f"delta_z: {self.delta_z}\n"
            "---------------------------\n"
        )
