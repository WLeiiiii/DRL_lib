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
