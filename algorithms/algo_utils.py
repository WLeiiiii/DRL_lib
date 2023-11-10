def soft_update(local_net, target_net, tau):
    """
    Softly updates the target network parameters based on the local network parameters.
    This method blends the parameters of the two networks according to a mixing factor tau.

    The formula for updating the target network parameters is:
    θ_target = tau*θ_local + (1 - tau)*θ_target

    Parameters:
        local_net (torch.nn.Module): The local network with parameters θ_local being used to compute the Q-values.
                                     It is typically the most recent version of the network that is being actively trained.
        target_net (torch.nn.Module): The target network with parameters θ_target that are being slowly updated.
                                      This network's action-value (Q) predictions are used to compute the loss for every action during training.
        tau (float): The interpolation parameter denoting the weight given to the local network's parameters in the update.
                     It controls the rate at which the target network is updated. A tau of 1.0 would mean copying the local network to the target network,
                     while a tau of 0 would mean not updating the target network at all.

    Returns:
        None: This function updates the target network in-place and does not return any value.
    """
    for target_param, local_param in zip(target_net.parameters(), local_net.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
