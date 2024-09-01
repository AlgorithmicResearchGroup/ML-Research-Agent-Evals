class CountParametersBenchmark:
    """
    A class for counting the parameters of a PyTorch model.

    Args:
        model (torch.nn.Module): The model to count the parameters of.

    Methods:
        count_parameters(): Counts the total, trainable, and non-trainable parameters of the model.
    """

    def __init__(self, model):
        self.model = model

    def count_parameters(self):
        # Count total number of parameters
        total_params = sum(p.numel() for p in self.model.parameters())

        # Count trainable parameters
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        # Count non-trainable parameters
        non_trainable_params = total_params - trainable_params

        # Format parameter counts with comma separators
        total_params_str = f"{total_params:,}"
        trainable_params_str = f"{trainable_params:,}"
        non_trainable_params_str = f"{non_trainable_params:,}"

        print(f"Total parameters: {total_params_str}")
        print(f"Trainable parameters: {trainable_params_str}")
        print(f"Non-trainable parameters: {non_trainable_params_str}")

        return total_params, trainable_params, non_trainable_params
