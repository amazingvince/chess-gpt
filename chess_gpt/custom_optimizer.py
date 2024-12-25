# custom optimizer fun:
from muon import Muon

# pip install git+https://github.com/KellerJordan/Muon
from torch.optim.lr_scheduler import LambdaLR


def warmup_constant_schedule(warmup_steps):
    """
    Creates a schedule with linear warmup and then constant learning rate.

    Args:
        warmup_steps (int): Number of warmup steps

    Returns:
        function: Schedule function to be passed to LambdaLR
    """

    def lr_lambda(current_step):
        # Linear warmup phase
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Constant phase
        return 1.0

    return lr_lambda


# Example usage
def create_scheduler(optimizer, warmup_steps=1000):
    """
    Creates a LambdaLR scheduler with warmup then constant schedule.

    Args:
        optimizer: PyTorch optimizer
        warmup_steps (int): Number of warmup steps

    Returns:
        LambdaLR: PyTorch learning rate scheduler
    """
    return optimizer, LambdaLR(
        optimizer, lr_lambda=warmup_constant_schedule(warmup_steps)
    )


def create_optimizer_with_muon(model, args):
    """
    Creates Muon optimizer for use with HuggingFace Trainer
    """
    # Parameters for Muon (≥2D parameters in the main model body and FEN encoder)
    muon_params = []
    muon_params.extend([p for p in model.model.parameters() if p.ndim >= 2])
    muon_params.extend([p for p in model.fen_encoder.parameters() if p.ndim >= 2])

    # Parameters for AdamW
    adamw_params = []

    # Add lower dimensional parameters from main model
    adamw_params.extend([p for p in model.model.parameters() if p.ndim < 2])

    # Add LM head parameters
    adamw_params.extend(model.lm_head.parameters())

    # Add lower dimensional FEN encoder parameters
    adamw_params.extend([p for p in model.fen_encoder.parameters() if p.ndim < 2])

    # Create optimizer using learning rates from training arguments
    optimizer = Muon(
        muon_params,
        lr=args.learning_rate * 6.67,  # 0.02 if args.learning_rate is 3e-4
        momentum=0.95,
        adamw_params=adamw_params,
        adamw_lr=args.learning_rate,
        adamw_betas=(0.90, 0.95),
        adamw_wd=args.weight_decay,
    )

    return optimizer
