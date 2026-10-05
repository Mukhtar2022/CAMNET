import os
import torch

class AverageMeter:
    """Computes and stores the average and current value of a metric.
    Typical usage:
        meter = AverageMeter()
        for batch in loader:
            loss = compute_loss(...)
            meter.update(loss.item(), n=batch_size)
        print(meter.avg)
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0  # most recent value
        self.avg = 0.0  # running average
        self.sum = 0.0  # sum of all values
        self.count = 0   # number of samples seen

    def update(self, val, n=1):
        """Update the meter with a new value.
        Args:
            val (float): New metric value.
            n (int): Number of samples the value corresponds to (default 1).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0.0

def save_checkpoint(state: dict, filepath: str = None):
    """Save training checkpoint to disk.
    Args:
        state (dict): Dictionary containing model state, optimizer state, epoch, etc.
        filepath (str, optional): Destination path. If omitted, uses
            state.get('ckpt_path') or defaults to 'checkpoint.pth'.
    """
    if filepath is None:
        filepath = state.get('ckpt_path', 'checkpoint.pth')
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    # Optionally also save a copy of the best model if indicated
    if state.get('is_best'):
        best_path = os.path.join(os.path.dirname(filepath), 'best_' + os.path.basename(filepath))
        torch.save(state, best_path)
