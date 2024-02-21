{"username":"mahaboobandbasha","key":"f9a1a6c863f923aab4559f695aabeafb"}
https://drive.google.com/file/d/1qsGJvEEpH0dYU6u1KkIADP5si16upneF/view?usp=sharing
import torch
import torch.nn.functional as F

def binary_cross_entropy_with_logits(input, target, weight=None, reduction='mean'):
    """
    Args:
        input (Tensor): The input tensor containing the logits.
        target (Tensor): The target tensor containing labels (0 or 1).
        weight (Tensor, optional): A manual rescaling weight given to the loss of each batch element.
        reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.

    Returns:
        Tensor: The calculated loss.
    """
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

    if weight is not None:
        loss = loss * weight

    if reduction == 'none':
        return loss
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'mean':
        return loss.mean()
    else:
        raise ValueError("Invalid value for reduction: {}".format(reduction))

# Example usage
logits = torch.randn(5, requires_grad=True)
targets = torch.empty(5).random_(2)
logits = torch.load('pred_scores5.pt')
targets = torch.load('target_scores5.pt')
loss = binary_cross_entropy_with_logits(logits, targets, reduction='sum') / max(targets.sum(), 1)

print(loss)

'''
7.1248
7.0986
6.9362
7.08
6.91
6.401
'''
