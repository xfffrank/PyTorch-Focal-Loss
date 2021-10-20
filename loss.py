import torch
from torch import nn
import torch.nn.functional as F


class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        self.alpha = torch.tensor([1-alpha, alpha])
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        '''
        
        inputs: logits
        '''
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
        pt = torch.exp(-bce_loss)
        alpha = self.alpha.gather(0, targets.data.view(-1))
        focal_loss = alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
    
    
class SoftmaxFocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2):
        '''
        alpha: list
        '''
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        '''
        
        inputs: N x D logits
        '''
        ce_loss = F.cross_entropy(inputs, targets.long(), reduction='none')
        pt = torch.exp(-ce_loss)
        alpha = self.alpha.gather(0, targets.data.view(-1))
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


'''tests

inputs = torch.tensor([-2., 1., -3.])
targets = torch.tensor([1, 1, 0])

criterion = SigmoidFocalLoss()
criterion.forward(inputs, targets)


inputs = torch.tensor([[-2., 1., -3.], [-2., 1., -3.], [-2., 1., -3.]])
targets = torch.tensor([1, 2, 0])
alpha = [0.1, 0.4, 0.5]

criterion = SoftmaxFocalLoss(alpha=alpha)
criterion.forward(inputs, targets)
'''