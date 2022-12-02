from torch import nn


class CELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()

    def forward(self, predicts, targets):
        return self.xent_loss(predicts, targets)


class SELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()
        # Define the λ
        self.lamb = 1.0

    def forward(self, a_ij, predicts, targets):
        ce_loss = self.xent_loss(predicts, targets)
        reg_loss = self.lamb * a_ij.pow(2).sum(dim=1).mean()
        return ce_loss + reg_loss
