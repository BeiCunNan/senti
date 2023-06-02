from torch import nn

# 若在model中最终没有加softmax,则交叉熵会自动帮我们加上

class CELoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()

    def forward(self, predicts, targets):
        return self.xent_loss(predicts, targets)

