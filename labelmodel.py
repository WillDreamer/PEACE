import torch
from torch import nn
from torch.nn import functional as F
def weights_init(m):
    nn.init.xavier_uniform(m.weight.data)
    nn.init.constant(m.bias.data, 0.00)

class centroid_gen(nn.Module):
    def __init__(self, label_dim, bit):
        """
        :param y_dim: dimension of tags
        :param bit: bit number of the final binary code
        """
        super(centroid_gen, self).__init__()
        self.module_name = "text_model"
        cl1 = nn.Linear(label_dim, 512)
        cl2 = nn.Linear(512, bit)
        self.cl_text = nn.Sequential(
            cl1,
            nn.ReLU(inplace=True),
            # cl3,
            # nn.ReLU(inplace=True),
            cl2,
            nn.Tanh()
        )
    def forward(self, x):
        y = self.cl_text(x)
        return y