import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.input_dim = configs.get('input_dim')
        self.hidden_dim1 = configs.get('hidden_dim1')
        self.hidden_dim2 = configs.get('hidden_dim2')
        self.output_dim = configs.get('output_dim')
        self.dropout_ratio = configs.get('dropout_ratio')
        self.use_batch_norm = configs.get('use_batch_norm')

        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim1)
        self.batch_normalization1 = nn.BatchNorm1d(self.hidden_dim1)
        self.leaky_relu1 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(p=self.dropout_ratio)
        self.linear2 = nn.Linear(self.hidden_dim1, self.hidden_dim2)
        self.batch_normalization2 = nn.BatchNorm1d(self.hidden_dim2)
        self.leaky_relu2 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(p=self.dropout_ratio)
        self.output = nn.Linear(self.hidden_dim2, self.output_dim)
    
    def forward(self, x):
        x = self.linear1(x)
        if self.use_batch_norm:
            x = self.batch_normalization1(x)
        x = self.leaky_relu1(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        if self.use_batch_norm:
            x = self.batch_normalization2(x)
        x = self.leaky_relu2(x)
        x = self.dropout2(x)
        x = self.output(x)

        return x
