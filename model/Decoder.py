import torch
import torch.nn as nn
import torch.optim as optim

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)  # TO DO: Añadir dropout 
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden, cell):
        output, (hidden, cell) = self.rnn(x, (hidden, cell))
        output = self.fc_out(output)
        return output, (hidden, cell)
