import torch
import torch.nn as nn


class MooreMachineLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(MooreMachineLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # embedded layer, lstm layer with fully connected layer
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x, hidden, cell):
        # embed the input
        out = self.embedding(x)
        # Forward propagate LSTM
        out, (hidden, cell) = self.lstm(out.unsqueeze(1), (hidden, cell))
        out = out.reshape(out.shape[0], -1)

        return out, (hidden, cell)

    def init_hidden(self, batch_size):
        # initialize the hidden state with zeros
        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        return hidden, cell

    def loss_optimizer(self, lr=0.001, momentum=0.9) -> tuple:
        """
        Create a loss function and an optimizer for the network.
        :param lr:
        :param momentum:
        :return: loss function and optimizer
        """
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return loss_fn, optimizer
