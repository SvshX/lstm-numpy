import numpy as np
from scipy.special import expit as sigmoid
import torch
from torch import nn


def forget_gate(x, h, Weights_hf, Bias_hf, Weights_xf, Bias_xf, prev_cell_state):
    forget_hidden = np.dot(Weights_hf, h) + Bias_hf
    forget_eventx = np.dot(Weights_xf, x) + Bias_xf
    return np.multiply(sigmoid(forget_hidden + forget_eventx), prev_cell_state)


def input_gate(x, h, Weights_hi, Bias_hi, Weights_xi, Bias_xi, Weights_hl, Bias_hl, Weights_xl, Bias_xl):
    ignore_hidden = np.dot(Weights_hi, h) + Bias_hi
    ignore_eventx = np.dot(Weights_xi, x) + Bias_xi
    learn_hidden = np.dot(Weights_hl, h) + Bias_hl
    learn_eventx = np.dot(Weights_xl, x) + Bias_xl
    return np.multiply(sigmoid(ignore_eventx + ignore_hidden), np.tanh(learn_eventx + learn_hidden))


def cell_state(forget_gate_output, input_gate_output):
    return forget_gate_output + input_gate_output


def output_gate(x, h, Weights_ho, Bias_ho, Weights_xo, Bias_xo, cell_state):
    out_hidden = np.dot(Weights_ho, h) + Bias_ho
    out_eventx = np.dot(Weights_xo, x) + Bias_xo
    return np.multiply(sigmoid(out_eventx + out_hidden), np.tanh(cell_state))


# Set Parameters for a small LSTM network
input_size = 2  # size of one 'event', or sample, in our batch of data
hidden_dim = 3  # 3 cells in the LSTM layer
output_size = 1  # desired model output


def model_output(lstm_output, fc_Weight, fc_Bias):
    '''Takes the LSTM output and transforms it to our desired 
    output size using a final, fully connected layer'''
    return np.dot(fc_Weight, lstm_output) + fc_Bias


# Initialize an PyTorch LSTM for comparison to our Numpy LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers=1):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, batch_first=True)
        # Final, fully-connected layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x, hidden):
        batch_size = 1
        # get LSTM outputs
        lstm_output, (h, c) = self.lstm(x, hidden)
        # shape output to be (batch_size*seq_length, hidden_dim)
        lstm_output = lstm_output.view(-1, self.hidden_dim)

        # get final output
        model_output = self.fc(lstm_output)

        return model_output, (h, c)


torch.manual_seed(5)
torch_lstm = LSTM(input_size=input_size,
                  hidden_dim=hidden_dim,
                  output_size=output_size,
                  )

state = torch_lstm.state_dict()
# print(state)

# Event (x) Weights and Biases for all gates
Weights_xi = state['lstm.weight_ih_l0'][0:3].numpy()  # shape  [h, x]
Weights_xf = state['lstm.weight_ih_l0'][3:6].numpy()  # shape  [h, x]
Weights_xl = state['lstm.weight_ih_l0'][6:9].numpy()  # shape  [h, x]
Weights_xo = state['lstm.weight_ih_l0'][9:12].numpy()  # shape  [h, x]

Bias_xi = state['lstm.bias_ih_l0'][0:3].numpy()  # shape is [h, 1]
Bias_xf = state['lstm.bias_ih_l0'][3:6].numpy()  # shape is [h, 1]
Bias_xl = state['lstm.bias_ih_l0'][6:9].numpy()  # shape is [h, 1]
Bias_xo = state['lstm.bias_ih_l0'][9:12].numpy()  # shape is [h, 1]

# Hidden state (h) Weights and Biases for all gates
Weights_hi = state['lstm.weight_hh_l0'][0:3].numpy()  # shape is [h, h]
Weights_hf = state['lstm.weight_hh_l0'][3:6].numpy()  # shape is [h, h]
Weights_hl = state['lstm.weight_hh_l0'][6:9].numpy()  # shape is [h, h]
Weights_ho = state['lstm.weight_hh_l0'][9:12].numpy()  # shape is [h, h]

Bias_hi = state['lstm.bias_hh_l0'][0:3].numpy()  # shape is [h, 1]
Bias_hf = state['lstm.bias_hh_l0'][3:6].numpy()  # shape is [h, 1]
Bias_hl = state['lstm.bias_hh_l0'][6:9].numpy()  # shape is [h, 1]
Bias_ho = state['lstm.bias_hh_l0'][9:12].numpy()  # shape is [h, 1]

# Final, fully connected layer Weights and Bias
fc_Weight = state['fc.weight'][0].numpy()  # shape is [h, output_size]
fc_Bias = state['fc.bias'][0].numpy()  # shape is [,output_size]

# --------------------------------------------------------------------
# Simple Time Series Data
data = np.array(
    [[1, 1],
     [2, 2],
     [3, 3]])

# Initialize cell and hidden states with zeroes
h = np.zeros(hidden_dim)
c = np.zeros(hidden_dim)

# Loop through a batch of data, updating the hidden and cell states each time
print('NumPy LSTM Output:')
for eventx in data:
    f = forget_gate(eventx, h, Weights_hf, Bias_hf, Weights_xf, Bias_xf, c)
    i = input_gate(eventx, h, Weights_hi, Bias_hi, Weights_xi, Bias_xi,
                   Weights_hl, Bias_hl, Weights_xl, Bias_xl)
    c = cell_state(f, i)
    h = output_gate(eventx, h, Weights_ho, Bias_ho, Weights_xo, Bias_xo, c)
    print(model_output(h, fc_Weight, fc_Bias))

# PyTorch expects an extra dimension for batch size:
torch_batch = torch.Tensor(data).unsqueeze(0)
torch_output, (torch_hidden, torch_cell) = torch_lstm(torch_batch, None)

print('\nPyTorch LSTM Output:')
print(torch_output)
print('\n', '-'*40)
print(f'Torch Hidden State: {torch_hidden}')
print(f'Torch Cell State: {torch_cell}\n')
print(f'np Hidden State: {h}')
print(f'np Cell State: {c}')
