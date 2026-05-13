import os
import warnings

import torch
import torch.nn as nn
import __main__

class LSTMModel2D(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel2D, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.last = nn.Softmax(dim=1)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.last(out)
        return out
    

class prob_dist():
    
    def __init__(self, model_dir):
        requested = os.environ.get("FOLLOW_AHEAD_DEVICE", "").strip().lower()
        if requested == "cpu":
            self.device = torch.device("cpu")
        elif requested == "cuda" or torch.cuda.is_available():
            try:
                probe = torch.zeros(1, device="cuda")
                _ = probe.cpu()
                self.device = torch.device("cuda")
            except Exception as exc:
                warnings.warn(
                    f"CUDA is available but unusable for Follow-Ahead-Reaction; "
                    f"falling back to CPU. Details: {exc}"
                )
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        # Older checkpoints were serialized with __main__.LSTMModel2D.
        setattr(__main__, "LSTMModel2D", LSTMModel2D)
        # PyTorch 2.6 flipped the default `weights_only` to True, but the
        # packaged LSTM checkpoint pickles a full ``LSTMModel2D`` object
        # (not a state-dict), so the legacy full-pickle behaviour is needed.
        self.model = torch.load(model_dir, map_location=self.device, weights_only=False)
        self.model = self.model.to(self.device)
        self.model.eval()

    def forward(self, history):
        history = torch.tensor(history).float().to(self.device)
        history -= torch.clone(history[-1])
        history = history.unsqueeze(0)
        out = self.model(history).detach().squeeze().cpu().numpy()

        return {'left': out[0], 'straight': out[1], 'right': out[2]}
       
