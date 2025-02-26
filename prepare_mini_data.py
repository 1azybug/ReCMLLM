import torch
data = torch.load("../ReCMLLM_outputs/data/4080.pt", weights_only=False)
torch.save(data[:256], "mini_data.pt")