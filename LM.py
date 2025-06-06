from transformers import AutoModelForMaskedLM
import torch.nn as nn
from torch_geometric.nn.models import MLP
import torch

class LM_Model(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.LM_model_name = model_config['lm_model']

        card = model_config['card']
        self.LM = AutoModelForMaskedLM.from_pretrained(card)

    def forward(self, tokenized_tensors):

        outputs = self.LM(**tokenized_tensors, output_attentions=True)
        logits = outputs.logits
        return logits, outputs.attentions