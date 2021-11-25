import torch.nn as nn
from transformers import AutoModel

class ToxicModel(nn.Module):
    def __init__(self, model_name, args):
        super(ToxicModel, self).__init__()
        self.args = args
        self.model = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(p=0.2)
        self.output = nn.LazyLinear(self.args.num_classes)
    
        
    def forward(self, toxic_ids, toxic_mask):
        
        out = self.model(
            input_ids=toxic_ids,
            attention_mask=toxic_mask,
            output_hidden_states=False
        )
        
        out = self.dropout(out[1])
        outputs = self.output(out)

        return outputs
        