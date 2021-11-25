import torch

class ToxicDataset:
    def __init__(self, df, tokenizer, max_length):
        self.less_toxic = df['less_toxic'].values
        self.more_toxic = df['more_toxic'].values
        self.tokenizer = tokenizer
        self.max_len = max_length
        
    def __len__(self):
        return len(self.less_toxic)
    
    def __getitem__(self, idx):

        inputs_more_toxic = self.tokenizer.encode_plus(
                                self.more_toxic[idx].lower(),
                                truncation=True,
                                add_special_tokens=True,
                                max_length=self.max_len,
                                padding='max_length'
                            )
        inputs_less_toxic = self.tokenizer.encode_plus(
                                self.less_toxic[idx].lower(),
                                truncation=True,
                                add_special_tokens=True,
                                max_length=self.max_len,
                                padding='max_length'
                            )
        
        more_toxic_ids = inputs_more_toxic['input_ids']
        more_toxic_mask = inputs_more_toxic['attention_mask']
        
        less_toxic_ids = inputs_less_toxic['input_ids']
        less_toxic_mask = inputs_less_toxic['attention_mask']
        
        target = 1
       
        return {
            'less_toxic_ids': torch.tensor(less_toxic_ids, dtype=torch.long),
            'less_toxic_mask': torch.tensor(less_toxic_mask, dtype=torch.long),
            'more_toxic_ids': torch.tensor(more_toxic_ids, dtype=torch.long),
            'more_toxic_mask': torch.tensor(more_toxic_mask, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long)
        }
       