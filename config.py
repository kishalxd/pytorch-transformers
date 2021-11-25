class Config:
    num_classes=1
    epochs=10
    margin=0.5
    model_name = 'vinai/bertweet-base'
    batch_size = 8
    lr = 1e-5
    weight_decay=0.01
    scheduler = 'CosineAnnealingLR'
    #     scheduler = 'LinearWarmup'
    max_length = 128
    accumulation_step = 1
    patience = 1
    seed = 11
    warmup_steps = 10