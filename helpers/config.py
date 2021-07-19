class Config:
    apex = False
    debug = False
    print_freq = 100
    num_workers = 4
    model_name = 'tf_efficientnet_b7_ns'
    scheduler = 'CosineAnnealingLR'
    # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    epochs = 3
    # factor=0.2 # ReduceLROnPlateau
    # patience=4 # ReduceLROnPlateau
    # eps=1e-6 # ReduceLROnPlateau
    T_max = 3  # CosineAnnealingLR
    # T_0=3 # CosineAnnealingWarmRestarts
    lr = 1e-4
    min_lr = 1e-6
    batch_size = 64
    weight_decay = 1e-6
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 42
    target_size = 1
    target_col = 'target'
    n_fold = 5
    trn_fold = [0]  # [0, 1, 2, 3, 4]
    train = True
