class Config:
    debug = True
    print_freq = 100
    num_workers = 4
    model_name = ''
    epochs = 3
    lr = 1e-4
    batch_size = 64
    seed = 2021
    target_size = 1
    target_col = 'target'
    n_fold = 5
    trn_fold = [0]  # [0, 1, 2, 3, 4]
    train = True

    # scheduler = 'CosineAnnealingLR'
    # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']

    # factor=0.2 # ReduceLROnPlateau
    # patience=4 # ReduceLROnPlateau
    # eps=1e-6 # ReduceLROnPlateau
    # T_max = 3  # CosineAnnealingLR
    # T_0=3 # CosineAnnealingWarmRestarts

    # min_lr = 1e-6

    # weight_decay = 1e-6
    # gradient_accumulation_steps = 1
    # max_grad_norm = 1000
