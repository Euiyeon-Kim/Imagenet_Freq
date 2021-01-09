class Config:
    # Directories
    exp_name = 'similar_G12_lfc'
    results_dir = f'exps/{exp_name}'
    load_classifier_weight_path = None

    # Fourier
    r = 12
    mode = 'lfc'
    mask_shape = 'gaussian'

    # Trainer
    num_epochs = 10000
    lr = 1e-3

    # GradCAM location
    cam_layer = 'conv5_block3_out'

    # Basics
    n_gpus = 1
    n_workers = 8
    batch_size = 32
    epochs_to_validate = 1
    epochs_to_save_gradCAM = 1
    epochs_to_save_weights = 10

    # Data
    data_mode = 'similar'
    num_classes = 20
    input_shape = (224, 224)
    root_dir = 'data/imagenet'
    train_txt_path = f'data/{data_mode}_train_infos.txt'
    val_txt_path = f'data/{data_mode}_val_infos.txt'
    test_txt_path = f'data/{data_mode}_test_infos.txt'
