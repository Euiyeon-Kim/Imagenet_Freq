class Config:
    # Directories
    exp_name = 'similar_G_12_lfc'
    results_dir = f'exps/{exp_name}'
    load_classifier_weight_path = None

    # Fourier
    r = 12
    mode = 'low'
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
    num_classes = 20
    input_shape = (224, 224)
    root_dir = 'data/imagenet_freq_similar/G_12_lfc'
    train_txt_path = 'data/imagenet_freq_similar/poc_similar_train_infos.txt'
    val_txt_path = 'data/imagenet_freq_similar/poc_similar_val_infos.txt'
    test_txt_path = 'data/imagenet_freq_similar/poc_similar_test_infos.txt'
