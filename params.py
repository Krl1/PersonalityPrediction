RANDOM_SEED = 42
DATA_NAME = 'BFD'

class WandbConfig:
    project_name = 'BFD_crop'
    run_name = 'final_run'
    entity = 'krl1'
    save_dir = '.'


class CreateDataConfig:
    classification = True
    test_size_ratio = 0.2
    Y_threshold = 0.5
    connect = True


class LocationConfig:
    checkpoints_dir = f'model/{DATA_NAME}/checkpoints'
    crop_images = f'data/{DATA_NAME}/crop_images/'
    best_model = f'model/{DATA_NAME}/best.pt'
    images = f'data/{DATA_NAME}/images/'
    labels = f'data/{DATA_NAME}/'
    data = f'data/{DATA_NAME}/'
    
    
class TrainingConfig:
    accumulate_grad_batches = 1
    deterministic = True
    batch_size = 32
    patience = batch_size
    epochs = batch_size*10
    gpus = 1
    
class NetworkConfig:
    negative_slope = 0.0
    batch_norm = False
    dropout = 0.3
    lr = 1e-4
