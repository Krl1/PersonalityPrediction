RANDOM_SEED = 42
DATA_NAME = 'DENTA'
DATA_TYPE = 'rgb'

class WandbConfig:
    project_name = f'{DATA_NAME}_{DATA_TYPE}'
    run_name = 'final_run'
    entity = 'krl1'
    save_dir = '.'


class CreateDataConfig:
    classification = True
    test_size_ratio = 0.1
    Y_threshold = 0.5
    connect = True


class LocationConfig:
    checkpoints_dir = f'model/{DATA_NAME}/checkpoints'
    crop_images = f'data/{DATA_NAME}/crop_images/'
    best_model = f'model/{DATA_NAME}/best.pt'
    images = f'data/{DATA_NAME}/images/'
    labels = f'data/{DATA_NAME}/'
    enc = f'data/{DATA_NAME}/enc/'
    data = f'data/{DATA_NAME}/{DATA_TYPE}/'
    rgb = f'data/{DATA_NAME}/rgb/'
    gray = f'data/{DATA_NAME}/gray/'
    
    
class TrainingConfig:
    accumulate_grad_batches = 1
    deterministic = True
    batch_size = 32
    patience = batch_size
    epochs = batch_size*10
    gpus = 1
    
class NetworkConfig:
    negative_slope = 0.0
    batch_norm = True
    dropout = 0.3
    lr = 1e-4
