import os
from vissl.models import build_model
from classy_vision.generic.util import load_checkpoint
from vissl.utils.checkpoint import init_model_from_consolidated_weights
from PIL import Image
import torchvision.transforms as transforms
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict


 python3 tools/run_distributed_engines.py \
>     hydra.verbose=true \

>     config.DATA.TRAIN.DATASET_NAMES=[dummy_data_folder] \
>     config.DATA.TRAIN.DATA_SOURCES=[disk_folder] \
>     config.DATA.TRAIN.LABEL_SOURCES=[disk_folder] \
>     config.DATA.TRAIN.BATCHSIZE_PER_REPLICA=16 \
>     config.DATA.TRAIN.DATA_PATHS=["/home/siddhaganju/recursion-ssl/rxrx1/images"] \
>     config.DATA.TEST.DATA_SOURCES=[disk_folder] \
>     config.DATA.TEST.LABEL_SOURCES=[disk_folder] \
>     config.DATA.TEST.DATASET_NAMES=[dummy_data_folder] \
>     config.DATA.TEST.BATCHSIZE_PER_REPLICA=16 \
>     config.DATA.TEST.DATA_PATHS=["/home/siddhaganju/recursion-ssl/rxrx1/images"] \
>     config.DISTRIBUTED.NUM_NODES=1 \
>     config.DISTRIBUTED.NUM_PROC_PER_NODE=1 \
>     config.OPTIMIZER.num_epochs=2 \
>     config.OPTIMIZER.param_schedulers.lr.values=[0.01,0.001] \
>     config.OPTIMIZER.param_schedulers.lr.milestones=[1] \
>     config.HOOKS.TENSORBOARD_SETUP.USE_TENSORBOARD=true \
>     config.HOOKS.MEMORY_SUMMARY.PRINT_MEMORY_SUMMARY=false \
>     config.CHECKPOINT.DIR="/home/siddhaganju/recursion-ssl/rxrx1/fulltune_checkpoints" \
>     config.MODEL.WEIGHTS_INIT.PARAMS_FILE="" \
>     config.MODEL.WEIGHTS_INIT.APPEND_PREFIX="trunk._feature_blocks." \
>     config.MODEL.WEIGHTS_INIT.STATE_DICT_KEY_NAME=""

    
cfg = [
  'config=benchmark/fulltune/imagenet1k/eval_resnet_8gpu_transfer_in1k_fulltune.yaml',
  'config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/home/siddhaganju/recursion-ssl/rxrx1/checkpoints/model_phase147.torch', # Specify path for the model weights.
  'config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True', # Turn on model evaluation mode.
  'config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY=False', # Freeze trunk. 
  'config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_AND_HEAD=True', 
  'config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_TRUNK_AND_HEAD=True', 
  'config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY=True', # Extract the trunk features, as opposed to the HEAD.
  'config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=False', # Do not flatten features.
]
# Compose the hydra configuration.
cfg = compose_hydra_configuration(cfg)
# Convert to AttrDict. This method will also infer certain config options
# and validate the config is valid.
_, cfg = convert_to_attrdict(cfg)

model = build_model(cfg.MODEL, cfg.OPTIMIZER)

weights = load_checkpoint(checkpoint_path=cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE)

init_model_from_consolidated_weights(
    config=cfg,
    model=model,
    state_dict=weights,
    state_dict_key_name="classy_state_dict",
    skip_layers=[],  # Use this if you do not want to load all layers
)
print("Weights have loaded")

pipeline = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img_dir = "/path/to/my/data/test"

for img_name in os.listdir(img_dir):
    img_fname = os.path.join(img_dir, img_name)
    image = Image.open(img_fname).convert("RGB")
    x = pipeline(image)
    features = model(x.unsqueeze(0))
    _, pred = features[0].float().topk(1,  largest=True, sorted=True) 
print(img_fname, features[0][-1], pred[0])