import vissl
import tensorboard
import apex
import torch
import pickle

from omegaconf import OmegaConf
from vissl.utils.hydra_config import AttrDict
from vissl.models import build_model
from vissl.utils.hydra_config import compose_hydra_configuration, convert_to_attrdict
from classy_vision.generic.util import load_checkpoint
from vissl.utils.checkpoint import init_model_from_consolidated_weights
import os


# Config is located at vissl/configs/config/pretrain/simclr/simclr_8node_resnet.yaml.
# All other options override the simclr_8node_resnet.yaml config.

cfg = [
  'config=quick_1gpu_resnet50_simclr ',
  'config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/home/siddhaganju/recursion-ssl/rxrx1/checkpoints/model_phase147.torch', # Specify path for the model weights.
  # 'config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/home/siddhaganju/recursion-ssl/rxrx1/checkpoint_supervised/model_final_checkpoint_phase1.torch',
  # 'config.MODEL.WEIGHTS_INIT.PARAMS_FILE=/home/siddhaganju/recursion-ssl/rxrx1/fulltune_checkpoints/model_phase12.torch',
  'config.MODEL.FEATURE_EVAL_SETTINGS.EVAL_MODE_ON=True', # Turn on model evaluation mode.
  'config.MODEL.FEATURE_EVAL_SETTINGS.FREEZE_TRUNK_ONLY=True', # Freeze trunk. 
  'config.MODEL.FEATURE_EVAL_SETTINGS.EXTRACT_TRUNK_FEATURES_ONLY=True', # Extract the trunk features, as opposed to the HEAD.
  'config.MODEL.FEATURE_EVAL_SETTINGS.SHOULD_FLATTEN_FEATS=False', # Do not flatten features.
  'config.MODEL.FEATURE_EVAL_SETTINGS.LINEAR_EVAL_FEAT_POOL_OPS_MAP=[["res5avg", ["Identity", []]]]' # Extract only the res5avg features.
]

# Compose the hydra configuration.
cfg = compose_hydra_configuration(cfg)
# Convert to AttrDict. This method will also infer certain config options
# and validate the config is valid.
_, cfg = convert_to_attrdict(cfg)

model = build_model(cfg.MODEL, cfg.OPTIMIZER)

# Load the checkpoint weights.
print("Loading weights from: ", cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE)
weights = load_checkpoint(checkpoint_path=cfg.MODEL.WEIGHTS_INIT.PARAMS_FILE)


# Initializei the model with the simclr model weights.
init_model_from_consolidated_weights(
    config=cfg,
    model=model,
    state_dict=weights,
    state_dict_key_name="classy_state_dict",
    skip_layers=[],  # Use this if you do not want to load all layers
)

print("Weights have loaded")

from PIL import Image
import torchvision.transforms as transforms

def extract_features(path):
  image = Image.open(path)

  # Convert images to RGB. This is important
  # as the model was trained on RGB images.
  image = image.convert("RGB")

  # Image transformation pipeline.
  pipeline = transforms.Compose([
      transforms.CenterCrop(224),
      transforms.ToTensor(),
  ])
  x = pipeline(image)

  features = model(x.unsqueeze(0))
  features_shape = features[0].shape
  #print(f"Features extracted have the shape: { features_shape }")
  #flatten the output 
  return features[0].flatten()

#get all the images in the directories 
import glob
directory = "/home/siddhaganju/recursion-ssl/rxrx1/images/test/"

all_features = []
all_class_ids = []
all_wells = []

for root, dirs, files in os.walk(directory):
  # print(root, len(files))
  if len(files) > 0:
    for filename in files:
      if "w1" in str(filename):
        well = str(filename)[:3]
        print(filename, well)
        output = extract_features(os.path.join(root, filename))
        np_output = output.cpu().detach().numpy()
        all_features.append(output)
        all_wells.append(well)
        x = os.path.basename(root)
        all_class_ids.append(x)
      
# with open('/home/siddhaganju/recursion-ssl/all_features_supervised_SIMCLR.pkl', 'wb') as f:
#   pickle.dump(all_features, f)
with open('/home/siddhaganju/recursion-ssl/all_wells_PTSIMCLR.pkl', 'wb') as f:
  pickle.dump(all_wells, f)
# with open('/home/siddhaganju/recursion-ssl/all_class_ids_supervised_SIMCLR.pkl', 'wb') as f:
#   pickle.dump(all_class_ids, f)