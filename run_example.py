import grad_CAM_pipeline

# Image directory
img_dir = '/Users/august/ETSN20/images/day'

# Put custom model and weights in same folder
model_dir = '/Users/august/ETSN20/Tunnel_model.h5'
weight_dir = model_dir.split('.h5')[0] + '_weights.h5' # name of weights file should be like model name + _weights.h5

# Use for custom model
labels = ['Empty', 'Person', 'Dog', 'Bike']

# Custom model
grad_CAM_pipeline.run_pipeline(model_dir=model_dir, img_dir=img_dir, weight_dir=weight_dir, labels = labels)

# Tensorflow included model
# grad_CAM_pipeline.run_pipeline(model_dir='ResNet50', img_dir=img_dir)
