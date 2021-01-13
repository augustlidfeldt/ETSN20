import grad_CAM_pipeline
import click
import os.path

@click.command()
@click.option('--model-path', required=True, type=click.Path(exists=True))
@click.option('--weights-path', required=True, type=click.Path(exists=True))
@click.option('--dataset-path', required=True, type=click.Path(exists=True))
def generate(model_path, weights_path, dataset_path):

    print("### Model file located: " + model_path)
    print("### Weights file located: " + weights_path)
    print("### Dataset folder located: " + dataset_path)

    # Image directory
    #img_dir = os.path.join('images', 'day')
    #print(img_dir)

    # Put custom model and weights in same folder
    #model_file = os.path.join('model', 'Tunnel_model.h5')
    #print(model_file)
    #weight_file = model_file.split('.h5')[0] + '_weights.h5' # name of weights file should be like model name + _weights.h5
    #print(weight_file)

    # Use for custom model
    labels = ['Empty', 'Person', 'Dog', 'Bike']

    # Custom model
    grad_CAM_pipeline.run_pipeline(model_path=model_path, weights_path=weights_path, dataset_path=dataset_path, labels=labels)

    # Tensorflow included model
    # grad_CAM_pipeline.run_pipeline(model_dir='ResNet50', img_dir=img_dir)

if __name__ == '__main__':
    generate()
