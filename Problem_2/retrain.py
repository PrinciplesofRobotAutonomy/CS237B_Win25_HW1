import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.models import Inception_V3_Weights
from utils import *
from torch.utils.tensorboard import SummaryWriter

BATCH_SIZE = 100
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
lr = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_bottleneck_dataset(model, img_dir):
    transform = transforms.Compose([])

    ######### Your code starts here #########
    train_dataset = ...
    train_dataloader = ...

    ######### Your code ends here ###########
    bottleneck_x_l = []
    bottleneck_y_l = []

    print("Generating Bottleneck Dataset... this may take some minutes.")

    model.eval()
    for image, label, _ in train_dataloader:
        if image is None:
            continue
        image = image.to(device).permute(0, 3, 1, 2)


        ######### Your code starts here #########


        ######### Your code ends here ###########

    bottleneck_x = np.vstack(bottleneck_x_l)
    bottleneck_y = np.vstack(bottleneck_y_l)
    
    bottleneck_ds = torch.utils.data.TensorDataset(
        torch.tensor(bottleneck_x, dtype = torch.float32), 
        torch.tensor(bottleneck_y, dtype=torch.long)
    )

    return bottleneck_ds, len(train_dataset)

def retrain(image_dir):
    # Create the base model from the pre-trained model InceptionV3
    base_model = models.inception_v3(weights = Inception_V3_Weights.DEFAULT, transform_input=False)
    base_model.fc = nn.Identity()  # Remove the last fully-connected layer
    base_model.to(device)
    base_model.eval()

    bottleneck_train_ds, num_train = get_bottleneck_dataset(base_model, img_dir=image_dir)
    train_dataloader = DataLoader(
        bottleneck_train_ds, batch_size=BATCH_SIZE, shuffle=True,
    )

    print("Done generating Bottleneck Dataset")
    
    ######### Your code starts here #########
    # We want to create a linear classifier which takes the bottleneck data as input
    # 1. Get the size of the bottleneck tensor.
    # 2. Define a new neural network model which is a linear classifier
    #   2.1 Define a linear layer (retrain_linear):
    #       - Inputs are the bottleneck tensors
    #       - Outputs are the LABELS -> [cat, dog, neg]
    #   2.3 Define the activation function (retrain activation)
    #   2.4 Create a new model and move it to the right device
    # 3. Define a loss function and a optimization scheme


    ########################### Training Loop #######################################
    writer = SummaryWriter("logs")  # Initialize TensorBoard
    EPOCHS = 1000 # Feel free to adjust this to obtain a lower loss
    STEPS_PER_EPOCH = 50  # Number of steps per epoch

    print("Begin Training...")

    retrain_model.train()
    for epoch in range(EPOCHS):
        running_loss = 0.0
        step = 0
        while step < STEPS_PER_EPOCH:
            for i, (bottleneck_inputs, labels) in enumerate(train_dataloader, 0):
                if step >= STEPS_PER_EPOCH:
                    break

                bottleneck_inputs = bottleneck_inputs.to(device)
                labels = labels.squeeze().to(device)  # Keep labels as integers
                ######### Your code starts here #########
                # Perform the training loop
                # Zero out the gradients
                # Get the retrain_model outputs from the bottleneck_input data
                # Compute the loss, backpropagate and update the weights



                ######### Your code ends here #########

                running_loss += loss_val.item()
                step += 1

        if epoch % (EPOCHS // 10) == 0:
            print(f"Epoch: {epoch}, Avg loss: {running_loss / STEPS_PER_EPOCH:.4f}")
            writer.add_scalar("Loss/train", running_loss / STEPS_PER_EPOCH, epoch)

    writer.close()  # Close TensorBoard writer


    ######### Your code starts here #########
    # Create a combined model using nn.Sequential 
    # that combines the base_model and retrain_model
    

    ######### Your code ends here #########        

    print("Saving model...")
    maybe_makedirs("trained_models")
    torch.save(model, "trained_models/trained.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str)
    FLAGS, _ = parser.parse_known_args()
    retrain(FLAGS.image_dir)