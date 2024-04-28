from imports import *
from tqdm import tqdm, tqdm_notebook
import torch

DEVICE = torch.device('cuda')

def train(net, data_object, train_loader, test_loader, loss_fn, optimizer, scheduler, epochs):
    train_losses = []  # List to store loss of each training epoch
    test_losses = []  # List to store loss of each testing epoch

    with tqdm(range(epochs), desc="Epoch Progress") as epoch_pbar: # Main progress bar for epochs
        
        for epoch in epoch_pbar: # Loop over the specified number of epochs, showing progress with tqdm
            # Check if it is the first epoch and if training is 'Noise2Clean' to perform pre-training evaluation
            if epoch == 0 and data_object.training_type == "Noise2Clean":
                print("Pre-training evaluation")
                # Calculate initial metrics and losses (commented out part is for an alternative initial test)
                testmet = getMetricsonLoader(test_loader, net, False)  # Evaluate the model on the test loader

                # Write initial test metrics to a results file
                with open(data_object.basepath + "/results.txt", "w+") as f:
                    f.write("Initial : \n")
                    f.write(str(testmet))
                    f.write("\n")
                    
            # Train the model for one epoch using the training loader and update the model weights
            train_loss = train_epoch(net, train_loader, loss_fn, optimizer)
            test_loss = 0  # Initialize test loss
            
            # Evaluate the model on the test loader without updating weights
            with torch.no_grad():
                test_loss, testmet = test_epoch(net, test_loader, loss_fn, use_net=True)

            scheduler.step()  # Update the learning rate based on the scheduler
            print("Saving model....")
            
            # Store the losses from the current epoch
            train_losses.append(train_loss)
            test_losses.append(test_loss)

            # Update the epoch progress bar with latest information
            epoch_pbar.set_postfix({'Train Loss': train_loss, 'Test Loss': test_loss})

            # Write the metrics for the current epoch to the results file
            with open(data_object.basepath + "/results.txt", "a") as f:
                f.write("Epoch :" + str(epoch + 1) + "\n" + str(testmet))
                f.write("\n")

            print("OPed to txt")  # Output to text completion message

            # Save the model and optimizer states to files
            torch.save(net.state_dict(), data_object.basepath + '/Weights/dc20_model_' + str(epoch + 1) + '.pth')
            torch.save(optimizer.state_dict(), data_object.basepath + '/Weights/dc20_opt_' + str(epoch + 1) + '.pth')
            print("Models saved")

            # Clean up GPU memory and collect garbage to free up resources
            torch.cuda.empty_cache()
            gc.collect()
    
    # Return the last training and testing losses
    return train_losses, test_losses

def train_epoch(net, train_loader, loss_fn, optimizer):
    net.train()  # Set the network to training mode (this enables features like dropout)
    train_ep_loss = 0.  # Initialize training loss accumulator
    counter = 0  # Initialize a counter to keep track of the number of batches processed
    # Add tqdm progress bar for the batches
    with tqdm(total=len(train_loader), desc="Training Batches", leave=False) as pbar:
        for noisy_x, clean_x in train_loader:  # Iterate over batches of noisy and clean data
            noisy_x, clean_x = noisy_x.to(DEVICE), clean_x.to(DEVICE)  # Move data to the device (GPU or CPU)

            net.zero_grad()  # Zero the gradients to prevent accumulation from previous iterations

            pred_x = net(noisy_x)  # Forward pass: compute the predicted outputs by passing noisy input to the model

            loss = loss_fn(noisy_x, pred_x, clean_x)  # Compute the loss between the predicted and true outputs
            loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters
            optimizer.step()  # Perform a single optimization step (parameter update)

            train_ep_loss += loss.item()  # Accumulate the loss over batches
            counter += 1  # Increment the batch counter
            pbar.update(1)  # Update the progress bar for each batch

    train_ep_loss /= counter  # Calculate the average loss over all batches

    gc.collect()  # Garbage collection to free memory
    torch.cuda.empty_cache()  # Clear cache of the GPU to free memory
    return train_ep_loss  # Return the average training loss for this epoch

def test_epoch(net, test_loader, loss_fn, use_net=True):
    net.eval()  # Set the network to evaluation mode (this disables features like dropout)
    test_ep_loss = 0.  # Initialize test loss accumulator
    counter = 0.  # Initialize a counter to keep track of the number of batches processed

    # The following commented-out block would normally calculate test loss per batch

    testmet = getMetricsonLoader(test_loader, net, use_net)  # Evaluate the model using custom metrics function

    gc.collect()  # Garbage collection to free memory
    torch.cuda.empty_cache()  # Clear cache of the GPU to free memory

    return test_ep_loss, testmet  # Return the accumulated test loss and evaluation metrics

