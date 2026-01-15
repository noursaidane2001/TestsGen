# Documentation

## Global Variables

- `device`: The device (CPU or CUDA) to run computations on. Determined by `torch.cuda.is_available()`.
- `train_transform`: A `torchvision.transforms.Compose` object defining transformations applied to training images (resize, horizontal flip, to tensor, normalize).
- `test_transform`: A `torchvision.transforms.Compose` object defining transformations applied to testing images (resize, to tensor, normalize).
- `test`: A `torchvision.datasets.ImageFolder` instance for the testing dataset.
- `train`: A `torchvision.datasets.ImageFolder` instance for the training dataset.
- `train_dataset`, `val_dataset`: Subsets of the `train` dataset, created using `torch.utils.data.random_split` for training and validation.
- `train_loader`, `val_loader`, `test_loader`: `torch.utils.data.DataLoader` instances for the respective datasets, batching and shuffling data.
- `processor`: An `AutoImageProcessor` from the transformers library, configured for the "google/vit-base-patch16-224" model.
- `model`: An `AutoModelForImageClassification` from the transformers library, configured for "google/vit-base-patch16-224", with its classifier head adapted to the number of classes in the dataset.
- `num_classes`: An integer representing the total number of unique classes in the training dataset.
- `loss_function`: A `torch.nn.CrossEntropyLoss` instance for calculating the loss during training.
- `optimizer`: An `Adam` optimizer for updating model weights, configured with a learning rate and weight decay.
- `train_N`, `test_N`: Integers representing the number of samples in the training and testing loaders, respectively.
- `epochs`: An integer defining the number of training epochs.
- `class_names`: A list of strings representing the names of the classes in the dataset.

## Functions

(No standalone functions defined in the provided code snippet.)

## Training and Evaluation Loop

The script implements a standard deep learning training and evaluation loop:

1.  **Epoch Iteration**: The outer loop iterates for a specified number of `epochs`.
2.  **Training Phase**: 
    *   The model is set to training mode (`model.train()`).
    *   It iterates through batches in `train_loader`.
    *   For each batch, input data (`x`) and labels (`y`) are moved to the appropriate `device`.
    *   Model predictions (`logits`) are generated.
    *   Gradients are zeroed (`optimizer.zero_grad()`).
    *   The `loss_function` calculates the batch loss.
    *   Backpropagation (`batch_loss.backward()`) computes gradients.
    *   Optimizer steps (`optimizer.step()`) update model weights.
    *   Training loss and correct predictions are accumulated.
    *   Progress is printed periodically.
    *   After each epoch, average training loss and accuracy are printed.
3.  **Validation Phase**: 
    *   The model is set to evaluation mode (`model.eval()`).
    *   Gradient calculations are disabled (`torch.no_grad()`).
    *   It iterates through batches in `val_loader`.
    *   Input data and labels are moved to `device`.
    *   Model predictions are generated.
    *   Validation loss and correct predictions are accumulated.
    *   After each epoch, average validation loss and accuracy are printed.

## Classification Report

After training and validation, the script generates a detailed classification report:

1.  **Data Collection**: It iterates through the `test_loader` in evaluation mode (`model.eval()`, `torch.no_grad()`).
2.  **Prediction**: For each batch, it gets model predictions.
3.  **Storage**: Predictions and true labels are collected into lists (`all_predictions`, `all_true_labels`).
4.  **Report Generation**: The `sklearn.metrics.classification_report` function is called with the collected true labels, predictions, and class names to produce a comprehensive report including precision, recall, F1-score, and support for each class.
