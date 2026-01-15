import pytest
import torch
from main import train_transform, test_transform, train_dataset, val_dataset, train_loader, val_loader, test_loader, processor, model, loss_function, optimizer, train, test, device, epochs, num_classes, class_names

# Mock dataset creation since the original code relies on external files
class MockDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=10, num_classes=4):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.data = torch.randn(num_samples, 3, 224, 224)
        self.targets = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

@pytest.fixture
def mock_dataloaders():
    # Replace the actual datasets with mock datasets for testing
    mock_train_ds = MockDataset(num_samples=100, num_classes=num_classes)
    mock_val_ds = MockDataset(num_samples=20, num_classes=num_classes)
    mock_test_ds = MockDataset(num_samples=30, num_classes=num_classes)

    # Use mock datasets to create mock dataloaders
    mock_train_loader = torch.utils.data.DataLoader(mock_train_ds, batch_size=16, shuffle=True)
    mock_val_loader = torch.utils.data.DataLoader(mock_val_ds, batch_size=16, shuffle=True)
    mock_test_loader = torch.utils.data.DataLoader(mock_test_ds, batch_size=16, shuffle=True)

    return mock_train_loader, mock_val_loader, mock_test_loader

def test_transforms():
    # Test if transforms are created and are of the correct type
    assert isinstance(train_transform, torch.nn.Sequential)
    assert isinstance(test_transform, torch.nn.Sequential)

def test_dataset_loading_and_splitting():
    # Test if datasets and dataloaders are initialized
    # Note: These tests assume the existence of '/kaggle/input/brain-tumor-classification-mri/Testing' and '/kaggle/input/brain-tumor-classification-mri/Training'
    # For a truly runnable test without these files, we'd need to mock ImageFolder as well, which is complex.
    # We'll proceed assuming the directories exist for this test.
    try:
        assert len(train) > 0
        assert len(test) > 0
        assert len(train_dataset) + len(val_dataset) == len(train)
        assert len(train_loader.dataset) > 0
        assert len(val_loader.dataset) > 0
        assert len(test_loader.dataset) > 0
    except RuntimeError as e:
        # Handle cases where the directories might not exist during local testing
        print(f"Skipping dataset loading tests due to missing directories: {e}")
        pytest.skip("Skipping dataset loading tests due to missing directories.")

def test_model_initialization():
    # Test if the model and processor are initialized correctly
    assert processor is not None
    assert isinstance(model, torch.nn.Module)
    assert model.classifier is not None
    assert model.classifier.out_features == num_classes
    assert next(model.parameters()).device.type == device.type

def test_loss_function_optimizer_initialization():
    # Test if loss function and optimizer are initialized
    assert isinstance(loss_function, torch.nn.CrossEntropyLoss)
    assert isinstance(optimizer, torch.optim.Adam)


def test_training_step(mock_dataloaders):
    # Test a single training step
    mock_train_loader, _, _ = mock_dataloaders
    if len(mock_train_loader.dataset) == 0:
        pytest.skip("Mock dataloader is empty.")

    x, y = next(iter(mock_train_loader))
    x, y = x.to(device), y.to(device)

    initial_params = [p.clone() for p in model.parameters()]

    optimizer.zero_grad()
    output = model(x)
    logits = output.logits
    batch_loss = loss_function(logits, y)
    batch_loss.backward()
    optimizer.step()

    # Check if parameters have changed
    params_changed = False
    for p1, p2 in zip(model.parameters(), initial_params):
        if not torch.equal(p1, p2):
            params_changed = True
            break
    assert params_changed, "Model parameters did not change after a training step."

def test_validation_step(mock_dataloaders):
    # Test a single validation step
    _, mock_val_loader, _ = mock_dataloaders
    if len(mock_val_loader.dataset) == 0:
        pytest.skip("Mock dataloader is empty.")

    x, y = next(iter(mock_val_loader))
    x, y = x.to(device), y.to(device)

    model.eval()
    with torch.no_grad():
        output = model(x)
        logits = output.logits
        val_loss = loss_function(logits, y)

    assert val_loss is not None
    assert val_loss.item() >= 0

def test_classification_report_generation(mock_dataloaders):
    # Test if classification report can be generated (without checking specific values)
    # We need to ensure model is in eval mode and no_grad context for this.
    mock_train_loader, _, mock_test_loader = mock_dataloaders

    # Dummy data for predictions and true labels if test_loader is empty
    if len(mock_test_loader.dataset) == 0:
        all_predictions = [0, 1, 0]
        all_true_labels = [0, 0, 1]
        class_names_for_report = [f"class_{i}" for i in range(num_classes)]
        print("Using dummy data for classification report test as test_loader is empty.")
    else:
        all_predictions = []
        all_true_labels = []
        model.eval()
        with torch.no_grad():
            for x, y in mock_test_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                logits = output.logits
                predictions = logits.argmax(dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_true_labels.extend(y.cpu().numpy())
        class_names_for_report = class_names # Use actual class names if available

    if not all_true_labels: # Ensure there's data to report on
        pytest.skip("No data available for classification report.")

    report = classification_report(all_true_labels, all_predictions, target_names=class_names_for_report, digits=4)
    assert isinstance(report, str)
    assert "classification report" in report.lower()

# Note: The actual training loop is too complex to test in isolation with pytest
# without extensive mocking. The tests above focus on individual components and steps.
