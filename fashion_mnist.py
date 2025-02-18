import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from clearml import Task

# Initialize ClearML task
task = Task.init(project_name="FashionMNIST", task_name="PyTorch CNN Training")

# This function will help in logging and experiment tracking on ClearML Webview
task.connect({
    "batch_size": 64,
    "epochs": 100,
    "learning_rate": 0.01,
    "momentum": 0.5,
    "cuda_enabled": torch.cuda.is_available(),
    "log_interval": 10
})

# This will convert Tensor and normalize pixel values
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# This function will automatically download the FashionMNIST dataset
trainset = torchvision.datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
testset = torchvision.datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

# Create Data loader for processing in batches
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# Define CNN model with every layers
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Max pooling layer to reduce the size of the feature maps
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layers mapping of 64*7*7 to 128 neurons
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        # Output layer with 10 classes
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25) # Dropout for regularization
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x))) # Applies the first layer
        x = self.pool(self.relu(self.conv2(x))) # Applies the second layer
        x = x.view(-1, 64 * 7 * 7) # Flattens the feature maps for dense layers
        x = self.relu(self.fc1(x)) # Apply first fully connected layer
        x = self.dropout(x) # Applies dropout
        x = self.fc2(x) # This is the Output layer
        return x

# Model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# This is the training function
def train_model(model, trainloader, epochs=100):
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device) # This will move data to device
            
            optimizer.zero_grad() # Resets the gradients
            outputs = model(images) # Forward pass
            loss = criterion(outputs, labels) # This will comput the loss
            loss.backward() # Backpropagation
            optimizer.step() # This will update the weights of the bias
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1) # Getting the predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader):.4f}, Accuracy: {accuracy:.2f}%")
        # This will log training in metrics
        task.get_logger().report_scalar("Training Loss", "Loss", iteration=epoch, value=running_loss/len(trainloader))
        task.get_logger().report_scalar("Training Accuracy", "Accuracy", iteration=epoch, value=accuracy)
    
    print("Training complete.")

# Run training
train_model(model, trainloader)

# Function to evaluate model performance
def evaluate_model(model, testloader):
    model.eval() # Setting the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad(): # Here we have disabled gradient calculation for faster processing
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # This will log the test accuracy onto the terminal
    task.get_logger().report_scalar("Test Accuracy", "Accuracy", iteration=1, value=accuracy)

evaluate_model(model, testloader)
