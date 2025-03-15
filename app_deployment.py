import torch
from model.model import SimpleNN  # This should be the correct format

# Input from the user (Make sure to get all 32 inputs if your model expects 32)
glucose = float(input("Enter the glucose level: "))
slow_insulin = float(input("Enter the slow insulin: "))
physical_effort = float(input("Enter the physical effort (0-10): "))
movement = float(input("Is glucose going up (1), Down (-1), Or stable (0): "))
meal = float(input("Did you eat yes(1) no(0): "))

# Combine the features into one list (adjust the number of features according to your model)
inputs = [glucose, slow_insulin, physical_effort, movement, meal]

# Initialize the model again (same architecture as when saving)
model = SimpleNN(input_dim=5)  # Adjust to 32 if your model expects 32 features
model.load_state_dict(torch.load('model.pth'))
model.eval()  # Set the model to evaluation mode

# Prepare the input data
new_data = torch.tensor([inputs], dtype=torch.float32)

# Get the prediction
with torch.no_grad():  # No need to track gradients for inference
    predicted_value = model(new_data)

predicted_value = predicted_value.item()
predicted_value = abs(predicted_value)  # Ensure the value is positive
predicted_value = round(predicted_value)  # Round to the nearest integer

print("Predicted Insulin to Inject:", predicted_value)
