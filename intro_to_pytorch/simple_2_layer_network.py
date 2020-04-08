import torch

device = torch.device("cpu")
# device = torch.device("cuda:0") # if you want to run the model on GPU

epochs = 100
batch_size = 2
input_size = 100
hidden_layer_nodes = 200
output_size = 10
learning_rate = 0.0001

# Create random input and output data
input = torch.randn(batch_size, input_size, device=device, dtype=torch.float)
output = torch.randn(batch_size, output_size, device=device, dtype=torch.float)

# Randomly initialize weights
w1 = torch.randn(input_size, hidden_layer_nodes, device=device, dtype=torch.float)
w2 = torch.randn(hidden_layer_nodes, output_size, device=device, dtype=torch.float)

for epoch in range(epochs):
    # Forward pass:
    layer1_out = input.mm(w1)
    layer1_relu = layer1_out.clamp(min=0)
    output_pred = layer1_relu.mm(w2)

    # MSE Loss
    diff = (output_pred - output)
    loss = diff.pow(2).sum().item()
    print("Epoch {}: loss {}".format(epoch, loss))

    # Backprop
    gradient_output_pred = 2.0 * diff
    gradient_w2 = layer1_relu.t().mm(gradient_output_pred)
    gradient_layer1_relu = gradient_output_pred.mm(w2.t())
    gradient_layer1_out = gradient_layer1_relu.clone()

    # Make gradient to zero when layer1_out < 0
    gradient_layer1_out[layer1_out < 0] = 0

    gradient_w1 = input.t().mm(gradient_layer1_out)

    # Update weights using gradient descent
    w1 -= learning_rate * gradient_w1
    w2 -= learning_rate * gradient_w2
