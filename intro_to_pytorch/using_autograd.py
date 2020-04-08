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
# Setting requires_grad=True indicates that we want to compute gradients
# Setting requires_grad=False indicates that we do not need to compute gradients

w1 = torch.randn(input_size, hidden_layer_nodes, device=device, dtype=torch.float, requires_grad=True)
w2 = torch.randn(hidden_layer_nodes, output_size, device=device, dtype=torch.float, requires_grad=True)

for epoch in range(epochs):
    # Forward pass: simple matrix multiplication from input->hidden->output
    output_pred = input.mm(w1).clamp(min=0).mm(w2)

    # MSE Loss
    loss = (output_pred - output).pow(2).sum()
    print("Epoch {}: loss {}".format(epoch, loss.item()))

    # Backprop
    # This call will compute the gradient of loss with respect to all Tensors with requires_grad=True.
    # w1.grad and w2.grad will hold the gradients of w1 and w2 respectively
    loss.backward()

    # Manually update weights using gradient descent
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad
        # Manually zero the gradients after updating weights
        w1.grad.zero_()
        w2.grad.zero_()

    # OR
    # torch.optim.SGD to optimize


