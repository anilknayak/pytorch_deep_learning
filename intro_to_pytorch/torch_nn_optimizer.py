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
input = torch.randn(batch_size, input_size)
output = torch.randn(batch_size, output_size)

# Create model
model = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_layer_nodes),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_layer_nodes, output_size),
)

# Apply loss function
loss_fn = torch.nn.MSELoss(reduction='sum')

# Declare optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    # Forward pass: simple matrix multiplication from input->hidden->output
    output_pred = model(input)

    # MSE Loss
    loss = loss_fn(output_pred, output)
    print("Epoch {}: loss {}".format(epoch, loss.item()))

    # Zero the gradients before running the backward pass.
    optimizer.zero_grad()

    # Backprop
    # Internally, all the parameters of each Module are stored in Tensors with requires_grad=True
    # so this call will compute gradients for all learnable parameters in the model
    loss.backward()

    # Optimizer makes an update to its parameters
    optimizer.step()


