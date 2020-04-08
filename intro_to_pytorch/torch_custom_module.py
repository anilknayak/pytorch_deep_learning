import torch

device = torch.device("cpu")
# device = torch.device("cuda:0") # if you want to run the model on GPU

class Relu(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward saves the information for Backprop
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super(Net, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, hidden_layer_size)
        self.layer2 = torch.nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        relu = Relu.apply
        outpur_pred = self.layer2(relu(self.layer1(x)))
        return outpur_pred

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
model = Net(input_size, hidden_layer_nodes, output_size)

# Apply loss function
loss_fn = torch.nn.MSELoss(reduction='sum')

# Declare optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

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


