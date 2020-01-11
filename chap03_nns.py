import torch
import matplotlib.pyplot as plt 
from datetime import datetime
import seaborn as sns
import numpy as np

plot_dir = "plots/"

sns.set_style("whitegrid")


class Perceptron(torch.nn.Module):
	""" A perceptron is one linear layer """
	def __init__(self, input_dim: int):
		""" 
		Args: 
			input_dim(int): size of input features
		"""
		super(Perceptron, self).__init__()
		self.fc1 = torch.nn.Linear(input_dim, 1)

	def forward(self, x_in: torch.Tensor):
		""" The forward pass of the perceptron
		Args:
			x_in (torch.Tensor): an input data tensor
			x_in.shape should be (batch, num_features)
		Returns: 
			the resulting tensor. tensor.shape should be (batch,).
		"""
		return torch.sigmoid(self.fc1(x_in)).squeeze()


def ex3_2():
	# plotting the sigmoid function
	x = torch.arange(-5, 5, 0.1)
	y = torch.sigmoid(x)
	
	myplot(x, y, "3_2", "Sigmoid function")


def ex3_3():
	# plotting the tanh function
	x = torch.arange(-5, 5, 0.1)
	y = torch.tanh(x)
		
	myplot(x, y, "3_3", "Tanh function")


def ex3_4():
	# plotting the relu function
	x = torch.arange(-5, 5, 0.1)
	relu = torch.nn.ReLU()
	y = relu(x)
		
	myplot(x, y, "3_4", "ReLU function")


def ex3_5():
	# plotting the prelu function
	x = torch.arange(-5, 5, 0.1)
	prelu = torch.nn.PReLU(num_parameters=1)
	y = prelu(x)
		
	myplot(x, y, "3_5", "PReLU function")


def ex3_6():
	# plotting the softmax function
	x = torch.arange(-5, 5, 0.1).unsqueeze(0)
	softmax = torch.nn.Softmax(dim=1)
	y = softmax(x)
	print(x.shape, y.shape)
		
	myplot(x.squeeze(), y.squeeze(), "3_6", "Softmax function")


def ex3_7():
	# calculating MSE loss
	mse_loss = torch.nn.MSELoss()
	outputs = torch.randn(3, 5, requires_grad=True)
	targets = torch.randn(3, 5)
	loss = mse_loss(outputs, targets)
	print(loss)

def ex3_8():
	# cross entropy loss
	ce_loss = torch.nn.CrossEntropyLoss()
	outputs = torch.randn(3, 5, requires_grad=True)
	targets = torch.tensor([1, 0, 3], dtype=torch.int64)
	loss = ce_loss(outputs, targets)
	print(loss)

def ex3_9():
	# binary cross entropy loss
	bce_loss = torch.nn.BCELoss()
	sigmoid = torch.nn.Sigmoid()
	probabilities = sigmoid(torch.randn(4, 1, requires_grad=True))
	targets = torch.tensor([1, 0, 1, 0], dtype=torch.float32).view(4, 1)
	loss = bce_loss(probabilities, targets)
	print(probabilities)
	print(loss)

def myplot(x, y, name, title):
	f, ax = plt.subplots()
	sns.scatterplot(x=x.detach().numpy(), y=y.detach().numpy())
	ax.set(xlabel='x', ylabel='f(x)', title=title)
	f.savefig(plot_dir + __file__[:-3] + "_" + name + ".pdf")


def ex3_10():

	seed = 1337
	torch.manual_seed(seed)
	np.random.seed(seed)

	input_dim = 2
	lr = 0.001
	n_epochs = 12
	n_batches = 5

	perceptron = Perceptron(input_dim=input_dim)
	bce_loss = torch.nn.BCELoss()
	optimiser = torch.optim.Adam(params=perceptron.parameters(), lr=lr)
	losses = []

	# each epoch is a complete pass over the training data
	for epoch_i in range(n_epochs):
		for batch_i in range(n_batches):

			perceptron.zero_grad()
			x_data, y_target = get_toy_data(batch_size=1000)
			
			y_pred = perceptron.forward(x_data).squeeze()
			loss = bce_loss(y_pred, y_target)
			loss.backward()
			optimiser.step()

			# keep track of loss
			losses.append(loss.item())

	print("Loss values (should get smaller with time):", losses)


def get_toy_data(batch_size, left_center=(3, 3), right_center=(3, -2)):
	# copied from the original notebooks

	x_data = []
	y_targets = np.zeros(batch_size)
	for batch_i in range(batch_size):
		if np.random.random() > 0.5:
			x_data.append(np.random.normal(loc=left_center))
		else:
			x_data.append(np.random.normal(loc=right_center))
			y_targets[batch_i] = 1
	return torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_targets, dtype=torch.float32)


def main():
	print(datetime.now(), 'Started')

	ex3_10()

	print(datetime.now(), 'Finished')


if __name__ == "__main__":
	main()


