import torch
import matplotlib.pyplot as plt 
from datetime import datetime
import seaborn as sns


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



def describe(x):
	print('Type {} \t Shape {} \n {}'.format(x.type(), x.shape, x), '')

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


def myplot(x, y, name, title):
	f, ax = plt.subplots()
	sns.scatterplot(x=x.detach().numpy(), y=y.detach().numpy())
	ax.set(xlabel='x', ylabel='f(x)', title=title)
	f.savefig(plot_dir + __file__[:-3] + "_" + name + ".pdf")


def main():
	print(datetime.now(), 'Started')

	ex3_6()

	print(datetime.now(), 'Finished')


if __name__ == "__main__":
	main()


