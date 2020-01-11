import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
from datetime import datetime
import seaborn as sns


plot_dir = "plots/"


class Perceptron(nn.Module):
	""" A perceptron is one linear layer """
	def __init__(self, input_dim: int):
		""" 
		Args: 
			input_dim(int): size of input features
		"""
		super(Perceptron, self).__init__()
		self.fc1 = nn.Linear(input_dim, 1)

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
	x = torch.range(-5, 5, 0.1)
	y = torch.sigmoid(x)
		
	f, ax = plt.subplots()
	sns.scatterplot(x=x, y=y)
	ax.set(xlabel='x', ylabel='f(x)', title="Sigmoid function")
	f.savefig(plot_dir + __file__[:-3] + "_3_2.pdf")

def ex3_3():
	# plotting the tanh function
	x = torch.range(-5, 5, 0.1)
	y = torch.tanh(x)
		
	f, ax = plt.subplots()
	sns.scatterplot(x=x, y=y)
	ax.set(xlabel='x', ylabel='f(x)', title="Tanh function")
	f.savefig(plot_dir + __file__[:-3] + "_3_3.pdf")


def main():
	print(datetime.now(), 'Started')

	ex3_3()

	print(datetime.now(), 'Finished')

if __name__ == "__main__":
	main()


