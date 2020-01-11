from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

plot_dir = "plots/"

def describe(x):
	print('Type {} \t Shape {} \n {}'.format(x.type(), x.shape, x), '')



def ex1_1():
	# one hot encoding
	corpus = ['Time flies flies like an arrow.', 'Fruit flies like a banana']
	one_hot_vectorizer = CountVectorizer(binary=True)
	one_hot = one_hot_vectorizer.fit_transform(corpus).toarray()
	vocab = one_hot_vectorizer.get_feature_names()

	f, ax = plt.subplots()
	sns.heatmap(one_hot, annot=True, cbar=False, xticklabels=vocab, yticklabels=['Sentence 1', 'Sentence 2'])
	f.savefig(plot_dir + __file__[:-3] + "_1_1.pdf")


def ex1_2():
	# term frequency (inverse document) representation
	corpus = ['Time flies flies like an arrow.', 'Fruit flies like a banana']
	tfidvectorizer = TfidfVectorizer()
	tfidf = tfidvectorizer.fit_transform(corpus).toarray()

	vocab = tfidvectorizer.get_feature_names()
	
	f, ax = plt.subplots()
	sns.heatmap(tfidf, annot=True, cbar=False, xticklabels=vocab, yticklabels=['Sentence 1', 'Sentence 2'])
	f.savefig(plot_dir + __file__[:-3] + "_1_2.pdf")


# all about tensors

def ex1_3(): # initiaiting
	x = torch.rand(2, 3)  # uniform random FloatTensor
	y = torch.randn(2, 3) # random normal
	z = torch.zeros(2, 3) # zeros
	z1 = torch.ones(2, 3) # ones
	z5 = z1.fill_(5)	  # fives
	z6 = torch.Tensor([[1, 2, 3], [4, 5, 6]]) # by python list

	npy = np.random.rand(2, 3).astype(np.float32)
	z7 = torch.Tensor(torch.from_numpy(npy)) # now type DoubleTensor

	describe(z7)

def ex1_9(): # adding
	x = torch.randn(2, 3).fill_(2)
	y = torch.add(x, x)
	describe(y)
	z = x + y
	describe(z)

def ex1_10(): # transforming
	x = torch.arange(6)
	x = x.view(2, 3)

	describe(torch.sum(x, dim=0))
	describe(torch.sum(x, dim=1))
	describe(torch.transpose(x, 0, 1))


def ex1_11(): # slicing
	x = torch.arange(6).view(2, 3)
	describe(x[:1, :2])
	describe(x[0, 1])


def ex1_12(): # indexing
	x = torch.arange(6).view(2, 3)
	indices = torch.LongTensor([0, 2])
	describe(torch.index_select(x, dim=1, index=indices))

	indices = torch.LongTensor([0, 0])
	describe(torch.index_select(x, dim=0, index=indices))

	row_indices = torch.arange(2).long()
	col_indices = torch.LongTensor([0, 1])
	describe(x[row_indices, col_indices])

def ex1_13(): # concatenating
	x = torch.arange(6).view(2, 3)
	describe(torch.cat([x, x], dim=0))
	describe(torch.cat([x, x], dim=1))
	describe(torch.stack([x, x]))


def ex1_14(): #multiplying
	x1 = torch.arange(6).view(2, 3)
	x2 = torch.ones(3, 2)
	x2[:, 1] += 1
	describe(x2)
	describe(torch.mm(x1, x2))


def ex1_15(): # requires grad
	x = torch.ones(2, 2, requires_grad=True)
	describe(x)
	print(x.grad is None)

	y = (x + 2) * (x + 5) + 3
	describe(y)
	print(x.grad is None)

	z = y.mean()
	describe(z)
	z.backward()
	print(x.grad is None)

def ex1_16():
	if not torch.cuda.is_available():
		print("Cuda is not available")
		device = "cpu"
	else:
		device = torch.device("cuda")
	print("Device is set to:", device)

	x = torch.rand(3, 3).to(device)
	describe(x)


def main():
	print(datetime.now(), 'Started')

	ex1_16()

	print(datetime.now(), 'Finished')

if __name__ == "__main__":
	main()

























