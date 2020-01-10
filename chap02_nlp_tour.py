import torch
import spacy
from nltk.tokenize import TweetTokenizer
from datetime import datetime

nlp = spacy.load('en_core_web_sm')

def describe(x):
	print('Type {} \t Shape {} \n {}'.format(x.type(), x.shape, x), '')


def ex2_1():
	text = "Mary, don't slap the green witch"

	print("Text example:", text)
	print("Tokenising with spacy: ", [str(token) for token in nlp(text.lower())])

	tweet = "Snow White and the Seven Degrees #MakeAMovieCold @midnight :)"

	tokenizer = TweetTokenizer()
	print("Tweet example:", tweet)
	print("Tokenising with nltk:", tokenizer.tokenize(tweet.lower()))
	print("Tokenising with scipy", [str(token) for token in nlp(tweet.lower())])


def n_grams(text, n):
	return [text[i: i + n] for i in range(len(text) - n + 1)]

def ex2_2():
	# create trigrams
	text = ['mary', 'n', 'slap', 'green', 'witch', '.']
	print(n_grams(text, 3))


def ex2_3():
	# lemmatisation
	doc = nlp("he was running late")
	for token in doc:
		print('{} --> {}'.format(token, token.lemma_))

def ex2_4():
	# pos tagging (part-of-speech)
	doc = nlp("Mary slapped the green witch.")
	for token in doc:
		print('{} --> {}'.format(token, token.pos_))

def ex2_5():
	# chunking
	doc = nlp("Mary slapped the green witch.")
	for chunk in doc.noun_chunks:
		print('{} --> {}'.format(chunk, chunk.label_))

def main():
	print(datetime.now(), 'Started')

	ex2_4()

	print(datetime.now(), 'Finished')

if __name__ == "__main__":
	main()

























