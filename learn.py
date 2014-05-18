from optparse import OptionParser

from bs4 import BeautifulSoup
import sys
import re
import math
import numpy as np
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize
from sklearn.decomposition import TruncatedSVD
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.cross_validation import KFold

from gensim import corpora
from gensim.models.ldamodel import LdaModel

from sklearn.lda import LDA

from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import Ward

import warnings
warnings.simplefilter("ignore")

class Instance:
    def __init__(self, topics, title, tags, training):
        self.topic = topics
        self.title = title
        self.tags = tags
        self.bag = []
        self.training = training

    def __str__(self):
		return "(" + self.topic + ") " + self.title

instances = []

# Loads a specific Reuter's file
def loadDataset(name):
	print "Loading dataset: " + name
	soup = BeautifulSoup(open(name))
	
	for reut in soup.find_all('instance'):
		topic = eval(reut.get('topics'))

		inst = Instance(topic, reut.get('title'), eval(reut.string), eval(reut.get("training"))) 
		instances.append(inst)
	
# Load all the files from the reuters dataset
def loadAllDatasets(count):
	for i in range(count):
		loadDataset("processed/reut" + str(i) + ".xml")

# Normalise the dataset so instances with multiple topics are repeated, once for each topic
def normaliseDatasets(instances):
	insts = []
	for inst in instances:
		for topic in inst.topic:
			insts.append(Instance([topic], inst.title, inst.tags, inst.training))

	return insts

# Print a summary of the distribution of the classes from a set of instances
def printSummary(instances):
	print("Summary of Dataset")
	print("Total of instances:\t" + str(len(instances)))

	classes = getClasses(instances)
	for cls in classes:
		count = len(filter((lambda inst: inst.topic[0].__contains__(cls)), instances))
		print("Number of '" + cls + "':\t" + str(count))

def getClasses(instances):
	classes = []
	for inst in instances:
		if not classes.__contains__(inst.topic[0]):
			classes.append(inst.topic[0])
	return classes

# Filter the dataset to a list of topics
def filterTopics(instances, topics):
	return filter((lambda inst: topics.__contains__(inst.topic[0])), instances)

# Build up the corpus from the loaded files
def buildCorpus(instances):
	text = []
	training = []
	label = []
	for inst in instances:
		words = map((lambda tag: tag[0] + "-" + tag[1]), inst.tags)
		t = " ".join(words)
		text.append(t)
		training.append(inst.training)
		label.append(inst.topic[0])
		
	return (text, training, label)

# Split the loaded corpus into training or test sets depending on the training flag
def subsetCorpusBag(corpus, bag, training):
	text = []
	label = []
	words = bag

	for i in range(0, len(corpus[0])):
		if((training and corpus[1][i]) or (not training and not corpus[1][i])):
			text.append(words[i])
			label.append(corpus[2][i])

	return (text, label)

def printLine():
	print("-" * 80)

# Train and test a model
def testModel(model, training, test):
	global classes

	print("Testing Model: " + str(model.__class__.__name__))
	X = training[0]
	Y = training[1]
	model.fit(X, Y)

	X = test[0]
	Y = test[1]
	prediction = model.predict(X)

	prfs = metrics.precision_recall_fscore_support(Y, prediction, labels=classes)

	print("Class\tAccuracy\tPrecision\tRecall\tFScore")
	for c in range(0, 10):
		a = metrics.accuracy_score(Y, prediction)
		p = prfs[0][c]
		r = prfs[1][c]
		f = prfs[2][c]
		print(str(classes[c]) + "\t" + str(a) + "\t" + str(p) + "\t" + str(r) + "\t" + str(f))

	printLine()

# Perform k cross fold validation on a model
def kcrossfold(model, training, K):
	print("Testing Model: " + str(model.__class__.__name__))
	fold = 1
	foldAvgs = []
	foldMatrix = []
	global classes

	for train, test in KFold(len(training[0]), K):
		print("Fold: " + str(fold))
		X = [training[0][i] for i in train]
		Y = [training[1][i] for i in train]
		model.fit(X, Y)

		X = [training[0][i] for i in test]
		Y = [training[1][i] for i in test]
		prediction = model.predict(X)
		avg = metrics.accuracy_score(Y, prediction)

		foldMatrix.append((metrics.confusion_matrix(Y, prediction, labels=classes), metrics.precision_recall_fscore_support(Y, prediction, labels=classes)))

		print(metrics.classification_report(Y, prediction))
		print("Accuracy: " + str(avg))

		fold += 1
		foldAvgs.append(avg)
		printLine()

	printLine()

	# Print the raw data for each class in the data
	print("Model\tFold\tClass\tTP\tFP\tFN\tAccuracy\tPrecision\tRecall\tF")
	for i in range(0, K):
		matrix = foldMatrix[i][0]
		prfs = foldMatrix[i][1]
		for c in range(0, 10):
			tp = truePositive(matrix, c)
			fp = falsePositive(matrix, c, classes)
			fn = falseNegative(matrix, c, classes)
			p = prfs[0][c]
			r = prfs[1][c]
			f = prfs[2][c]
			print(model.__class__.__name__ + "\t" + str(i + 1) + "\t" + str(classes[c]) + "\t" + str(tp) + "\t" + str(fp) + "\t" + str(fn) + "\t" + str(foldAvgs[i]) + "\t" + str(p) + "\t" + str(r) + "\t" + str(f))

	printLine()

	# Print fold averages along with mean, std and confidence interval
	print("Fold\t"+str(model.__class__.__name__))
	for i in range(0,K):
		print(str(i+1) + "\t" + str(foldAvgs[i]))
	print("Mean\t"+str(np.mean(foldAvgs)))
	print("Std\t"+str(np.std(foldAvgs)))
	print("CI\t"+str(np.mean(foldAvgs) - 1.96 * np.std(foldAvgs)) + " - " + str(np.mean(foldAvgs) + 1.96 * np.std(foldAvgs)))
	printLine()

# Calculate TP / FP / FN for a given confusion matrix and class
def truePositive(matrix, cls):
	return matrix[cls][cls]

def falseNegative(matrix, cls, classes):
	total = 0
	for i in range(0, len(classes)):
		if i != cls:
			total += matrix[cls][i]
	return total

def falsePositive(matrix, cls, classes):
	total = 0
	for i in range(0, len(classes)):
		if i != cls:
			total += matrix[i][cls]
	return total

parser = OptionParser()
parser.add_option("-l", "--LDA", dest="lda", action="store_true",
                  help="Use the latent Dirichlet Allocation topic modelling for feature selection", default=False)
parser.add_option("-f", "--files", dest="files", action="store", type="int",
                  help="The total number of files to load (1-22)", default=22)
parser.add_option("-t", "--tfidf", dest="tfidf", action="store_true",
                  help="Use TF*IDF as part of feature selection", default=True)
parser.add_option("-c", "--count", dest="count", action="store_true",
                  help="Use Term Frequency as part of feature selection", default=False)
parser.add_option("-b", "--binary", dest="binary", action="store_true",
                  help="Use Binary Frequency as part of feature selection", default=False)
parser.add_option("-k", "--k", dest="kfold", action="store",
                  help="Set the number of folds in K-cross fold validation", default=10)

(options, args) = parser.parse_args()

# Load the reuters dataset and flatten so there each document has one topic
loadAllDatasets(options.files)
instances = normaliseDatasets(instances)
instances = shuffle(instances)
allInstances = instances


classes = ["earn", "acq", "money-fx", "grain", "crude", "trade", "interest", "ship", "wheat", "corn"]
instances = filterTopics(instances, classes)
printLine()
printSummary(instances)

corpus = buildCorpus(instances)

# LDA TOPIC MODELLING
if options.lda:
	print("LDA Topic Modelling")

	texts = []
	allwords = []
	numTopics = 50
	for inst in instances:
		words = map((lambda tag: tag[0]), inst.tags)
		texts.append(words)
		allwords += words

	tf = Counter(allwords)
	texts = map((lambda text: filter((lambda word: tf[word] > 1), text)), texts)

	dictionary = corpora.Dictionary(texts)
	corpus_lda = [dictionary.doc2bow(text) for text in texts]

	print("Building Model")

	lda = LdaModel(corpus_lda, id2word=dictionary, num_topics=numTopics)
	output = lda[corpus_lda]
	BagWords = []
	for doc in output:
		probs = [0] * numTopics
		for i, p in doc:
			probs[i] = p
		BagWords.append(probs)

	training = subsetCorpusBag(corpus, BagWords, True)
	test = subsetCorpusBag(corpus, BagWords, False)

#FEATURE SELECTION
if not options.lda:
	print("Feature Selection")

	# Select vectorising mmethod to use
	if options.count:
		print("Using term frequency")
		vectorizer = CountVectorizer(min_df=1, binary=False, ngram_range=(2,2))
	elif options.binary:
		print("Using binary frequency")
		vectorizer = CountVectorizer(min_df=1, binary=True, ngram_range=(2,2))
	else:
		print("Using TF*IDF")
		vectorizer = TfidfVectorizer(min_df=1, smooth_idf=True, ngram_range=(1,2))

	# Split data into training and test sets
	training = subsetCorpusBag(corpus, corpus[0], True)
	test = subsetCorpusBag(corpus, corpus[0], False)

	training = (normalize(vectorizer.fit_transform(training[0]).toarray()), training[1])
	test = (normalize(vectorizer.transform(test[0]).toarray()), test[1])

# TEST THE THREE CLASSIFIERS
printLine()
kcrossfold(MultinomialNB(), training, options.kfold)
kcrossfold(svm.LinearSVC(), training, options.kfold)
kcrossfold(RandomForestClassifier(n_estimators=10), training, options.kfold)


# CLASSIFY USING THE BEST CLASSIFIER
printLine()
#testModel(MultinomialNB(), training, test)
testModel(svm.LinearSVC(), training, test)
#testModel(RandomForestClassifier(n_estimators=10), training, test)


#CLUSTERING
corpus = buildCorpus(allInstances)
vectorizer = TfidfVectorizer(min_df=1, ngram_range=(2,2))
lsa = TruncatedSVD()
BagWords = vectorizer.fit_transform(corpus[0])
BagWords = lsa.fit_transform(BagWords)
BagWords = normalize(BagWords)

def buildCluster(model, corpus, words):
	print("Testing Cluster: " + str(model.__class__.__name__))
	X = words
	Y = corpus[2]
	model.fit(X)

	labels = model.labels_
	sil = metrics.silhouette_score(X, labels, metric='euclidean')
	hom = metrics.homogeneity_score(Y, labels)
	com = metrics.completeness_score(Y, labels)
	v = metrics.v_measure_score(Y, labels)
	ran = metrics.adjusted_rand_score(Y, labels)

	print("Silhouette: " + str(sil))
	print("Homogeneity: " + str(hom))
	print("Completeness: " + str(com))
	print("V-measure: " + str(v))
	print("Adjusted Rand-Index: " + str(ran))
	printLine()

num_clusters = len(set(corpus[2]))
print("Number of topics / clusters: " + str(num_clusters))
buildCluster(KMeans(n_clusters=num_clusters, max_iter=100), corpus, BagWords)
buildCluster(DBSCAN(eps=0.5, min_samples=100), corpus, BagWords)
buildCluster(Ward(n_clusters=num_clusters), corpus, BagWords)