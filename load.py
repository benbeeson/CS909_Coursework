from bs4 import BeautifulSoup
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import sys
import re

# An instance of the reuters data containing the topics, its title, text, pos tags, training?
class Instance:
    def __init__(self, topics, title, text, train):
        self.topic = topics
        self.title = title
        self.text = text
        self.tags = []
        self.processedText = ""
        self.isTraining = train

    def __str__(self):
		return "(" + self.topic + ") "

instances = []

# Loads a specific Reuter's data set
def loadDataset(name):
	print "Loading dataset: " + name
	soup = BeautifulSoup(open(name))
	
	for reut in soup.find_all('reuters'):
		if(reut.get('lewissplit') == "NOT-USED"):
			continue

		topic = reut.topics
		if topic != None:
			topic = map((lambda d: str(d.string)), topic.findChildren())
		title = reut.title
		if title != None:
			title = title.string
		body = reut.body
		if body != None:
			body = body.string
		
		inst = Instance(topic, title, body, (reut.get('lewissplit') == "TRAIN")) 
		instances.append(inst)
	

def loadAllDatasets():
	for i in range(int(sys.argv[1])):
		loadDataset("reuters21578/reut2-0" + str(i).rjust(2, "0") + ".sgm")

# Remove any stories in the dataset that do not have any topics
def filterNoTopics(instances):
	return filter((lambda inst: len(inst.topic) > 0), instances)

# Strip unneccessary data from a string of test
def stripDate(text):
	text = re.sub(r'[0-9]{1,2}\/[0-9]{1,2}\/[0-9]{1,2}', 'DATE', text)
	return text
def stripNumeric(text):
	text = re.sub(r'[0-9]+(\.[0-9]+)?', 'NUMERIC', text)
	return text
def stripPunctuation(text):
	text = re.sub(r'[\t\n-]', ' ', text.lower())
	text = re.sub(r'[^a-z\. ]', '', text)
	text = re.sub(r'[ ]+', ' ', text)
	return text

# Clean the body of text / title for a set of instances
def cleanInstances(instances):
	for inst in instances:
		if inst.title != None:
			inst.title = stripPunctuation(stripNumeric(stripDate(inst.title)))
		if inst.text != None:
			inst.text = stripPunctuation(stripNumeric(stripDate(inst.text)))
	return instances

# Tokenise and POS tag words in an instance
def tagInstance(inst):
	if inst.text == None:
		inst.tags = []
		return inst
	tokens = nltk.word_tokenize(inst.text)
	tags = nltk.pos_tag(tokens)
	if tags != None:
		inst.tags = tags
	return inst

# Remove any stopwords from an instance
def filterStopWords(inst):
	if inst.tags == None:
		return inst
	stopwords = nltk.corpus.stopwords.words('english')
	tags = filter((lambda tag: not stopwords.__contains__(tag[0])), inst.tags)
	inst.tags = tags
	return inst

# Perform stemming on an instance
def performStemming(inst):
	if inst.tags == None:
		return inst
	lmtzr = WordNetLemmatizer()
	tags = map((lambda tag: (lmtzr.lemmatize(str(tag[0]).replace(".", ""), ("v" if tag[1][0] == "V" else "n")), tag[1])), inst.tags)
	inst.tags = tags
	return inst

def tagToString(inst):
	return " ".join(map((lambda tag: str(tag[0]).replace(".", "") + "-" + tag[1]), inst.tags))

# Convert a preprocessed instance back to an xml format for saving
def instanceToXML(inst):
	xml = "<INSTANCE topics=\"" + str(inst.topic) + "\" title=\"" + ("" if inst.title == None else inst.title) + "\" training=\"" + str(inst.isTraining) + "\">"
	if hasattr(inst, 'tags'):
		xml += str(inst.tags)
	xml += "</INSTANCE>"
	return xml

# Perform POS tagging -> stopword removal -> stemming on set of instances
def preProcessInstances(instances):
	for inst in instances:
		inst.processedText = tagToString(performStemming(filterStopWords(tagInstance(inst))))
	return instances

# Preprocess all the reuters files into new xml files stored in ./processed/reutX.xml
def loadAllDatasets():
	global instances

	for i in range(int(sys.argv[1])):
		instances = []
		loadDataset("reuters21578/reut2-0" + str(i).rjust(2, "0") + ".sgm")
		try:
			instances = filterNoTopics(instances)
			instances = cleanInstances(instances)
			instances = preProcessInstances(instances)
			f = open('processed/reut' + str(i) + '.xml','w')
			f.write("\n".join(map((lambda inst : instanceToXML(inst)), instances)))
			f.close()
			print("Completed dataset: reuters21578/reut2-0" + str(i).rjust(2, "0") + ".sgm")
		except Exception as e:
			print("ERROR dataset: reuters21578/reut2-0" + str(i).rjust(2, "0") + ".sgm")
			print(e)

loadAllDatasets()