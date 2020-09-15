from pdb import set_trace
import pandas, torch, os, pickle, string

labelVecs = {"en": 0, "es": 1, "other": 2}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model params
embed_dim_words = 32
hid_dim_words = 32
embed_dim_chars = 16
hid_dim_chars = 16

def buildWordVec(wordSeries): # builds dictionary to map words to a vector
	wordVec = {}
	for index, word in wordSeries.iteritems():
		word = word.lower()
		if not word in wordVec:
			wordVec[word] = len(wordVec)
	
	wordVec[None] = len(wordVec) # used to handle unknown words
	return wordVec

def genWordVec(): # loads training data and builds vector dictionary
	train_data = loadTrainingData("code_switching/data/train_data.tsv", ["tweetID" , "userID", "start", "end", "token", "gold label"])
	wordVecs = buildWordVec(train_data["token"])
	return wordVecs

def loadTrainingData(dataLocation, cols, delim = "	"): # read tsv, convert to df
	with open(dataLocation, encoding="utf8") as f:
		content = [x.strip().split(delim) for x in f.readlines()]
	
	df = pandas.DataFrame(content, columns = cols)
	return df

def prepData(): # returns dataframe of tweets and labels assigned to each word for the tweet
	train_data = loadTrainingData("code_switching/data/train_data.tsv", ["tweetID" , "userID", "start", "end", "token", "gold label"])
	wordVecs = buildWordVec(train_data["token"])
	if os.path.isfile("code_switching/data/trainingData.pkl"):
		token_2_label = pandas.read_pickle("code_switching/data/trainingData.pkl")
	else:
		train_tweets = loadTrainingData("code_switching/data/train_tweets.tsv", ["tweetID" , "userID", "text"])
		token_2_label = []
		for index, tweet in train_tweets.iterrows():
			print(index)
			tweet_id = tweet["tweetID"]
			id_match = train_data['tweetID'] == tweet_id
			tokens = train_data[id_match]
			tokens.sort_values('start')
			sentence_vec, label_vec = [], []
			for i, row in tokens.iterrows():
				sentence_vec.append(row['token'])
				label_vec.append(row['gold label'])
			
			token_2_label.append([tweet_id, sentence_vec, label_vec])
			
		token_2_label = pandas.DataFrame(token_2_label, columns = ['tweetID','tokenVec', 'labelVec'])
		token_2_label.to_pickle("code_switching/data/trainingData.pkl")
	return token_2_label, wordVecs

def getDataVecs(): # get file with word and data vectors, generate file if none exists
	
	if os.path.isfile("code_switching/data/trainingDataVecs.pkl"):
		trainingData = pandas.read_pickle("code_switching/data/trainingDataVecs.pkl")
	else:
		trainingData, wordVecs = prepData()
		
		for i, row in trainingData.iterrows():
			print(i)
			for j, word in enumerate(row['tokenVec']):
				row['tokenVec'][j] = wordVecs[word.lower()]
			for j, label in enumerate(row['labelVec']):
				row['labelVec'][j] = labelVecs[label]
				
		trainingData.to_pickle("code_switching/data/trainingDataVecs.pkl")
	
	return trainingData

def buildCharVec(charVec, wordSeries): # builds dictionary to map chars to a vector
	for index, word in wordSeries.iteritems():
		for char in word:
			if not char.lower() in charVec:
				charVec[char] = len(charVec)
	
	return charVec

def genCharVec(): # generates char vector map to be interpreted by tensors
	train_data = loadTrainingData("code_switching/data/train_data.tsv", ["tweetID" , "userID", "start", "end", "token", "gold label"])
	charVec = buildCharVec({}, train_data["token"])
	dev_data = loadTrainingData("code_switching/data/dev_data.tsv", ["tweetID" , "userID", "start", "end", "token", "gold label"])
	charVec = buildCharVec(charVec, dev_data["token"])
	return charVec

def words_2_vecs(words, wordVecs): # convert list of words to vectors
	ret_words = []
	for i, word in enumerate(words):
		try:
			ret_words.append(wordVecs[word.lower()])
		except KeyError: # handles previously unknown words
			ret_words.append(wordVecs[None])
	return torch.tensor(ret_words, device = device)
	
def targs_2_vecs(targs): # convert target labels to vectors
	ret_targs = []
	for i, targ in enumerate(targs):
		ret_targs.append(labelVecs[targ])
	return torch.tensor(ret_targs, device = device)
	
class LSTMTagger(torch.nn.Module): # single LSTM token tagger based on https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html

	def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, wordVecs, bidir, layers):
		super(LSTMTagger, self).__init__()
		self.hidden_dim = hidden_dim
		self.wordVecs = wordVecs
		self.word_embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

		# The LSTM takes word embeddings as inputs, and outputs hidden states with dimensionality hidden_dim.
		self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim, bidirectional = bidir, num_layers = layers)

		# The linear layer that maps from hidden state space to tag space
		self.hidden2tag = torch.nn.Linear(hidden_dim * (2 if bidir else 1), tagset_size)

	def forward(self, sentence):
		sentence = words_2_vecs(sentence, self.wordVecs)
		embeds = self.word_embeddings(sentence)
		lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
		tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
		tag_scores = torch.nn.functional.log_softmax(tag_space, dim = 1)
		return tag_scores
		
def trainModelSingleLSTM(modelName = "defaultModelSingleLSTM", learningRate = 0.01, numEpochs = 5, epochSaves = False, bidir = True, layers = 1):
	
	trainingData, wordVecs = prepData()
	trainingData = trainingData.sample(frac=1).reset_index(drop=True)
	model = LSTMTagger(embed_dim_words, hid_dim_words, len(wordVecs), len(labelVecs), wordVecs, bidir, layers).to(device)
	loss_function = torch.nn.NLLLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr = learningRate)

	for epoch in range(numEpochs):
		print(epoch)
		for index, row in trainingData.iterrows():
			#print(index)
			# Step 1. Remember that Pytorch accumulates gradients.
			# We need to clear them out before each instance
			model.zero_grad()

			# Step 2. Get our inputs ready for the network, that is, turn them into Tensors of word indices.
			#sentence_in = torch.tensor(row["tokenVec"], device = device)
			targets = targs_2_vecs(row["labelVec"])

			# Step 3. Run our forward pass.
			tag_scores = model(row["tokenVec"])

			# Step 4. Compute the loss, gradients, and update the parameters by calling optimizer.step()
			loss = loss_function(tag_scores, targets)
			loss.backward()
			optimizer.step()
		if epochSaves and numEpochs - epoch > 1:
			torch.save(model, "code_switching/models/{0}epoch{1}.pt".format(modelName, epoch + 1))
	torch.save(model, "code_switching/models/{0}.pt".format(modelName))

class LSTMTagger2Layer(torch.nn.Module): # double LSTM tagger, perform LSTM pass on word characters and pass final hidden state to be concatenated with input for LSTM on word level

	def __init__(self, embedding_dim_words, hidden_dim_words, wordVecs, tagset_size, embedding_dim_chars, hidden_dim_chars, charVecs, bidir, layers):
		super(LSTMTagger2Layer, self).__init__()
		self.charVecs = charVecs
		self.embedding_dim_words = embedding_dim_words
		self.wordVecs = wordVecs
		self.hidden_dim_chars = hidden_dim_chars
		self.bidir = bidir
		self.layers = layers

		self.word_embeddings = torch.nn.Embedding(len(wordVecs), embedding_dim_words)
		self.char_embeddings = torch.nn.Embedding(len(charVecs), embedding_dim_chars)

		# The LSTM takes word embeddings as inputs, and outputs hidden states with dimensionality hidden_dim.
		self.lstm_words = torch.nn.LSTM(embedding_dim_words + hidden_dim_chars * (2 if bidir else 1), hidden_dim_words, bidirectional = bidir, num_layers = layers)
		self.lstm_char = torch.nn.LSTM(embedding_dim_chars, hidden_dim_chars, bidirectional = bidir, num_layers = layers)

		# The linear layer that maps from hidden state space to tag space
		self.hidden2tag = torch.nn.Linear(hidden_dim_words * (2 if bidir else 1), tagset_size)

	def forward(self, sentenceWords):
		sentenceVecs = words_2_vecs(sentenceWords, self.wordVecs)
		word_embeds = self.word_embeddings(sentenceVecs)
		
		hidden_states = []
		for word in sentenceWords: # create list of character vectors, generate list of character embeddings, perform lstm operation and take final hidden state
			c_vecs = []
			for char in word:
				c_vecs.append(self.charVecs[char.lower()])
			c_embeds = self.char_embeddings(torch.tensor(c_vecs, dtype=torch.long, device = device))
			c_lstm_out, c_state = self.lstm_char(c_embeds.view(len(word), 1, -1)) # c_lstm_out: (seq_len, batch, num_directions * hidden_size)

			hidden_states.append(c_lstm_out[len(word) - 1, 0]) # take final hidden state from character lstm for each word
			
		catTensor = torch.cat((word_embeds, torch.stack(hidden_states)), 1) # concatenate word embeddings tensor with character hidden states tensor
		
		w_lstm_out, _ = self.lstm_words(catTensor.view(len(sentenceVecs), 1, -1))
		tag_space = self.hidden2tag(w_lstm_out.view(len(sentenceVecs), -1))
		tag_scores = torch.nn.functional.log_softmax(tag_space, dim=1)
		return tag_scores

def trainModelDoubleLSTM(modelName = "defaultModelDoubleLSTM", learningRate = 0.01, numEpochs = 5, epochSaves = False, bidir = True, layers = 1):
	trainingData, wordVecs = prepData()
	trainingData = trainingData.sample(frac=1).reset_index(drop=True)
	charVecs = genCharVec()
	model = LSTMTagger2Layer(embed_dim_words, hid_dim_words, wordVecs, len(labelVecs), embed_dim_chars, hid_dim_chars, charVecs, bidir, layers).to(device)
	loss_function = torch.nn.NLLLoss()
	optimizer = torch.optim.SGD(model.parameters(), lr = learningRate)
	
	for epoch in range(numEpochs):
		print(epoch)
		for index, row in trainingData.iterrows():
			#print(index)

			model.zero_grad()

			#inputs_in, sentence_in = prepare_sequence2Layer(row["tokenVec"], wordVecs)
			targets = targs_2_vecs(row["labelVec"])

			tag_scores = model(row["tokenVec"])

			loss = loss_function(tag_scores, targets)
			loss.backward()
			optimizer.step()
		if epochSaves and numEpochs - epoch > 1:
			torch.save(model, "code_switching/models/{0}epoch{1}.pt".format(modelName, epoch + 1))
		
	#torch.save(model.state_dict(), "code_switching/models/{0}.pt".format(modelName))
	torch.save(model, "code_switching/models/{0}.pt".format(modelName))
	print("end")

def loadDevData(): # load test data
	if os.path.isfile("code_switching/data/devData.pkl"):
		token_2_label = pandas.read_pickle("code_switching/data/devData.pkl")
	else:
		dev_data = loadTrainingData("code_switching/data/dev_data.tsv", ["tweetID" , "userID", "start", "end", "token", "gold label"])
		dev_tweets = loadTrainingData("code_switching/data/dev_tweets.tsv", ["tweetID" , "userID", "text"])
		token_2_label = []
		for index, tweet in dev_tweets.iterrows():
			print(index)
			tweet_id = tweet["tweetID"]
			id_match = dev_data['tweetID'] == tweet_id
			tokens = dev_data[id_match]
			tokens.sort_values('start')
			sentence_vec, label_vec = [], []
			for i, row in tokens.iterrows():
				sentence_vec.append(row['token'])
				label_vec.append(row['gold label'])
			
			token_2_label.append([tweet_id, sentence_vec, label_vec])
			
		token_2_label = pandas.DataFrame(token_2_label, columns = ['tweetID','tokenVec', 'labelVec'])
		token_2_label.to_pickle("code_switching/data/devData.pkl")
	return token_2_label
	
def testModel(modelName): # load model and test predictions against test data, track and output accuracy metrics
	
	dev_data = loadDevData()
	wordVec = genWordVec()
	model = torch.load("code_switching/models/" + modelName + ".pt")
	model.eval()
	
	successCount, trialCount = 0, 0
	
	print("Testing " + modelName + "...")
	print(model)
	
	for i, row in dev_data.iterrows():
		preds = model(row["tokenVec"])
		targs = targs_2_vecs(row["labelVec"])
		
		predMaxs = torch.max(preds, 1)[1]
		for j, ind in enumerate(predMaxs):
			if ind == targs[j]:
				successCount += 1
			trialCount += 1
		
	print("Final stats: Trials {0}, Successes {1}, Success Rate {2}".format(trialCount, successCount, successCount / trialCount))
	
def testModels(modelList): # an arguably unnecessary function
	for model in modelList:
		testModel(model)
	
#trainModelSingleLSTM(modelName = "singleLSTM_uni", bidir = False)
#testModel("code_switching/models/defaultModelSingleLSTM.pt")
trainModelDoubleLSTM(modelName = "DoubleLSTM - bidir - 2layer", bidir = True, layers = 2)
testModel("DoubleLSTM - bidir - 2layer")
#testModels(["doubleLSTMepoch1", "doubleLSTMepoch2", "doubleLSTMepoch3", "doubleLSTMepoch4", ])
print("done")