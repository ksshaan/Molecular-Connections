import numpy as np
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os.path
#from pyfastext import FastText
from gensim.test.utils import common_texts
from gensim.models import FastText
#sentences_ted = [['i love ice-cream'], ['he loves ice-cream'], ['you love ice cream']]

#model_ted = FastText(sentences_ted, size=100, window=5, min_count=5, workers=4,sg=1)


def readData(file_name, max_length):
    data_path = './resources/' + file_name
    if not os.path.isfile(data_path):
        print(file_name + " not found")
        exit()
    with open(data_path, 'rb') as pckl:
        text = pickle.load(pckl)
        for o, doc in enumerate(text):
            text[o] = " ".join(text[o].split()[:max_length])
        return text


def editData(text_file, label_file, max_length, tokenizer):
    text_path = './resources/' + text_file
    if not os.path.isfile(text_path):
        print(text_file + " not found")
        exit()
    with open(text_path, 'rb') as pckl:
        texts = pickle.load(pckl)
    for o, doc in enumerate(texts):
        texts[o] = " ".join(texts[o].split()[:max_length])
    sequences = tokenizer.texts_to_sequences(texts)
    del texts
    data = pad_sequences(sequences, maxlen=max_length,
                         padding='post', truncating='post')
    del sequences
    label_path = './resources/' + label_file
    if not os.path.isfile(label_path):
        print(label_file + " not found")
        exit()
    with open(label_path, 'rb') as pckl:
        labels = pickle.load(pckl)
    data = data.astype(np.uint16)
    return data, labels


def preprocess(fasttext_name, embedding_dim, max_length, max_num_words):
    #model_ted.build_vocab(sentences, update=False)
    #model = FastText(size=4, window=3, min_count=1)  # instantiate
    #model.build_vocab(sentences=common_texts)
    #model.train(sentences=common_texts, total_examples=len(common_texts), epochs=10)  # train
    #FastText(size=4, window=3, min_count=1, sentences=common_texts, iter=10)
    #fastmodel = FastText('./resources/' + str(fasttext_name)) # original
    #print("important",embedding_dim, max_num_words)
    fastmodel = FastText(size=4, window=3, min_count=1) #instantiate
    fastmodel.build_vocab('./resources/' + str(fasttext_name))
    #fastmodel = model_ted('./resources/' + fasttext_name)
    texts_tokenize = readData('train_texts.pkl', max_length)
    print("Tokenizing data ..")
    tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(texts_tokenize)
    print("Tokenization and fitting done!")
    print("Loading data ...")
    x_train, y_train = editData('train_texts.pkl', 'train_labels.pkl',
                                max_length, tokenizer)
    print("Training data loaded")
    x_val, y_val = editData('test_texts.pkl', 'test_labels.pkl',
                            max_length, tokenizer)
    print("Test Data loaded")
    word_index = tokenizer.word_index
    print('Preparing embedding matrix ...')
    embedding_matrix = np.zeros((max_num_words, embedding_dim)) # initial
    #embedding_matrix = np.zeros((4,1 )) # changed
    
    #embedding_matrix = [0, 0, 0, 0,0,0,0]
    #print('shape of embeding matrix',embedding_matrix.shape)
    for word, i in word_index.items():
        if i >= max_num_words:
            continue
        #embedding_vector = np.zeros((max_num_words, embedding_dim)) 
        b = fastmodel[word]
        shap = b.shape[0]
        print("shape of b",shap)
        zero_pad = 300 -shap
        b=np.pad(b, (0, zero_pad), 'constant')
        print("shape of b after padding  zeors",b.shape)
        #b = b.reshape(1,23140)
        #if embedding_vector is not None: embedding_matrix[i] = embedding_vector
        #print(fastmodel[word].shape)
        #x = fastmodel[word]

        #embedding_matrix[i] =  fastmodel[word]#initial
        print("embedding matrix", embedding_matrix.shape)
        embedding_matrix[i] = b
    print("Embedding done!")
    return x_train, y_train, x_val, y_val, embedding_matrix
