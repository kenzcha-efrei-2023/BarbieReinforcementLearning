from utils import printV
import config
import numpy as np
import re
import codecs
import json
import pickle

def build_vocab(word_count_threshold : int=5):
    """
    ### build_vocab(word_count_threshold)
    Generates vocabulary (index to word and word to index) from a corpus text contained in the path indicated in the corpus_path variable of config.py in addition of an initial bias vector
    
    Args:

        word_count_threshold (int, optional): Threshold that indicates the minimum of times a word must appear in the corpus. Defaults to 5.
    
    Returns :

        word_indx : dict, a word to index map containing all vocab words as keys and indexes as value
        indx_word : dict, a index to word map containing all indexes as keys and vocab words as value
        bias_vect : numpy array, bias vector containg shifted log of normalized number of apparition
    """
    printV("Method : Build vocabulary")
    printV("Downloading all the sentences ...")
    #Open the file containing all words and remove all punctuation
    corpus_text = open(config.preprocessed_corpus_path).read()
    #put the result in a numpy array
    corpus = np.asarray(corpus_text.split("\n"))
    del corpus_text
    printV("Done !")
    #counting all the words
    count_w = {}
    for sentence in corpus :
        sent = sentence.lower().split(" ")
        for word in sent :
            count_w[word] = count_w.get(word, 0) + 1
    len_corp = len(corpus)
    #adding pad, bos, eos and unk to vocab
    count_w ={**{"<pad>":len_corp, "<bos>":len_corp, "<eos>":len_corp, "<unk>":len_corp},**count_w}
    #removing the empty word
    if '' in count_w:
        del count_w['']
    del len_corp
    printV("Building vocabulary + bias vector")
    vocabulary = np.array([word for word in count_w.keys() if count_w[word] > word_count_threshold])
    #creating word to index + index to word dictionnary
    indx_word = {}
    word_indx = {}
    for id, w in np.ndenumerate(vocabulary[:20]):
        indx_word[id[0]] = w
        word_indx[w] = id[0]
    # bias vector = shifted log of normalized number of apparition
    bias_vect = np.array([float(count_w[indx_word[i]]) for i in indx_word])
    bias_vect /= np.sum(bias_vect)
    bias_vect = np.log(bias_vect)
    bias_vect -= np.max(bias_vect)
    print(bias_vect[:20])
    printV("Done !")
 
    return word_indx, indx_word, bias_vect

def preprocess(phrase : str):
    """
    #### preprocess(phrase)

    preprocesses a string by removing all char that are not alphebetical, unique ', - or whitesepaces 

    Args:

        phrase (string): the string that needs basic preprocessing

    Returns:

        string: the phrase that was preprocessed
    """
    #separating into words
    words = re.findall(r"[a-zA-Z\'\-]+", phrase)
    #removing multiple '
    words = ["".join(w.split("'")) for w in words]
    return (" ".join(words)).lower()

def preprocess_corpus():
    """
    ### generate_tokenized()

    Writes the preprocessed version of words utterances in the file which path is specified in config.only_words_corpus_path in config.preprocessed_corpus_path
    """
    printV("Downloading all the sentences ...")
    sentences = open(config.only_words_corpus_path).read().split('\n')
    printV("Removing all the non-alphebitical symbols except ' and -")
    sentences = [preprocess(i) for i in sentences]
    #writing the resulting phrases in file
    with codecs.open(config.preprocessed_corpus_path, "w", encoding='utf-8', errors='ignore') as f:
        for sent in sentences :
            f.write(sent + "\n")

def from_corpus_to_only_words():
    """
    ### from_corpus_to_only_words()

    Writes all the text contained in the raw utterances jsonl in a file specified in config.only_words_corpus_path
    """
    printV("Extracting the text of the utterances")
    with open(config.utterance_raw_path, 'r') as json_file:
        with codecs.open(config.only_words_corpus_path, "w", encoding='utf-8', errors='ignore') as f:
            for js in list(json_file) :
                utt = json.loads(js)
                f.write(utt["text"].lower() + "\n")
    printV("Done !")
            

def from_json_to_dictionnary():
    """
    ### from_corpus_to_only_words()
    Serialize the id + utterance as a dictionnary in a file which path is specified in config.dictionnary_utterances
    """
    printV("Extracting the text and id of the utterances ...")
    dictionnary = {}
    with open(config.utterance_raw_path, 'r') as json_file:
        for js in list(json_file) :
            utt = json.loads(js)
            dictionnary[utt["id"]] = utt["text"]
    printV("Serializing the dictionnary as an object")
    pickle.dump(dictionnary, open(config.dictionnary_utterances, 'wb'), True)
    printV("Done !")


#script to be executed when file is called as main

if __name__ == '__main__':
    #extract text from the raw jsonl file 
    from_corpus_to_only_words()
    #preprocess raw text
    preprocess_corpus()
    #create dictionnary of utterances
    from_json_to_dictionnary()