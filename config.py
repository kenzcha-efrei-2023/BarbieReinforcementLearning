"""
PATHS 
"""

# data

utterance_raw_path = "data/utterances.jsonl"
only_words_corpus_path = "data/all_words.txt"
preprocessed_corpus_path = "data/tokenized_words.txt"
dictionnary_utterances = "data/utterance_dict"
reversed_conv = "data/reversed_conversations_lenmax22"

# models

word2vec_model = "model/word_vector.bin"

"""
HYPERPARAMETERS
"""

WORD_DIM = 100
LEN_MAX_CONV = 22

"""
OTHER
"""

verbose = True