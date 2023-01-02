from gensim.models import word2vec
import config


def generate_w2v_model():
    """
        ### generate_w2v_model()
        Generates a gensim word 2 vec model from the vocabulary contained in the tokenized version of all utterances and serialize it in the path in config.word2vec_model.

        The dimension of words is an hyperparameter contained in config.WORD_DIM
    """
    corpus = word2vec.Text8Corpus(config.preprocessed_corpus_path)
    word_vector = word2vec.Word2Vec(corpus, vector_size=config.WORD_DIM)
    word_vector.wv.save_word2vec_format(config.word2vec_model , binary=True)