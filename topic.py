from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.parsing.preprocessing import remove_stopwords
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import numpy as np


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


def idf_list(filename):
    idf = []
    with open(filename, 'r') as file:
        for line in file:
            line_1 = line.strip('\n')
            idf.append(line_1.split(' ')[0])
    return idf[:49]


def list_inter(list_1, list_2):
    return [a for a in list_1 if a not in list_2 and len(a) > 1]


def topic_classification_gensim(filename_1, filename_2):
    idf_1 = idf_list("idf11-13.txt")
    idf_2 = idf_list("idf14-16.txt")
    common_texts = []
    with open(filename_1, 'r') as file:
        for line in file:
            line_1 = line.strip('\n')
            line_1 = remove_stopwords(line_1)
            if line_1.split(' ')[-1] == "":
                common_texts.append(list_inter(line_1.split(' ')[:-1], idf_1))
            else:
                common_texts.append(list_inter(line_1.split(' '), idf_1))
    common_dictionary = Dictionary(common_texts)
    common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
    '''
    coherence_score = []
    for i in range(20, 40):
        lda = LdaModel(common_corpus, id2word=common_dictionary, iterations=50, num_topics=i)
        coherence_model_lda = CoherenceModel(model=lda, texts=common_texts, dictionary=common_dictionary,
                                             coherence='u_mass')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)
        coherence_score.append(coherence_lda)
    plt.plot(range(20, 40, 1), coherence_score)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show() 
    # how to find the optimal number of topics.
    '''

    lda = LdaModel(common_corpus, id2word=common_dictionary, iterations=50, num_topics=35)
    for index, topic in lda.show_topics(formatted=False, num_words=20, num_topics=30):
        print('Topic: {} \nWords: {}'.format(index, [w[0] for w in topic]))
    
    other_texts = []
    with open(filename_2, 'r') as file:
        for line in file:
            line_1 = line.strip('\n')
            line_1 = remove_stopwords(line_1)
            if line_1.split(' ')[-1] == "":
                other_texts.append(list_inter(line_1.split(' ')[:-1], idf_2))
            else:
                other_texts.append(list_inter(line_1.split(' '), idf_2))
    other_corpus = [common_dictionary.doc2bow(text) for text in other_texts]
    topic_14 = [0.00 for n in range(30)]
    topic_11 = [0.00 for n in range(30)]
    for seen_doc in common_corpus:
        vector_1 = lda[seen_doc]
        for vec in vector_1:
            topic_11[vec[0]] = topic_11[vec[0]]+vec[1]
    # topic_11 = np.array(topic_11) / np.linalg.norm(topic_11)
    # print(topic_11)
    for unseen_doc in other_corpus:
        vector = lda[unseen_doc]
        for vec in vector:
            topic_14[vec[0]] = topic_14[vec[0]]+vec[1]
    # topic_14 = np.array(topic_14)/np.linalg.norm(topic_14)
    # print(topic_14)


def topic_classification(filename):
    corpus = []
    with open(filename, 'r') as file:
        for line in file:
            corpus.append(line.strip('\n'))
    vectorizer = CountVectorizer(stop_words='english', max_df=0.2, min_df=0.1)
    cntTf = vectorizer.fit_transform(corpus)
    lda = LatentDirichletAllocation(n_components=30,
                                    learning_method='batch',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(cntTf)
    tf_feature_names = vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, 6)


def main():
    # topic_classification('word90-92.txt')
    topic_classification_gensim('word11-13.txt', 'word14-16.txt')
    '''
    topic_classification('word90-92.txt')
    topic_classification('word93-95.txt')
    topic_classification('word96-98.txt')
    topic_classification('word99-01.txt')
    topic_classification('word02-04.txt')
    topic_classification('word05-07.txt')
    topic_classification('word08-10.txt')
    topic_classification('word11-13.txt')
    '''


if __name__ == '__main__':
    main()
