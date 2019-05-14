from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import TSNE
from gensim.corpora.dictionary import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.parsing.preprocessing import remove_stopwords
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import numpy as np

AREA = ["Artificial_intelligence", "Computer_vision", "Machine_learning",
        "Natural_language_processing", "Information_retrieval"]


def print_top_words(model, feature_names, n_top_words):
    """print top words for sk-learn methods"""
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()


def idf_list(filename, n):
    """construct a list with top n idf words"""
    idf = []
    with open(filename, 'r') as file:
        for line in file:
            line_1 = line.strip('\n')
            idf.append(line_1.split(' ')[0])
    return idf[:n-1]


def list_inter(list_1, list_2):
    """find the elements in lise_1 not in list_2"""
    return [a for a in list_1 if a not in list_2 and len(a) > 1]


def process_doc(filename_1, top_idf_number):
    """preprocess document"""
    file_1 = filename_1.split("/")
    if len(file_1) == 1:
        idf_file_1 = "idf/idf"+file_1[0]
        filename_1 = "word/word" + file_1[0]
    else:
        idf_file_1 = "idf/" + file_1[0]+"/idf" + file_1[1]
        filename_1 = "word/" + file_1[0]+"/word" + file_1[1]
    idf_1 = idf_list(idf_file_1, top_idf_number)
    common_texts = []
    with open(filename_1, 'r') as file:
        for line in file:
            line_1 = line.strip('\n')
            line_1 = remove_stopwords(line_1)
            if line_1.split(' ')[-1] == "":
                common_texts.append(list_inter(line_1.split(' ')[:-1], idf_1))
            else:
                common_texts.append(list_inter(line_1.split(' '), idf_1))
    return common_texts


def optimal_topic_number(filename_1, top_idf_number):
    """how to find the optimal number of topics using coherence score."""
    common_texts = process_doc(filename_1, top_idf_number)
    common_dictionary = Dictionary(common_texts)
    common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
    coherence_score = []
    for i in range(20, 40):
        lda = LdaModel(common_corpus, id2word=common_dictionary, iterations=50, num_topics=i,
                       random_state=np.random.RandomState(23455))
        coherence_model_lda = CoherenceModel(model=lda, texts=common_texts, dictionary=common_dictionary,
                                             coherence='u_mass')
        coherence_lda = coherence_model_lda.get_coherence()
        print('\nCoherence Score: ', coherence_lda)
        coherence_score.append(coherence_lda)
    plt.plot(range(20, 40, 1), coherence_score)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend("coherence_values", loc='best')
    plt.show()


def topic_classification_gensim_train(filename_1, topic_number, top_idf_number):
    """use gensim to perform lda algorithm"""
    common_texts = process_doc(filename_1, top_idf_number)
    common_dictionary = Dictionary(common_texts)
    common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
    lda = LdaModel(common_corpus, id2word=common_dictionary, iterations=50, num_topics=topic_number,
                   random_state=np.random.RandomState(23455))
    for index, topic in lda.show_topics(formatted=False, num_words=20, num_topics=topic_number):
        print('Topic: {} \nWords: {}'.format(index, [w[0] for w in topic]))
    # print the topic and words
    topic_2 = [0.00 for n in range(topic_number)]
    for seen_doc in common_corpus:
        vector_1 = lda[seen_doc]
        for vec in vector_1:
            topic_2[vec[0]] = topic_2[vec[0]]+vec[1]
        # find the distribution of each topic.
    topic_2 = np.array(topic_2) / np.linalg.norm(topic_2)
    print(filename_1+" word distribution:")
    print(topic_2)
    return topic_2, lda, common_dictionary


def topic_classification_gensim_fit(filename_2, topic_number, top_idf_number, lda_model, common_dictionary):
    topic_1 = [0.00 for n in range(topic_number)]
    common_texts = process_doc(filename_2, top_idf_number)
    common_corpus = [common_dictionary.doc2bow(text) for text in common_texts]
    Y = []
    for unseen_doc in common_corpus:
        vector = lda_model[unseen_doc]
        y = np.zeros(35)
        for vec in vector:
            topic_1[vec[0]] = topic_1[vec[0]]+vec[1]
            y[vec[0]] = vec[1]
        Y.append(y)
    Y = np.array(Y)
    tsne = TSNE(n_components=2)
    tsne.fit(Y)
    #print(tsne.embedding_)
    plt.plot(tsne.embedding_[:,0],tsne.embedding_[:,1])
    plt.show()
    topic_1 = np.array(topic_1)/np.linalg.norm(topic_1)
    print(filename_2 + " word distribution:")
    print(topic_1)
    return topic_1


def topic_classification(filename):
    """Use sk-learn to perform lda algorithm"""
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


def plot_scatter(topic_1, topic_2, topic_number):
    """plot the two topic distributions in scatter"""
    ax = plt.subplot()
    ax.scatter(range(0, topic_number), topic_1, c='red', alpha=0.6)
    ax.scatter(range(0, topic_number), topic_2, c='green', alpha=0.6)
    plt.show()


def plot_trend(matrix, area, year):
    apparent_change = []
    var = []
    for i in range(len(matrix[0])):
        topic_dis = []
        for vector in matrix:
            topic_dis.append(vector[i])
        topic_dis = np.array(topic_dis)
        var.append(np.var(topic_dis))
    print(var)
    for index in range(len(var)):
        if var[index] > 0.0005:
            apparent_change.append(index)
    for index in apparent_change:
        topic_value = []
        for vector in matrix:
            topic_value.append(vector[index])
        plt.plot(topic_value, label="topic_"+str(index))
        plt.title("topic "+str(index))
        plt.ylabel('topic distribution')
        plt.savefig("img/" + area + "/topic_" + str(index) + "_" + year + ".png")
        plt.show()


def similarity(matrix):
    """calculate the cosine similarity of two vectors"""
    sim = []
    for index in range(len(matrix)-1):
        vector1 = np.array(matrix[index])
        vector2 = np.array(matrix[index+1])
        dis = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * (np.linalg.norm(vector2)))
        sim.append(dis)
    print(sim)


def main():
    # topic_classification('word90-92.txt')
    # optimal_topic_number('11-13.txt', 50)
    topic_matrix = []
    topic_2, lda, dictionary = topic_classification_gensim_train('.txt', 35, 50)
    topic_matrix.append(topic_2)
    file = [AREA[4]+'/02-04.txt', AREA[4]+'/05-07.txt',
            AREA[4]+'/08-10.txt',
            AREA[4]+'/11-13.txt', AREA[4]+'/14-16.txt']
    for i in file:
        topic_matrix.append(topic_classification_gensim_fit(i, 35, 50, lda, dictionary))
    plot_trend(topic_matrix[1:-1], AREA[4], "02-13")
    # plot_scatter(topic_matrix[1], topic_matrix[2], 35)


if __name__ == '__main__':
    main()
