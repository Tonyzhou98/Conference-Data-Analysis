import json
import re
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from stemming.porter2 import stem

CONFERENCE_TYPE = {
    "Artificial_intelligence": ["AAAI", "Association for the Advancement of Artificial Intelligence", "IJCAI",
                                "International Joint Conference on Artificial Intelligence"],
    "Computer_vision": ["CVPR", "Computer Vision and Pattern Recognition", "ICCV",
                        "International Conference on Computer Vision",
                        "IEEE International Conference on Computer Vision", "ECCV",
                        "European Conference on Computer Vision"],
    "Machine_learning": ["ICML", "International Conference on Machine Learning", "KDD",
                         "Knowledge Discovery and Data Mining", "NIPS",
                         "Neural Information Processing Systems"],
    "Natural_language_processing": ["ACL", "Association for Computational Linguistics",
                                    "Meeting of the Association for Computational Linguistics", "EMNLP",
                                    "Empirical Methods in Natural Language Processing", "NAACL",
                                    "North American Chapter of the Association for Computational Linguistics"],
    "Information_retrieval": ["SIGIR", "Special Interest Group on Information Retrieval", "WWW",
                              "International World Wide Web Conference"]
}


def conference_classification():
    with open('conference_list_extract.txt', 'r') as f:
        for line in f:
            line = json.loads(line)
            for key, value in CONFERENCE_TYPE.items():
                if line['venue'] in value:
                    file_name = "conference/" + key + ".txt"
                    fw = open(file_name, 'r+', encoding="utf-8")
                    fw.read()
                    fw.write(str(line)+"\n")
                    fw.close()


def abstract(read_filename, write_filename, year):
    result = []
    output = []
    stemmer = SnowballStemmer("english")
    wnl = WordNetLemmatizer()
    with open(read_filename, 'r') as f:
        for line in f:
            line = json.loads(line)
            if line['year'] == year + 2000 or line["year"] == year + 2001 or line["year"] == year + 2002:
                result.append(line)
                try:
                    tmp = line["abstract"]
                except:
                    continue
                tmp = re.sub("- ", "", tmp)
                tmp = re.sub("[^a-zA-Z]", " ", tmp)
                tmp = re.sub(" +", " ", tmp).lower()
                tmp = [wnl.lemmatize(word) for word in tmp.split(" ")]
                output.append(tmp)

    with open(write_filename, 'w') as f:
        for out in output:
            if out[0] == "without" and out[1] == "abstract":
                continue
            for tmp in out:
                if tmp:
                    f.write(tmp + ' ')
            f.write('\n')


def main():
    conference_classification()
    # abstract('conference_list_extract.txt', 'word/word14-16.txt', 14)


if __name__ == '__main__':
    main()
