import json
import re
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from stemming.porter2 import stem


def main():
    # nltk.download('wordnet')
    result = []
    output = []
    stemmer = SnowballStemmer("english")
    wnl = WordNetLemmatizer()
    with open('conference_list_extract.txt', 'r') as f:
        for line in f:
            line = json.loads(line)
            if line['year'] == 2014 or line["year"] == 2015 or line["year"] == 2016:
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

    with open('word14-16.txt', 'w') as f:
        for out in output:
            if out[0] == "without" and out[1] == "abstract":
                continue
            for tmp in out:
                if tmp:
                    f.write(tmp+' ')
            f.write('\n')


if __name__ == '__main__':
    main()
