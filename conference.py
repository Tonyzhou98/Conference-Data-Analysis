import ast


def takeF(elem):
    return elem[3]


def main(list_file_name):
    fileObject = open('conference_list_2.txt', 'w', encoding='utf-8')
    fileObject.write("rank\tyear\tconference\ttotal_number\n")
    conferences = []
    results = []

    with open("conference_list_extract.txt", "r") as datum:
        for data in datum:
            try:
                data_dict = ast.literal_eval(data)
            except ValueError:
                continue
            try:
                year = data_dict["year"]
                conference_name = data_dict["venue"]
                if year < 1990:
                    continue
                list_1 = (year, conference_name)
                conferences.append(list_1)
            except KeyError:
                continue
    i = 0
    for confer in set(conferences):
        confer_1 = [i, confer[0], confer[1], conferences.count(confer)]
        results.append(confer_1)
    results.sort(key=takeF, reverse=True)
    results = results[:200]
    for confer in results:
        i = i + 1
        confer[0] = i
        for re in confer:
            fileObject.write(str(re) + "\t")
        fileObject.write('\n')
    print(results)
    fileObject.close()


if __name__ == '__main__':
    main("aminer_papers_0/aminer_papers_")
