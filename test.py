from sklearn.ensemble import RandomForestClassifier
import numpy as np
import math
import re

domainlist = []

class Domain:
    def __init__(self, _name, _label, _length, _entropy, _number, _segment):
        self.name = _name
        self.label = _label
        self.length = _length
        self.entropy = _entropy
        self.number = _number
        self.segment = _segment

    def return_data(self):
        return [self.length, self.entropy, self.number, self.segment]

    def return_label(self):
        if self.label == "dga":
            return 1
        else:
            return 0

def cal_entropy(string):
    # probability of chars in string
    prob = [float(string.count(c)) / len(string) for c in dict.fromkeys(list(string))]
    entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
    return entropy

def init_data(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            tokens = line.split(",")
            name = tokens[0]
            label = tokens[1]
            length = len(line)
            entropy = cal_entropy(line)
            number = len(re.findall(r'\d', line))
            segment = len(tokens[0].split("."))
            domainlist.append(Domain(name, label, length, entropy, number, segment))

def train(clf):
    init_data("train.txt")
    featureMatrix = []
    labelList = []
    for item in domainlist:
        featureMatrix.append(item.return_data())
        labelList.append(item.return_label())

    clf.fit(featureMatrix, labelList)

def predict(filename, clf):
    with open("result.txt", 'w') as f_w:
        with open(filename) as f_r:
            for line in f_r:
                line = line.strip()
                if line.startswith("#") or line == "":
                    continue
                length = len(line)
                entropy = cal_entropy(line)
                number = len(re.findall(r'\d', line))
                segment = len(line.split("."))
                str=line
                if clf.predict([[length, entropy, number, segment]])==0:
                    str+=",notdga\n"
                else:
                    str+=",dga\n"
                f_w.write(str)

def main():
    clf = RandomForestClassifier(random_state=0)
    train(clf)
    predict("test.txt", clf)

if __name__ == '__main__':
    main()
