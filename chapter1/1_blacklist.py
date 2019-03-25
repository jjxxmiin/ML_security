import os
import pickle
import email_read_util
from nltk.corpus import words

# 데이터
DATA_DIR = 'datasets/trec07p/data/'
LABELS_FILE = 'datasets/trec07p/full/index'
TRAINING_SET_RATIO = 0.7

labels = {}
spam_words = set()
ham_words = set()

# 라벨링 읽기
with open(LABELS_FILE) as f:
    for line in f:
        line = line.strip()
        label, key = line.split()
        labels[key.split('/')[-1]] = 1 if label.lower() == 'ham' else 0

# train, test 분리하기
filelist = os.listdir(DATA_DIR)
X_train = filelist[:int(len(filelist)*TRAINING_SET_RATIO)]
X_test = filelist[int(len(filelist)*TRAINING_SET_RATIO):]

# pickle 파일 읽기
if not os.path.exists('blacklist.pkl'):
    for filename in X_train:
        path = os.path.join(DATA_DIR, filename)
        if filename in labels:
            label = labels[filename]
            stems = email_read_util.load(path)
            if not stems:
                continue
            if label == 1:
                ham_words.update(stems)
            elif label == 0:
                spam_words.update(stems)
            else:
                continue
    blacklist = spam_words - ham_words
    pickle.dump(blacklist, open('blacklist.pkl', 'wb'))
else:
    blacklist = pickle.load(open('blacklist.pkl', 'rb') )

print('Blacklist of {} tokens successfully built/loaded'.format(len(blacklist)))

word_set = set(words.words())
word_set.intersection(blacklist)

fp = 0
tp = 0
fn = 0
tn = 0

# train
for filename in X_test:
    path = os.path.join(DATA_DIR, filename)
    if filename in labels:
        label = labels[filename]
        stems = email_read_util.load(path)
        if not stems:
            continue
        stems_set = set(stems)
        if stems_set & blacklist:
            if label == 1:
                fp = fp + 1
            else:
                tp = tp + 1
        else:
            if label == 1:
                tn = tn + 1
            else:
                fn = fn + 1

print("HAM is HAM : ",tn)
print("HAM is SPAM : ",fp)
print("SPAM is HAM : ",fn)
print("SPAM is SPAM : ",tp)

count = tn + tp + fn + fp

print("Classification accuracy: {}".format("{:.1%}".format((tp+tn)/count)))