'''
LSH(지역 민감 해상)

스팸 메시지에 퍼지 해시를 적용해 유사한 해시 제거
'''

import os
import pickle
import email_read_util
from datasketch import MinHash, MinHashLSH

# predict
def lsh_predict_label(stems,lsh):
    '''
    LSH 매처의 반환값:
        0 스팸
        1 햄
       -1 에러
    '''
    minhash = MinHash(num_perm=128)
    if len(stems) < 2:
        return -1
    for s in stems:
        minhash.update(s.encode('utf-8'))
    matches = lsh.query(minhash)
    if matches:
        return 0
    else:
        return 1

DATA_DIR = 'datasets/trec07p/data/'
LABELS_FILE = 'datasets/trec07p/full/index'
TRAINING_SET_RATIO = 0.7

labels = {}

# 라벨링 읽기
# ham : 1
# spam : 0
with open(LABELS_FILE) as f:
    for line in f:
        line = line.strip()
        #print(line)
        label, key = line.split()
        labels[key.split('/')[-1]] = 1 if label.lower() == 'ham' else 0

# train, test 분리하기
filelist = os.listdir(DATA_DIR)
X_train = filelist[:int(len(filelist)*TRAINING_SET_RATIO)]
X_test = filelist[int(len(filelist)*TRAINING_SET_RATIO):]

print("[INFO]extracting spam...")

# spam 파일 뽑기
spam_files = [x for x in X_train if labels[x] == 0]

print("[INFO]extracting spam finish")

# 자카르드 유사도 모드?
# 임계치 0.5 순열 수 128
lsh = MinHashLSH(threshold=0.5, num_perm=128)

print("[INFO]LSH")

# LSH 매처 전달
for idx, f in enumerate(spam_files):
    minhash = MinHash(num_perm=128)
    stems = email_read_util.load(os.path.join(DATA_DIR, f))
    if len(stems) < 2: continue
    for s in stems:
        minhash.update(s.encode('utf-8'))
    lsh.insert(f, minhash)

print("[INFO]LSH finish")

fp = 0
tp = 0
fn = 0
tn = 0

print("[INFO]training...")

# train
for filename in X_test:
    path = os.path.join(DATA_DIR, filename)
    if filename in labels:
        label = labels[filename]
        stems = email_read_util.load(path)
        if not stems:
            continue
        pred = lsh_predict_label(stems,lsh)
        if pred == -1:
            continue
        elif pred == 0:
            if label == 1:
                fp = fp + 1
            else:
                tp = tp + 1
        elif pred == 1:
            if label == 1:
                tn = tn + 1
            else:
                fn = fn + 1

print("[INFO]train finish")

print("===predict===")
print("HAM is HAM : ",tn)
print("HAM is SPAM : ",fp)
print("SPAM is HAM : ",fn)
print("SPAM is SPAM : ",tp)

count = tn + tp + fn + fp

print("Classification accuracy: {}".format("{:.1%}".format((tp+tn)/count)))