import string
import email
import nltk

punctuations = list(string.punctuation)
stopwords = set(nltk.corpus.stopwords.words('english'))
stemmer = nltk.PorterStemmer()

# 이메일 부분을 문자열로 결합

def flatten_to_string(parts):
    ret = []
    if type(parts) == str:
        ret.append(parts)
    elif type(parts) == list:
        for part in parts:
            ret += flatten_to_string(part)
    elif parts.get_content_type == 'text/plain':
        ret += parts.get_payload()
    return ret

# 이메일 제목,내용 추출

def extract_email_text(path):
    # 입력 파일로부터 하나의 이메일을 불러온다.
    with open(path, errors='ignore') as f:
        msg = email.message_from_file(f)
    if not msg:
        return ""

    # 제목
    subject = msg['Subject']
    if not subject:
        subject = ""

    # 내용
    body = ' '.join(m for m in flatten_to_string(msg.get_payload()) if type(m) == str)
    if not body:
        body = ""

    return subject + ' ' + body

# 이메일 형태소 분석

def load(path):
    email_text = extract_email_text(path)
    if not email_text:
        return []

    # 메시지 토큰화
    tokens = nltk.word_tokenize(email_text)

    # 토큰에서 마침표를 제거
    tokens = [i.strip("".join(punctuations)) for i in tokens if i not in punctuations]

    # 자주 사용하지 않는 단어 제거
    if len(tokens) > 2:
        return [stemmer.stem(w) for w in tokens if w not in stopwords]
    return []


