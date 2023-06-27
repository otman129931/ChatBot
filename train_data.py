import json 
from Helper import NLP_helper
nlp_h=NLP_helper()
with open("data.json",'r') as f:
    intents= json.load(f)
tags=[]
pattern_words=[]
X_Y=[]
for intenet in intents['intents']:
    tag=intenet['tag']
    tags.append(tag)
    for pattern in intenet['patterns']:
        words=nlp_h.tokenezation_remove_stop_word(pattern)
        pattern_words.extend(words)
        X_Y.append((tag, words))
tags=sorted(set(tags))
pattern_words=[nlp_h.word_stema(word) for word in pattern_words ]
pattern_words=sorted(set(pattern_words))
print(pattern_words)