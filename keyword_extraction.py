from keybert import KeyBERT

kw_model = KeyBERT()

def extract_keywords(text, num_keywords=10):
    keywords = kw_model.extract_keywords(text, top_n=num_keywords)
    return [word for word, score in keywords]
