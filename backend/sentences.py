import nltk

def split_into_sentences(text: str):
    """
    Converts punctuation-restored text into proper sentences.
    """
    sentences = nltk.sent_tokenize(text)
    return [s.strip() for s in sentences if s.strip()]