from deepmultilingualpunctuation import PunctuationModel

punct_model = PunctuationModel()


def restore_punctuation(text: str) -> str:
    """
    Restores punctuation and capitalization on raw ASR text.
    """
    if not text.strip():
        return ""

    restored = punct_model.restore_punctuation(text)
    return restored