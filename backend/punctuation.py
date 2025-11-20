# from deepmultilingualpunctuation import PunctuationModel

# punct_model = PunctuationModel()


# def restore_punctuation(text: str) -> str:
#     """
#     Restores punctuation and capitalization on raw ASR text.
#     """
#     if not text.strip():
#         return ""

#     restored = punct_model.restore_punctuation(text)
#     return restored

# ============================================
# MOCKED VERSION FOR UI TESTING
# ============================================
# Original implementation commented out below

def restore_punctuation(text: str) -> str:
    """
    MOCKED: Simply returns the input text with basic capitalization.
    Use this version when you don't want to load the actual model.
    """
    if not text.strip():
        return ""
    
    # Simple mock: capitalize first letter and add period at end
    text = text.strip()
    if text:
        text = text[0].upper() + text[1:]
        if not text.endswith(('.', '!', '?')):
            text += '.'
    
    return text


# ============================================
# ORIGINAL IMPLEMENTATION (COMMENTED OUT)
# ============================================
# Uncomment below when you want to use actual punctuation model

"""
from deepmultilingualpunctuation import PunctuationModel

punct_model = PunctuationModel()


def restore_punctuation(text: str) -> str:
    '''
    Restores punctuation and capitalization on raw ASR text.
    '''
    if not text.strip():
        return ""

    restored = punct_model.restore_punctuation(text)
    return restored
"""