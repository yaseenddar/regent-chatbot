import unicodedata

class Normalizer:

    def __init__(self):
        pass

    def normalize_text(self,text:str) -> str:

        # Unicode normalization (e.g., full-width -> half-wodth)
        text = unicodedata.normalize("NFKC",text)

        return text