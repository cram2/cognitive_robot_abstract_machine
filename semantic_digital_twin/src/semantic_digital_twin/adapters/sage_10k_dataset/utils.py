from itertools import takewhile
from typing import Optional
import enchant

english_dictionary = enchant.Dict("en_US")


def clean_type_name(type_name: str) -> Optional[str]:
    cleaned_type = "".join(takewhile(str.isalpha, type_name))

    if not cleaned_type:
        return None
    if not english_dictionary.check(cleaned_type):
        return None
    return cleaned_type
