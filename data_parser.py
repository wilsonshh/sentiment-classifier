import re


def clean_text(raw_review):
    """
    Input:
            raw_review: raw text from the csv data set
    Output:
            Cleaned data.
    """

    # Retain only alphabets from the raw data
    text = re.sub("[^a-zA-Z]", " ", raw_review)

    # Slits words and and lowercase to ensure consistency
    text = text.lower().split()

    words = text

    return " ".join(words)
