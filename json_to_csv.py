import json
import zipfile
import pandas as pd
from pandas.io.json import json_normalize

def get_text():
    """
    extract review column from the json data frame and export as csv

    Output:
            csv file containing the review text
    """
    with zipfile.ZipFile("negative3.zip", "r") as z:
        for filename in z.namelist():
            with z.open(filename) as f:
                data = f.read()
                json_data = json.loads(data)
                review = json_data['text']
                json_list = []
                json_list.append(review)
                data_frame = pd.DataFrame(json_list)
                data_frame.to_csv('negative3.csv', mode = 'a', header=False, encoding="utf-8-sig")

def insert():
    """
    insert sentiment column where 1 represents as postive and 0 represents as negative

    Output:
            csv file containing sentiment column
    """
    df = pd.read_csv('positive3.csv', header=None)
    df.insert(0, 'sentiment', '1')
    df.to_csv('p.csv')

def merge():
    """
    merge two different csv files

    Output: combination of positive and negative samples as csv
    """
    negative = pd.read_csv('n.csv')
    positive = pd.read_csv('p.csv')
    negative_data_frame = pd.DataFrame(negative)
    positive_data_frame = pd.DataFrame(positive)
    frames = [negative_data_frame, positive_data_frame]
    results = pd.concat(frames)
    results.to_csv('l.csv')


def main():
    merge()


if __name__ == '__main__':
    main()