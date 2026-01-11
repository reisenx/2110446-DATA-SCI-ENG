import pandas as pd
from student import *


def main():
    """
    ASSIGNMENT 1:
    Using pandas to explore youtube trending data from GB (GBvideos.csv and GB_category_id.json) and answer the questions.
    """

    input_string = input().strip()
    if input_string != "Q1":
        vdo_df = pd.read_csv("./videos.csv")
        vdo_df.drop_duplicates(inplace=True)
        input_command = f"{input_string}(vdo_df)"
    else:
        input_command = f"{input_string}()"
    print(f"{eval(input_command)}")


if __name__ == "__main__":
    main()
