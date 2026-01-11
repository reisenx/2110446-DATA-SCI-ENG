import pandas as pd
from student import *


def main():
    input_string = input().strip()
    df = pd.read_csv("./scores.csv")
    input_command = f"{input_string}(df)"
    print(f"{eval(input_command)}")


if __name__ == "__main__":
    main()
