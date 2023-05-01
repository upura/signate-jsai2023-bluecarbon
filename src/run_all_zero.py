import pandas as pd

if __name__ == "__main__":

    sub = pd.read_csv("../input/jsai2023/submit_example.csv", header=None)
    sub[1] = 0
    sub.to_csv("submission_all_zero.csv", index=False, header=None)
