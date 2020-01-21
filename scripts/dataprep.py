import os
import pandas as pd

class GameReviews:
    def __init__(self, data_path):
        self.df = pd.read_json(data_path, lines=True)
        self.df.insert(0, 'title', os.path.splitext(os.path.basename(data_path))[0])

def main():
    arma3_revs = GameReviews('../data/Arma_3.jsonlines')
    pd.set_option('display.max_columns', None)
    print(arma3_revs.df.head())
    print(arma3_revs.df.isna().any())



if __name__ == '__main__':
    main()