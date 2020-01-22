from pathlib import Path
import pandas as pd

class GameReviews:
    def __init__(self, data_path):
        self.df = pd.read_json(data_path, lines=True)
        self.df.insert(0, 'title', Path(data_path).stem)
        # For simplicity, replace missing values with 0.
        self.df = self.df.fillna(0)

def main():
    arma3_revs = GameReviews('../data/Arma_3.jsonlines')
    pd.set_option('display.max_columns', None)
    print(arma3_revs.df.head())
    print('\nColumns containing missing values:')
    print(arma3_revs.df.isna().any())



if __name__ == '__main__':
    main()