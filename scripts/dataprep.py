from pathlib import Path
from yellowbrick.target import FeatureCorrelation
import matplotlib.pyplot as plt
import pandas as pd

class GameReviews:
    def __init__(self, data_path):
        self.rawdf = pd.read_json(data_path, lines=True)
        self.rawdf.insert(0, 'title', Path(data_path).stem)
        # For simplicity, replace missing values with 0.
        self.rawdf = self.rawdf.fillna(0)

def main():
    arma3_revs = GameReviews('../data/Arma_3.jsonlines')
    df = arma3_revs.rawdf
    pd.set_option('display.max_columns', None)
    # print(df.head())
    # print('\nColumns containing missing values:')
    # print(df.isna().any())
    y = df['found_helpful_percentage']
    # X = df.drop('found_helpful_percentage', 1)
    features = ['total_game_hours_last_two_weeks','num_groups','num_badges','num_found_funny','num_workshop_items','num_voted_helpfulness','num_found_helpful','friend_player_level','num_found_unhelpful','total_game_hours','num_guides','num_friends','num_screenshots','num_comments','num_reviews','num_games_owned']
    X = df[features]
    visualizer = FeatureCorrelation(method='mutual_info-regression')
    visualizer.fit(X, y)
    plt.tight_layout()
    visualizer.show()



if __name__ == '__main__':
    main()