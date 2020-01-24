from pathlib import Path
from yellowbrick.features import Rank2D
from yellowbrick.target import FeatureCorrelation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def word_count(rev_str):
    return len(rev_str.split())

def split_achievements(ach_str):
    ach_perc = ach_str['num_achievements_percentage']
    ach_att = ach_str['num_achievements_attained']
    ach_poss = ach_str['num_achievements_possible']
    return ach_perc, ach_att, ach_poss

class GameReviews:
    def __init__(self, data_path):
        self.rawdf = pd.read_json(data_path, lines=True)
        self.rawdf.insert(0, 'title', Path(data_path).stem)
        # Adding some more fields to the raw table
        self.rawdf['review_word_count'] = self.rawdf['review'].map(word_count)
        self.rawdf['num_achievements_percentage'], self.rawdf['num_achievements_attained'], self.rawdf['num_achievements_possible'] = zip(*self.rawdf['achievement_progress'].map(split_achievements))
        # For simplicity, replace missing values with 0.
        self.rawdf = self.rawdf.fillna(0)

    def numeric_features(self):
        return self.rawdf.select_dtypes(include=np.number).columns.tolist()
    
    def mutual_info(self):
        y = self.rawdf['found_helpful_percentage'].copy()
        features = self.numeric_features()
        features.remove('found_helpful_percentage')
        X = self.rawdf[features].copy()
        visualizer = FeatureCorrelation(method='mutual_info-regression')
        visualizer.fit(X, y)
        plt.subplots_adjust(left=0.3)
        return visualizer
    
    def corr_coeff(self):
        y = self.rawdf['found_helpful_percentage'].copy()
        features = self.numeric_features()
        features.remove('found_helpful_percentage')
        X = self.rawdf[features].copy()
        visualizer = Rank2D(algorithm='pearson')
        visualizer.fit(X, y)
        visualizer.transform(X)
        plt.subplots_adjust(left=0.25, bottom=0.45)
        return visualizer
    
    def select_features(self, corr_thr = 0.5, mi_thr = 0.05):
        plt.figure()
        vis_mi = self.mutual_info()
        plt.figure()
        vis_cc = self.corr_coeff()
        selected_features = vis_cc.features_.tolist().copy()
        hc_idx = np.argwhere(vis_cc.ranks_ > 0.5)
        hc_idx = hc_idx[hc_idx[:,0] < hc_idx[:,1]]

        for row in hc_idx:
            feature1 = vis_cc.features_[row[0]]
            feature2 = vis_cc.features_[row[1]]
            idx1 = vis_mi.features_.tolist().index(feature1)
            idx2 = vis_mi.features_.tolist().index(feature2)
            remove_feature = feature1
            if vis_mi.scores_[idx2] < vis_mi.scores_[idx1]:
                remove_feature = feature2
            if remove_feature in selected_features:
                selected_features.remove(remove_feature)
        # Remove all features with helpful as substring
        for f in selected_features:
            if 'helpful' in f:
                selected_features.remove(f)
        print(selected_features)


def visualize_mutual_info(df):
    y = df['found_helpful_percentage'].copy()
    features = ['total_game_hours_last_two_weeks','num_groups','num_badges','num_found_funny','num_workshop_items','num_voted_helpfulness','num_found_helpful','friend_player_level','num_found_unhelpful','total_game_hours','num_guides','num_friends','num_screenshots','num_comments','num_reviews','num_games_owned','review_word_count','num_achievements_percentage','num_achievements_attained','num_achievements_possible']
    X = df[features].copy()
    # Mutual information visualizer
    vis_mi = FeatureCorrelation(method='mutual_info-regression')
    vis_mi.fit(X, y)
    # plt.tight_layout() # This line obscures captions.
    plt.subplots_adjust(left=0.3)
    vis_mi.show()


def visualize_corr_coeff(df):
    y = df['found_helpful_percentage'].copy()
    features = ['total_game_hours_last_two_weeks','num_groups','num_badges','num_found_funny','num_workshop_items','num_voted_helpfulness','num_found_helpful','friend_player_level','num_found_unhelpful','total_game_hours','num_guides','num_friends','num_screenshots','num_comments','num_reviews','num_games_owned','review_word_count','num_achievements_percentage','num_achievements_attained','num_achievements_possible']
    X = df[features].copy()
    visualizer = Rank2D(algorithm='pearson')
    visualizer.fit(X, y)
    visualizer.transform(X)
    plt.subplots_adjust(left=0.25, bottom=0.45)
    # print(visualizer.ranks_)
    indices = np.argwhere(visualizer.ranks_ > 0.5)
    print(indices[indices[:,0] < indices[:,1]])
    visualizer.show()


def main():
    arma3_revs = GameReviews('../data/Arma_3.jsonlines')
    df = arma3_revs.rawdf
    pd.set_option('display.max_columns', None)
    print(df.head())
    print('\nColumns containing missing values:')
    print(df.isna().any())
    # viz_corr_coeff = arma3_revs.corr_coeff()
    # viz_corr_coeff.show()
    # viz_mutual_info = arma3_revs.mutual_info()
    # print(viz_mutual_info.features_)
    # print(viz_mutual_info.scores_)
    # viz_mutual_info.show()
    arma3_revs.select_features()



if __name__ == '__main__':
    main()