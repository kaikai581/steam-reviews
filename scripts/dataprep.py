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
    """
    Class for importing data from json files,
    augment data with some preprocessing,
    identify numerical fields, and select features
    according to correlation between features
    and mutual information between a feature and the target value.
    """
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
        """
        This method first identify colinear feature pairs and remove the one
        in each pair that has a lower mutual information score.
        Then it removes features with low mutual information scores.
        """
        vis_mi = self.mutual_info()
        vis_cc = self.corr_coeff()
        selected_features = vis_cc.features_.tolist().copy()
        hc_idx = np.argwhere(vis_cc.ranks_ > corr_thr)
        hc_idx = hc_idx[hc_idx[:, 0] < hc_idx[:, 1]]

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
        # Remove features with too small mutual information
        for f in selected_features:
            if vis_mi.scores_[vis_mi.features_.tolist().index(f)] < mi_thr:
                selected_features.remove(f)
        print(selected_features)


def main():
    arma3_revs = GameReviews('../data/Arma_3.jsonlines')
    df = arma3_revs.rawdf
    pd.set_option('display.max_columns', None)
    print(df.head())
    print('\nColumns containing missing values:')
    print(df.isna().any())
    viz_corr_coeff = arma3_revs.corr_coeff()
    viz_corr_coeff.show()
    viz_mutual_info = arma3_revs.mutual_info()
    viz_mutual_info.show()
    arma3_revs.select_features()


if __name__ == '__main__':
    main()
