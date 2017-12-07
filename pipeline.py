import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# NOTE: Make sure that the class is labeled 'target' in the data file
tpot_data = pd.read_csv('cleandata.mapped.tsv', sep='\t', dtype=np.float64)
features = tpot_data.drop('class', axis=1).values
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['class'].values, random_state=42)

# Score on the training set was:0.882509387374
exported_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    RandomForestClassifier(bootstrap=True, criterion="entropy", max_features=0.25, min_samples_leaf=3, min_samples_split=5, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
testing_results = exported_pipeline.predict(testing_features)
training_results = exported_pipeline.predict(training_features)
