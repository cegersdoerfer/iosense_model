import sys
import json
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels

def load_datasets(file1, file2):
    with open(file1, 'r') as f:
        data1 = json.load(f)
    with open(file2, 'r') as f:
        data2 = json.load(f)
    return data1, data2

def extract_features_per_ost_mdt(data):
    features_list = []

    for sample in data:
        # Process OST features
        ost_features = {}

        # Combine 'trace_features' and 'stats_features' for OSTs
        for feature_type in ['trace_features', 'stats_features']:
            for key, value in sample[feature_type]['ost'].items():
                # Key format: 'ost_N_feature_name'
                parts = key.split('_')
                # Find 'ost_N'
                ost_id = '_'.join(parts[:2])  # 'ost_N'
                feature_name = '_'.join(parts[2:])  # The rest is the feature name

                if ost_id not in ost_features:
                    ost_features[ost_id] = {}
                ost_features[ost_id][feature_name] = value

        # Append each OST's features as a separate sample
        for ost_id, features in ost_features.items():
            features['device_id'] = ost_id  # Optionally include the OST ID
            features_list.append(features)

        # Process MDT features
        mdt_features = {}

        # Combine 'trace_features' and 'stats_features' for MDTs
        for feature_type in ['trace_features', 'stats_features']:
            for key, value in sample[feature_type]['mdt'].items():
                # Key format: 'mdt_N_feature_name'
                parts = key.split('_')
                # Find 'mdt_N'
                mdt_id = '_'.join(parts[:2])  # 'mdt_N'
                feature_name = '_'.join(parts[2:])  # The rest is the feature name

                if mdt_id not in mdt_features:
                    mdt_features[mdt_id] = {}
                mdt_features[mdt_id][feature_name] = value

        # Append each MDT's features as a separate sample
        for mdt_id, features in mdt_features.items():
            features['device_id'] = mdt_id  # Optionally include the MDT ID
            features_list.append(features)

    return features_list

def flatten_features(features_list):
    # Get all feature keys except 'device_id'
    all_keys = set()
    for features in features_list:
        all_keys.update(k for k in features.keys() if k != 'device_id')
    all_keys = sorted(list(all_keys))

    # Flatten features into vectors
    feature_vectors = []
    for features in features_list:
        vector = [features.get(key, 0.0) for key in all_keys]
        feature_vectors.append(vector)
    return np.array(feature_vectors)

def compute_mmd(X, Y, kernel='rbf', **kwargs):
    XX = pairwise_kernels(X, X, metric=kernel, **kwargs)
    YY = pairwise_kernels(Y, Y, metric=kernel, **kwargs)
    XY = pairwise_kernels(X, Y, metric=kernel, **kwargs)
    mmd = XX.mean() + YY.mean() - 2 * XY.mean()
    return mmd

def main():
    if len(sys.argv) != 3:
        print("Usage: python compare_datasets.py dataset1.json dataset2.json")
        sys.exit(1)
    file1 = sys.argv[1]
    file2 = sys.argv[2]

    data1, data2 = load_datasets(file1, file2)

    features1 = extract_features_per_ost_mdt(data1)
    features2 = extract_features_per_ost_mdt(data2)

    X = flatten_features(features1)
    Y = flatten_features(features2)

    # Ensure both datasets have the same feature dimensions
    if X.shape[1] != Y.shape[1]:
        print("Datasets have different feature dimensions.")
        sys.exit(1)

    mmd_value = compute_mmd(X, Y, kernel='rbf', gamma=1.0 / X.shape[1])

    print(f"The Maximum Mean Discrepancy (MMD) between the two datasets is: {mmd_value}")

if __name__ == "__main__":
    main()