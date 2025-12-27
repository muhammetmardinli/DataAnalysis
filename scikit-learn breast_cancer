import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer


def load_dataset_as_df():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")  # 0/1
    df = pd.concat([X, y], axis=1)
    target_names = data.target_names  # e.g., ['malignant', 'benign']
    return df, data.feature_names, target_names


def basic_overview(df: pd.DataFrame, target_col="target"):
    print("=== BASIC OVERVIEW ===")
    print("Shape (rows, cols):", df.shape)
    print("\nColumns:", df.columns.tolist())

    print("\n=== TARGET DISTRIBUTION ===")
    counts = df[target_col].value_counts().sort_index()
    print(counts)
    print("Target ratio:", (counts / counts.sum()).to_dict())

    print("\n=== MISSING VALUES ===")
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if len(miss) == 0:
        print("No missing values found.")
    else:
        print(miss)

    print("\n=== DUPLICATES ===")
    dup = df.duplicated().sum()
    print("Duplicate rows:", dup)

    print("\n=== BASIC STATS (features) ===")
    feat_df = df.drop(columns=[target_col])
    print(feat_df.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]].head(10))
    print("\n(Showing first 10 features; you can remove .head(10) to print all.)")


def correlation_analysis(df: pd.DataFrame, target_col="target", top_k=15):
    """
    - Feature-feature correlation heatmap (top correlated pairs)
    - Correlation with target using Pearson (works since target is 0/1)
      This is equivalent to point-biserial correlation for binary target.
    """
    print("\n=== CORRELATION WITH TARGET (approx point-biserial) ===")
    feat_df = df.drop(columns=[target_col])
    y = df[target_col]

    # correlation of each feature with target
    corr_with_target = feat_df.apply(lambda col: col.corr(y))
    corr_sorted = corr_with_target.sort_values(key=lambda s: s.abs(), ascending=False)

    print(corr_sorted.head(top_k))
    return corr_sorted


def top_correlated_feature_pairs(df: pd.DataFrame, target_col="target", top_k=15):
    """
    Find highly correlated feature pairs (multicollinearity hints).
    """
    feat_df = df.drop(columns=[target_col])
    corr = feat_df.corr()

    # Take upper triangle (without diagonal)
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    pairs = (
        upper.stack()
        .rename("corr")
        .sort_values(key=lambda s: s.abs(), ascending=False)
        .head(top_k)
    )

    print("\n=== TOP CORRELATED FEATURE PAIRS ===")
    print(pairs)
    return pairs


def iqr_outlier_report(df: pd.DataFrame, target_col="target", top_k=15):
    """
    Simple outlier count per feature using IQR rule.
    """
    feat_df = df.drop(columns=[target_col])
    outlier_counts = {}

    for col in feat_df.columns:
        q1 = feat_df[col].quantile(0.25)
        q3 = feat_df[col].quantile(0.75)
        iqr = q3 - q1
        lo = q1 - 1.5 * iqr
        hi = q3 + 1.5 * iqr
        outlier_counts[col] = int(((feat_df[col] < lo) | (feat_df[col] > hi)).sum())

    outlier_series = pd.Series(outlier_counts).sort_values(ascending=False)
    print("\n=== OUTLIER REPORT (IQR rule) ===")
    print(outlier_series.head(top_k))
    print("\n(Showing top features with most outliers.)")
    return outlier_series


# ----------------
# Plotting helpers
# ----------------
def plot_target_distribution(df: pd.DataFrame, target_names=None, target_col="target"):
    counts = df[target_col].value_counts().sort_index()
    labels = counts.index.astype(str).tolist()
    if target_names is not None and len(target_names) == len(labels):
        # map 0/1 -> names
        labels = [f"{i} ({target_names[i]})" for i in counts.index]

    plt.figure()
    plt.bar(labels, counts.values)
    plt.title("Target Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_feature_histograms(df: pd.DataFrame, target_col="target", cols=None, bins=30):
    feat_df = df.drop(columns=[target_col])
    if cols is None:
        cols = feat_df.columns[:6]  # default: first 6 features

    for col in cols:
        plt.figure()
        plt.hist(feat_df[col].values, bins=bins)
        plt.title(f"Histogram: {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.show()


def plot_feature_by_class_boxplot(df: pd.DataFrame, feature, target_col="target"):
    """
    Boxplot feature grouped by class.
    """
    classes = sorted(df[target_col].unique())
    data = [df.loc[df[target_col] == c, feature].values for c in classes]

    plt.figure()
    plt.boxplot(data, labels=[str(c) for c in classes])
    plt.title(f"Boxplot by class: {feature}")
    plt.xlabel("Class")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()


def plot_corr_heatmap(df: pd.DataFrame, target_col="target", top_features=12):
    """
    Heatmap of correlations among top_features (by abs corr with target).
    (We keep it small to stay readable without seaborn.)
    """
    corr_sorted = correlation_analysis(df, target_col=target_col, top_k=top_features)
    top_cols = corr_sorted.head(top_features).index.tolist()

    corr = df[top_cols].corr().values

    plt.figure(figsize=(8, 6))
    plt.imshow(corr, aspect="auto")
    plt.xticks(range(len(top_cols)), top_cols, rotation=90)
    plt.yticks(range(len(top_cols)), top_cols)
    plt.title("Correlation Heatmap (Top features)")
    plt.colorbar()
    plt.tight_layout()
    plt.show()


# -------------
# Run all EDA
# -------------
if __name__ == "__main__":
    df, feature_names, target_names = load_dataset_as_df()

    basic_overview(df, target_col="target")

    corr_sorted = correlation_analysis(df, target_col="target", top_k=15)
    top_correlated_feature_pairs(df, target_col="target", top_k=15)
    iqr_outlier_report(df, target_col="target", top_k=15)

    # Plots
    plot_target_distribution(df, target_names=target_names, target_col="target")

    # Example: histogram of a few features
    plot_feature_histograms(df, target_col="target", cols=[
        "mean radius", "mean texture", "mean perimeter", "mean area"
    ])

    # Example: boxplot for the most target-correlated feature
    best_feature = corr_sorted.index[0]
    plot_feature_by_class_boxplot(df, best_feature, target_col="target")

    # Correlation heatmap for top features
    plot_corr_heatmap(df, target_col="target", top_features=10)
