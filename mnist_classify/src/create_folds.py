import pandas as pd
from sklearn import model_selection
from sklearn import datasets


if __name__ == "__main__":
    data = datasets.fetch_openml(
        'mnist_784',
        version=1,
        return_X_y=True
    )

    pixel_values, targets = data
    df = pd.DataFrame(pixel_values)
    df.loc[:, 'target'] = targets
    df['kfold'] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    kf = model_selection.StratifiedKFold(n_splits=5)

    for fold, (train_index, test_index) in enumerate(kf.split(X=df, y=df.target.values)):
        df.loc[test_index, 'kfold'] = fold
    
    df.to_csv('./input/mnist_kfold.csv', index=False)
