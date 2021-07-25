import pandas as pd
from sklearn.model_selection import StratifiedKFold
from config import Config


def create_frame(dataframe):
    frame = pd.read_csv('./data/'+dataframe)
    frame['kfold'] = -1
    Fold = StratifiedKFold(
        n_splits=Config.n_fold,
        shuffle=True,
        random_state=Config.seed
    )
    for n, (train_index, val_index) in enumerate(Fold.split(
                                        frame, frame[Config.target_col]
                                    )):
        frame.loc[val_index, 'kfold'] = int(n)
        frame['kfold'] = frame['kfold'].astype(int)

    frame.to_csv('/data/fold_'+dataframe, index=False)


if __name__ == "__main__":
    create_frame(Config.filename)
