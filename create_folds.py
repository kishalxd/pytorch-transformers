def create_folds(data, num_splits):
    data["kfold"] = -1
    kf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=11)
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data['Cover_Type'])):
        data.loc[v_, 'kfold'] = f
    return data
