import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set fonts for plotting functions
font = {'family': 'serif',
        'weight': 'normal',
        'size': 15,
        }


### Evaluation functions ###

'''
NOTE: For installation and usage of imblearn functions, refer to the docs:
http://contrib.scikit-learn.org/imbalanced-learn/stable/install.html
'''


def r2_cv(model, X, y, n_folds):
    '''
    INPUT:
        model; the model object to use to fit the data
        X; a numpy array of predictors
        y; a target variable vector
        k_folds; the number of folds in the k-fold cross
    OUTPUT:
        r2; the cross-validated R-Squared
    AUTHOR: jonoleson
    '''
    kf = KFold(n_folds, shuffle=True)
    r2 = cross_val_score(model, X, y, scoring='r2', cv = kf)
    return(r2)


def run_undersampling(X, y, sampling_strategy=0.33):
    '''
    INPUT:
        X; a numpy array of predictors
        y; a binary target vector
        ratio; the desired ratio of the minority class to the majority class
    OUTPUT:
        usx; the undersampled predictor numpy array
        usy; the undersampled target vector
    NOTES:
        usx and usy have been undersampled such that the ratio of the minority
        class to the majority class is equal to the 'ratio' parameter
    '''
    US = RandomUnderSampler(sampling_strategy)
    usx, usy = US.fit_sample(X, y)
    return usx, usy


def run_smote(X, y, **kwargs):
    '''
    INPUT:
        X; a numpy array of predictors
        y; a binary target vector
        **kwargs, other keyword arguments to SMOTE; see imblearn docs
    OUTPUT:
        smx; predictor array with synthetically oversampled minority examples
        smy; target vector with synthetically oversampled minority examples
    NOTES:
        Takes a predictor numpy array, X, and a binary target vector, y, and returns
        arrays, smx, and smy, where the minority class has been synthetically
        oversampled using the SMOTE method, creating balanced classes
    '''
    sm = SMOTE(**kwargs)
    smx, smy = sm.fit_sample(X, y)
    return smx, smy


def run_smoteenn(X, y, sampling_strategy='auto'):
    '''
    INPUT:
        X; a numpy array of predictors
        y; a binary target vector
        **kwargs, other keyword arguments to SMOTEENN; see imblearn docs
    OUTPUT:
        smx; predictor array with synthetically oversampled minority examples
        smy; target vector with synthetically oversampled minority examples
    NOTES:
        Takes a predictor numpy array, X, and a binary target vector, y, and returns
        arrays, smx, and smy, where the minority class has been synthetically
        oversampled using the SMOTE method, then cleaned with ENN
    '''
    sm = SMOTEENN(sampling_strategy=sampling_strategy, n_jobs=-1, random_state=1)
    smx, smy = sm.fit_resample(X, y)
    return smx, smy


def run_prob_cv(X, y, clf, n_folds, undersample=False, smote=False, smoteenn=False, resampling=None, sampling_strategy=0.33, **kwargs):
    '''
    INPUT:
        X; a numpy array of predictors
        y; a binary target vector
        clf; a classification model object
        n_folds; integer, number of folds to use in the kfold cross-val
        undersample; whether to apply undersampling to X and y
        smote; whether to apply SMOTE to X and y (smote kwargs not included here)
        resampling_technique; 'undersample', 'smote', or 'smoteenn'
        ratio; the desired ratio of the minority class to the majority class
        **kwargs; optional keyword arguments to clf fit method
    OUTPUT:
        y_prob; a vector of positive class probabilities, produced out-of-sample
                with kfold cross-val
    '''
    kf = KFold(n_folds, shuffle=True)
    y_prob = np.zeros(len(y))
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        if undersample == True:
            X_train, y_train = run_undersampling(X_train, y_train, sampling_strategy)
        elif smote:
            X_train, y_train = run_smote(X_train, y_train)
        elif smoteenn:
            X_train, y_train = run_smoteenn(X_train, y_train, sampling_strategy)
        clf.fit(X_train, y_train, **kwargs)
        # Predict probabilities, not classes
        y_prob[test_index] = clf.predict_proba(X_test)[:,1]
    return y_prob


def run_reg_cv(X, y, clf, n_folds):
    '''
    INPUT:
        X; a numpy array of predictors
        y; a numeric target vector
        clf; a regression model object
        n_folds; integer, number of folds to use in the kfold cross-val
        **kwargs; optional keyword arguments to clf_class
    OUTPUT:
        y_pred; a vector of regression predictions, produced out-of-sample
                with k-fold cross-val
    '''
    kf = KFold(n_folds, shuffle=True)
    y_pred = np.zeros(len(y))
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf.fit(X_train, y_train)
        # Predict y
        y_pred[test_index] = clf.predict(X_test)
    return y_pred


def run_reg_cv_pred_ints(X, y, n_folds, percentile=95, **kwargs):
    '''
    INPUT:
        X; a numpy array of predictors
        y; a numeric target vector
        n_folds; integer, number of folds to use in the kfold cross-val
        percentile; the percentage prediction interval to use
        **kwargs; optional keyword arguments to Random Forest Regressor
    OUTPUT:
        y_pred; a vector of regression predictions, produced out-of-sample
                with k-fold cross-val
        y_ints; an array of dimensions (len(X), 2) containing the low and
                high bound of the prediction interval for each
                point-prediction
    NOTES:
        Function is only designed to work with a Random Forest Regressor.
        The prediction intervals are produced by outputting predictions
        from each tree in the forest for each prediction, then taking
        the predictions at the 2.5th and 97.5th percentile of the
        resulting distribution (for a 95% CI, e.g.) to form each
        observation's prediction interval
    '''
    kf = KFold(n_folds, shuffle=True)
    y_pred = np.zeros(len(y))
    y_ints = np.zeros([len(y), 2])
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train = y[train_index]
        clf = RFR(**kwargs)
        clf.fit(X_train, y_train)
        # Predict y
        y_pred[test_index] = clf.predict(X_test)
        # Get prediction intervals
        for idx in test_index:
            preds = []
            for est in clf.estimators_:
                preds.append(est.predict(X[idx].reshape(1,-1))[0])
            y_ints[idx,0] = np.percentile(preds, (100 - percentile) / 2.)
            y_ints[idx,1] = np.percentile(preds, 100 - (100 - percentile) / 2.)

    return y_pred, y_ints


def build_bin_df(y, y_prob):
    '''
    INPUT:
        y; a numeric target vector
        y_prob; a positive class probability vector
    OUTPUT:
        bin_df; dataframe,  a gains-table showing the incremental capture of
                positive class examples by model score intervals of 0.1
                (scores range 0-1)
        dec_df; dataframe, a gains-table showing the incremental capture of
                positive class examples by population decile
    '''
    pred_df = pd.DataFrame(y, y_prob, columns= ['target_proportion'])
    pred_df = pred_df.sort_index()

    # Create predicted probability-range bins
    bins                 = np.arange(0,1.1,0.1)
    pred_df['prob_bins'] = pd.cut(pred_df.index, bins = bins)

    # Create population decile bins
    l = len(pred_df)
    arr = np.repeat(np.arange(10), l/10)
    for i in range(l-len(arr)):
        arr = np.append(arr, 9)
    pred_df['pop_decile'] = arr

    # Count observations in each probability bin
    bin_counts = pd.Series(pd.value_counts(pred_df.prob_bins)).sort_index()

    # Count observations in each population decile
    decile_counts = pd.Series(pd.value_counts(pred_df.pop_decile)).sort_index()

    # Calculate mean actual response rate within each predicted probability bin and attach the counts
    bin_df               = pred_df.groupby('prob_bins').mean()
    bin_df['bin_counts'] = bin_counts

    # Calculate mean actual response rate within each population decile (sorted by descending model score)
    dec_df                  = pred_df.groupby('pop_decile').mean()
    dec_df['decile_counts'] = decile_counts

    # Reset the index of the pred_df
    pred_df = pred_df.reset_index().rename(columns={'index': 'prob_response'})

    bin_df['target_count'] = pred_df.groupby('prob_bins').sum().target_proportion
    dec_df['target_count'] = pred_df.groupby('pop_decile').sum().target_proportion

    bin_df = bin_df[::-1]
    dec_df = dec_df[::-1]

    bin_df['cumulative_population'] = bin_df.bin_counts.cumsum(axis=0)
    bin_df['cumulative_targets'] = bin_df.target_count.cumsum(axis=0)
    bin_df['cumulative_population_percentage'] = (bin_df.cumulative_population.astype(float) / bin_df.bin_counts.sum())
    bin_df['cumulative_target_percentage'] = (bin_df.cumulative_targets.astype(float) / bin_df.target_count.sum())

    dec_df['cumulative_population'] = dec_df.decile_counts.cumsum(axis=0)
    dec_df['cumulative_targets'] = dec_df.target_count.cumsum(axis=0)
    dec_df['cumulative_population_percentage'] = (dec_df.cumulative_population.astype(float) / dec_df.decile_counts.sum())

    bin_df.drop('pop_decile', axis=1, inplace=True)

    return bin_df, dec_df


def get_gini(y, y_prob):
    '''
    INPUT:
        y; a numeric target vector
        y_prob; a positive class probability vector
    OUTPUT:
        gini; the gini coefficient calculated from
              the target vector and the class
              probability vector
    '''
    roc_auc = roc_auc_score(y, y_prob)
    gini    = round((2 * roc_auc) - 1, 3)
    return gini


def plot_roc(y, y_prob, title='ROC Curve'):
    '''
    INPUT:
        y; a numeric target vector
        y_prob; a positive class probability vector
        title; title for graph
    OUTPUT:
        plot; function plots an ROC curve from y and y_prob
    AUTHOR: jonoleson
    '''
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    roc_auc = roc_auc_score(y, y_prob)
    plt.figure(figsize=(8,8))
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()


def get_feature_importances(colnames, model):
    '''
    INPUT:
        colnames; the feature names of the model predictors, in the same order they
                  were inputted into the model
        model; the trained model object
    OUTPUT:
        feat_imps; the array of feature names with their respective importance,
                   sorted by importance
    NOTES:
        function only works with models that have a 'feature_importances_'
        attribute, such as Random Forest or Gradient Boosting
    '''
    feat_imps = model.feature_importances_
    feat_imps = sorted(zip(colnames, feat_imps), key=lambda x: x[1])[::-1]
    return feat_imps




def get_sensitivity_data(df, model_vars, sensitivity_vars, model, model_type='reg', missing_value=-999.0, low_bound_pct=0, scaled=False, verbose=False):
    '''
    INPUT:
        df; dataframe, same data used to train model
        model_vars; variables in the model
        sensitivity_vars; variables to get sensitivity data for
        model; the trained model object
        model_type; 'reg' or 'cls' for whether model is a regression or
                    classification model
        missing_value; imputed value used where variables are null
        low_bound; integer, what percentile to set the lower bound of the subject variable
    OUTPUT:
        sensistivity_frames; a dictionary of dataframes, where the keys are
                             variable names and the values are dataframes
                             containing data on how model output changes
                             with incremental adjustments to the
                             variable in question, holding all other
                             variables constant
    '''
    sensitivity_frames = {}

    for var in sensitivity_vars:
        if verbose==True:
            print(var)
        temp = df[model_vars].copy()

        if scaled:
            scaler = StandardScaler()
            scaled_df = temp.copy()
            scaled_df[model_vars] = scaler.fit_transform(scaled_df[model_vars])
            scaled_val_range = sorted(scaled_df[var].unique())

        val_range = sorted(df[var].unique())
        val_range = list(filter(lambda a: a != missing_value, val_range)) #Filter out missings

        if scaled == True:
            scaled_val_range = list(filter(lambda a: np.isnan(a) == False, scaled_val_range)) #Filter out missings

        if low_bound_pct > 0:
            low_bound = len(val_range) // low_bound_pct  #Get high, low bounds at selected percentile of distributions
            high_bound = len(val_range) - low_bound
        else:
            low_bound = 0
            high_bound = len(val_range) - 1

        val_range = np.linspace(val_range[low_bound], val_range[high_bound], num=200) #Reset val_range to 200 evenly spaced points between chosen bounds
        if scaled:
            scaled_val_range = np.linspace(scaled_val_range[low_bound], scaled_val_range[high_bound], num=200)

        var_value = var + '_value'
        var_frame = pd.DataFrame(np.zeros([len(val_range), 2]), columns=[var_value, 'average_prediction'])

        for i in range(len(val_range)):
            if verbose==True:
                if i % 100 == 0:
                    print(i)
            if scaled:
                val = scaled_val_range[i]
                scaled_df[var] = val
                X = scaled_df.values
                if model_type == 'reg':
                    preds = model.predict(X)
                elif model_type == 'cls':
                    preds = model.predict_proba(X)[:,1]
                else:
                    raise NameError('"model_type" must be "reg" or "cls"')
                var_frame.loc[i, var_value] = val_range[i] #present unscaled val in output
                var_frame.loc[i, 'average_prediction'] = np.mean(preds)
            else:
                val = val_range[i]
                temp[var] = val
                X = temp.values
                if model_type == 'reg':
                    preds = model.predict(X)
                elif model_type == 'cls':
                    preds = model.predict_proba(X)[:,1]
                else:
                    raise NameError('"model_type" must be "reg" or "cls"')
                var_frame.loc[i, var_value] = val
                var_frame.loc[i, 'average_prediction'] = np.mean(preds)
        sensitivity_frames[var] = var_frame

    return sensitivity_frames


def plot_sensitivities(sensitivity_frames, prefix=None, save=False):
    '''
    INPUT:
        sensistivity_frames; described above
        prefix; to modify the title
    OUTPUT:
        sensitivity plots for each each variable in the sensitivity frames dict
    '''
    for var in sensitivity_frames.keys():
        temp = sensitivity_frames[var]
        f, ax = plt.subplots(figsize=(10,10))
        colA, colB = temp.columns[0], temp.columns[1]
        ax.plot(temp[colA], temp[colB])
        ax.set_xlabel(colA)
        ax.set_ylabel(colB)
        if prefix is None:
            ax.set_title(var+' Sensitivity Plot')
        else:
            ax.set_title(prefix + ' ' + var + ' Sensitivity Plot')
        if save:
            plt.savefig(prefix+'_'+var+'_'+'sensitivity_plot')
        plt.show()


def get_feature_imps_over_time(df, vintage_var, folds, pred_vars, target_var, model_class, **kwargs):
    '''
    INPUT:
        df; dataframe containing observations over a time period
        vintage_var; year-month 'vintage', should look like YYYY-MM,
                     must be present in df
        folds; number of folds in which to obtain feature importances
        pred_vars; variables to get importances of over time
        target_var; model target variable
        model_class; class of model
        **kwargs; optional keyword arguments to model
    OUTPUT:
        df of variable importances over time
    '''

    # 'vintage_var' should look like YYYY-MM
    feat_imps_df = pd.DataFrame(pred_vars, columns=['var_name'])

    temp = df.copy()
    temp.sort_values(vintage_var, inplace=True)

    feat_imp_totals = np.zeros(len(pred_vars))
    i = 1

    feat_imp_cols = []

    for index in np.array_split(np.arange(len(temp)), folds):
        X = temp.iloc[index][pred_vars].values
        y = temp.iloc[index][target_var].values
        model = model_class(**kwargs)
        model.fit(X, y)

        if hasattr(model, 'coef_'):
            feat_imps = model.coef_[0]
            colname = 'coef_'+str(i)
        else:
            feat_imps = model.feature_importances_
            colname = 'feat_imp_'+str(i)
        feat_imp_cols.append(colname)
        feat_imps_df[colname] = feat_imps
        feat_imp_totals += feat_imps
        i += 1

    feat_imps_df['avg'] = feat_imp_totals / folds
    feat_imps_df['std'] = feat_imps_df[feat_imp_cols].std(axis=1)
    feat_imps_df['cov'] = feat_imps_df['std'] / feat_imps_df['avg'] # Coefficient of Variance

    return feat_imps_df


def plot_var_missingness_over_time(df, vintage_var, vars, missing_value=-999.0, title='Variable Missingness by Vintage'):
    '''
    INPUT:
        df; dataframe containing observations over a time period
        vintage_var; year-month 'vintage', should look like YYYY-MM,
                     must be present in df
        vars; variables to plot missingness of
        missing_value; value imputed to fill existing nulls, replaced with nulls
        title; plot title
    OUTPUT:
        variable missingness plot
    AUTHOR: jonoleson
    '''
    temp = df[[vintage_var]+vars].copy()
    missing_colnames = []

    for var in vars:
        # append 'missing' to var name if not using a missing indicator var name already
        if 'missing' not in var:
            var_missing = var+'_is_missing'
        else:
            var_missing = var
        temp[var_missing] = np.where(temp[var] == missing_value, 1, 0)
        missing_colnames.append(var_missing)

    missing_by_vintage = temp[[vintage_var]+missing_colnames].groupby(vintage_var).mean()
    missing_by_vintage = missing_by_vintage.transpose()

    f, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(missing_by_vintage, cmap="Blues", ax=ax)
    ax.set_title(title)
    plt.show()


def plot_var_distribution_over_time(df, vintage_var, vars, missing_value=-999.0):
    '''
    INPUT:
        df; dataframe containing observations over a time period
        vintage_var; year-month 'vintage', should look like YYYY-MM,
                     must be present in df
        vars; variables to plot distribution of over time
        missing_value; value imputed to fill existing nulls, replaced with nulls
        title; plot title
    OUTPUT:
        plots of variable distributions over time represented in df
    AUTHOR: jonoleson
    '''
    for var in vars:
        temp = df[[vintage_var, var]].copy()
        temp.loc[:,var] = temp[var].replace(missing_value, np.nan) #re-introduce nulls
        temp.sort_values(vintage_var, inplace=True)

        f, ax = plt.subplots(figsize=(10,8))
        sns.lvplot(x=vintage_var, y=var, data=temp, ax=ax)
        plt.xticks(rotation=70)
        ax.set_title(var+' Distribution by Vintage', fontdict=font)
        plt.show()


def plot_confusion_matrix(y, y_pred, class_names, colnames, figsize = (10,7), fontsize=14, cmap='coolwarm'):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    y: numpy.ndarray
        The true labels
    y_pred: numpy.ndarray
        The predicted labels
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    matrix = confusion_matrix(y, y_pred)
    df_cm = pd.DataFrame(
        matrix, index=class_names, columns=colnames,
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, cmap=cmap, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
