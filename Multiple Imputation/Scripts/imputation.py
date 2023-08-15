import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, SVR
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline



def multiple_imputations(train_data, test_data, numerical_vars, imputation_models, num_imputations=1, noise_factor=0.5):
    """
    Performs multiple imputations in the style of MICE, with optional added noise

    Parameters:
        train_data (pd.DataFrame): Training data to be imputed.
        test_data (pd.DataFrame): Testing data to be imputed.
        numerical_vars (list): List of variables that are continuous/integers.
        imputation_models (dict): Dictionary of models to be used for imputing each column.
        num_imputations (int): Number of imputations to be performed.
        noise_factor (float): The threshold for determining when to change the category for categorical imputations. 1
        turns this feature off.

    Returns:
        imputations (dict): Dictionary of imputed dataframes for both train and test data.
    """
    # Define categorical variables
    categorical_vars = [col for col in train_data.columns if col not in numerical_vars]
    all_vars = numerical_vars + categorical_vars

    # Create a dictionary to store imputations
    imputations = {}
    
    # Perform multiple imputations
    for imputation_num in range(num_imputations):
        # Copy the original data for the first imputation,
        # and use the previous imputation for subsequent ones
        if imputation_num == 0:
            imputed_train_data = train_data.copy()
            imputed_test_data = test_data.copy()
        else:
            imputed_train_data = imputations[f'Imputation_{imputation_num}']['Train'].copy()
            imputed_test_data = imputations[f'Imputation_{imputation_num}']['Test'].copy()

        # Impute each variable in order of missingness
        for var in imputed_train_data.isnull().sum().sort_values(ascending=False).index:
            # Identify observed and missing data
            missing_indices_train = train_data[var].isnull()
            missing_indices_test = test_data[var].isnull()
            observed_indices_train = ~missing_indices_train
            observed_indices_test = ~missing_indices_test
            
            # Prepare other variables used in imputation
            vars2 = [v for v in all_vars if v != var]
            
            # Only proceed if there are observed values and missing values in at least one data set
            if observed_indices_train.any() and (missing_indices_train.any() or missing_indices_test.any()):
                # Separate predictor and target variables
                X_train = imputed_train_data.loc[:, vars2].copy()
                y_train = imputed_train_data.loc[:, var].copy()
                X_test = imputed_test_data.loc[:, vars2].copy()
                y_test = imputed_test_data.loc[:, var].copy()
                
                # Initialize imputer
                imputer = SimpleImputer(strategy='constant')

                # Impute missing values in predictor variables
                X_train_imputed = pd.DataFrame(columns=vars2)
                X_test_imputed = pd.DataFrame(columns=vars2)
                
                for col in X_train.columns:
                    X_train_imputed[col] = imputer.fit_transform(X_train[[col]]).ravel()
                    X_test_imputed[col] = imputer.transform(X_test[[col]]).ravel()
                    
                # One-hot encode categorical variables
                if var in categorical_vars:
                    cat_vars = [v for v in categorical_vars if v != var]
                else:
                    cat_vars = categorical_vars
                
                encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                X_train_encoded = encoder.fit_transform(X_train_imputed.loc[:, cat_vars])
                X_test_encoded = encoder.transform(X_test_imputed.loc[:, cat_vars])
                encoded_column_names = encoder.get_feature_names_out(cat_vars)
                X_train_encoded = pd.DataFrame(X_train_encoded, columns=encoded_column_names)
                X_test_encoded = pd.DataFrame(X_test_encoded, columns=encoded_column_names)

                # Standardize numeric variables
                if var in numerical_vars:
                    num_vars = [v for v in numerical_vars if v != var]
                else:
                    num_vars = numerical_vars
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_imputed.loc[:, num_vars])
                X_test_scaled = scaler.transform(X_test_imputed.loc[:, num_vars])
                X_train_scaled = pd.DataFrame(X_train_scaled, columns=num_vars)
                X_test_scaled = pd.DataFrame(X_test_scaled, columns=num_vars)
                
                # Combine processed categorical and numeric variables
                X_train_pre = pd.concat([X_train_encoded, X_train_scaled], axis=1)
                X_test_pre = pd.concat([X_test_encoded, X_test_scaled], axis=1)
                
                # Reset indices to align data
                X_train_pre.reset_index(drop=True, inplace=True)
                observed_indices_train.reset_index(drop=True, inplace=True)
                missing_indices_test.reset_index(drop=True, inplace=True)
                imputed_train_data.reset_index(drop=True, inplace=True)
                missing_indices_train.reset_index(drop=True, inplace=True)
                
                # Separate observed and missing data for imputation
                X_train_obs = X_train_pre.loc[observed_indices_train, :].copy()
                y_train_obs = imputed_train_data.loc[observed_indices_train, var].copy()
                X_train_pred = X_train_pre.loc[missing_indices_train, :].copy()
                X_test_pred = X_test_pre.loc[missing_indices_test.values, :].copy()

                # Label encode categorical target variable
                if var in categorical_vars:    
                    le = LabelEncoder()
                    y_train_obs = le.fit_transform(y_train_obs)
                 
                # Set random state for reproducibility and num_class for xgboost
                if 'random_state' in imputation_models[var].get_params().keys():
                    imputation_models[var].random_state = imputation_num
                    
                if 'num_class' in imputation_models[var].get_params().keys() and var in categorical_vars:
                    imputation_models[var].num_class = len(y_train_obs.unique())
                    
                # Impute missing target values using the specified model
                if len(X_train_pred) == 0:
                    imputed_train_values = []
                else:
                    model = imputation_models[var]
                    model.fit(X_train_obs, y_train_obs)
                    imputed_train_values = model.predict(X_train_pred)
                    
                    if len(X_test_pred) == 0:
                        imputed_test_values = []
                    else:
                        imputed_test_values = model.predict(X_test_pred)
                
                #Reverse the encoding transformation for cat columns
                if var in categorical_vars:
                    y_train_obs = le.inverse_transform(y_train_obs)
                    imputed_train_values = le.inverse_transform(imputed_train_values)
                    imputed_test_values = le.inverse_transform(imputed_test_values)
                    
                    

                # Add noise to the imputed values
                if var in numerical_vars:
                    # For numerical variables, add Gaussian noise
                    noise_train = np.random.normal(0, np.std(y_train_obs), len(imputed_train_values))
                    noise_test = np.random.normal(0, np.std(y_train_obs), len(imputed_test_values))
                    imputed_train_values += noise_train
                    imputed_test_values += noise_test
                else:
                     #For categorical variables, randomly change the category based on the noise factor
                    unique, counts = np.unique(y_train_obs, return_counts=True)
                    observed_values = unique.tolist()
                    observed_probs = counts/sum(counts)
                    observed_probs = observed_probs.tolist()
                    noise_train = np.random.normal(size=len(imputed_train_values))
                    noise_test = np.random.normal(size=len(imputed_test_values))
                    for i in range(len(imputed_train_values)):
                        if abs(noise_train[i]) > noise_factor:
                            imputed_train_values[i] = np.random.choice(observed_values, p=observed_probs)
                    for i in range(len(imputed_test_values)):
                        if abs(noise_test[i]) > noise_factor:
                            imputed_test_values[i] = np.random.choice(observed_values, p=observed_probs)

                # Insert imputed values into the data
                if len(imputed_train_values) > 0:
                    imputed_train_data.loc[missing_indices_train, var] = imputed_train_values
                if len(imputed_test_values) > 0:
                    imputed_test_data.loc[missing_indices_test.values, var] = imputed_test_values
            
        ## Store the imputed data
        imputations[f'Imputation_{imputation_num+1}'] = {'Train': imputed_train_data, 'Test': imputed_test_data}

    return imputations




def score(original_train_data, y_train, num_vars, models, imputation_models, scoring, num_imputations=1, hyperparameters=None, rs_iter=5):
    """
    Evaluates models using k-fold cross-validation on imputed datasets.

    Parameters:
        original_train_data (pd.DataFrame): Original training data.
        y_train (pd.Series): Training target data.
        num_vars (list): List of numeric variables.
        models (dict): Dictionary of ML models {model_name: model}.
        imputation_models (dict): Dictionary of models for imputation.
        scoring (str): Scoring metric.
        num_imputations (int): Number of imputations to be performed.
        hyperparameters (dict, optional): Dictionary of hyperparameters for tuning.

    Returns:
        base_scores (dict): Dictionary of scores of baseline models.
        best_params (dict): Dictionary of parameters and scores of the best tuned models.
    """
    # Create dictionaries to store base scores and best parameters
    base_scores = {model_name: {} for model_name in models}
    best_params = {model_name: {} for model_name in models}

    # Define K-fold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=0)

    # Define preprocessor
    cat_vars = list(set(original_train_data.columns) - set(num_vars))
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_vars),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_vars)
        ]
    )
    
    # Evaluate each model
    for model_name, model in models.items():
        
        # Initialize variables to store scores for each imputation and fold
        cv_scores = []
        cv_best_params = []
        
        # Perform evaluation on each fold
        for train_index, val_index in cv.split(original_train_data):
            train_data = original_train_data.iloc[train_index].copy()
            val_data = original_train_data.iloc[val_index].copy()
            
            # Perform multiple imputations on the training data and validation data
            imputed_datasets = multiple_imputations(train_data, val_data, num_vars, imputation_models, num_imputations)
            
            # Check if imputed_datasets is empty
            if not imputed_datasets:
                print(f"No imputed datasets were generated for model {model_name}.")
                continue

            for dataset_name, dataset in imputed_datasets.items():
                X_train_imputed = dataset['Train']
                X_val_imputed = dataset['Test']

                # Create a pipeline that includes the preprocessing and the model
                pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

                # Generate baseline model scores
                try:
                    score = pipeline.fit(X_train_imputed, y_train.iloc[train_index]).score(X_val_imputed, y_train.iloc[val_index])
                    cv_scores.append(score)
                except Exception as e:
                    print(f"An error occurred while fitting or scoring the model {model_name} on dataset {dataset_name}: {e}")
                
                # Hyperparameter tuning if hyperparameters are provided
                if hyperparameters and model_name in hyperparameters:
                    try:
                        grid = RandomizedSearchCV(pipeline, hyperparameters[model_name], cv=cv, scoring=scoring, n_iter=rs_iter)
                        grid.fit(X_train_imputed, y_train.iloc[train_index])
                        cv_best_params.append({
                            'best_params': grid.best_params_,
                            'best_score': grid.best_score_
                        })
                    except Exception as e:
                        print(f"An error occurred during hyperparameter tuning for model {model_name} on dataset {dataset_name}: {e}")
        
        # Check if cv_scores or cv_best_params are empty
        if not cv_scores:
            print(f"No scores were generated for model {model_name}.")
        if not cv_best_params:
            print(f"No best parameters were found for model {model_name}.")
            
        # Average scores and best parameters over all imputations and folds
        base_scores[model_name] = np.mean(cv_scores) if cv_scores else None
        best_params[model_name] = {
            'best_params': [bp['best_params'] for bp in cv_best_params],
            'best_score': np.mean([bp['best_score'] for bp in cv_best_params]) if cv_best_params else None,
            'best_score_std': np.std([bp['best_score'] for bp in cv_best_params]) if cv_best_params else None
        }

    return base_scores, best_params