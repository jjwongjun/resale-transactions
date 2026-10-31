# HDB Resale Price Prediction

## Project Description
This project implements a machine learning pipeline to predict HDB (Housing & Development Board) resale prices in Singapore.
The pipeline includes data cleaning, preprocessing, model training, and evaluation. Multiple regression models are trained and compared to find the best performing one based on various metrics.

## Prerequisites and Installation Instructions

### Requirements
- Anaconda or Miniconda
- Python 3.11
- Required packages listed in `requirements.txt`

### Installation
1. Clone the repository:
    ```
    git clone https://github.com/jjwongjun/resale-transactions.git
    cd resale-transactions
    ```

2. Create and activate a conda environment:
    ```
    conda create -n hdbenv python=3.11
    conda activate hdbenv
    ```

3. Install the required packages:
    ```
    pip install -r requirements.txt
    ```

## Pipeline Execution
To run the complete pipeline, execute:
```
python main.py
```
A different configuration file can be used with `python main.py --config path/to/config.yaml` (default: `./src/config.yaml`). The pipeline expects the data at `./data/resale_transactions.csv` (see `file_path` in the config).

This will:
1. Validate the configuration and load the data
2. Validate and clean the data
3. Split data into training, validation, and test sets
4. Train baseline models (linear and tree-based)
5. Perform hyperparameter tuning and report cross-validation results
6. Compare all models on the validation set and select the best one
7. Retrain the best model on training + validation data and evaluate it on the test set
8. Report feature importance of the final model

## Logical Flow of the Pipeline

### 1. Configuration (config.yaml)
The pipeline starts with loading configuration
parameters from `src/config.yaml` and checks that all required keys exist and have valid types and values, which includes:
- Data file path
- Target column name
- Feature categorization (numerical, nominal, ordinal)
- Train/validation/test split ratios and the random seed
- Hyperparameter grids for model tuning (linear models and gradient boosting)

### 2. Data Preparation (`DataPreparation` class)
The `DataPreparation` class handles:
- Logging a data validation report (duplicates, missing values per column, number of unique values in categorical columns)
- Removing duplicates
- Standardizing flat type names
- Converting storey ranges to numerical values
- Filling missing town and flat model names
- Extracting year and month from date
- Converting remaining lease information to months
- Creating a preprocessor for feature transformation (including imputation of missing values)

### 3. Model Training (`ModelTraining` class)
The `ModelTraining` class manages:
- Splitting data into training, validation, and test sets
- Training baseline models (Linear Regression, Ridge, Lasso, Random Forest, Histogram Gradient Boosting)
- Hyperparameter tuning for Ridge, Lasso and Histogram Gradient Boosting models
- Reporting the mean and standard deviation of R², MAE and RMSE across the cross-validation folds
- Evaluating models using multiple metrics (MAE, MSE, RMSE, R²)
- Selecting the best model based on validation R² score
- Retraining the selected model on training + validation data before the final test evaluation
- Reporting permutation feature importance of the final model

### 4. Main Execution (`main.py`)
The main script orchestrates the entire pipeline by:
- Loading and validating the configuration, and loading the data (with error handling)
- Initializing data preparation and model training
- Running the training and evaluation process
- Logging a side-by-side comparison of all models
- Identifying the best model, retraining it on training + validation data and evaluating it on the test set

## Key Findings and Feature Handling

### Exploratory Data Analysis

Before diving into the model training process, an exploratory data analysis (EDA) was conducted to understand the data better and identify any potential issues. Here are some key findings from the EDA:

**EDA Findings and Explanations**: 
- `town_name` and `flatm_name` features have null entries/missing values
- Duplicate rows 
- `flat_type` has labelling error of'4 ROOM' and 'FOUR ROOM' categories
- Most resale prices cluster around a central range
- Resale prices vary widely
- July 2018 shows the highest number of listings while February 2017 and January 2018 show the lowest number of listings compared to other months
- Number of listings drops significantly around February each year, because Feb is shorter month and BTO sales typically occurs in Feb
- Resale listings are influenced by seasonal or market factors rather than a consistent monthly trend
- Flat type increases in size, also spread of prices becomes wider
- Median resale price increases with the size of the flat type, specifically highest for "EXECUTIVE" and "MULTI-GENERATION" 
- IQR is wider for larger flat types like "EXECUTIVE" and "MULTI-GENERATION"
- Presence of outliers in the resale prices for '2 ROOM', '3 ROOM', '4 ROOM', '5 ROOM', and 'EXECUTIVE' flats
- Outlier 3-room flats are terrace houses
- Frequencies of 'Yishun', 'Bedok', and 'Punggol' do not match the frequencies of their respective numerical identifiers, which are 26, 2, and 18, respectively
- Towns with the highest number of resale units are Jurong West, Woodlands, and Sengkang; towns with the fewest resale units are Central Area, Marine Parade, and Bukit Timah
- Larger flat types (such as Executive and Multi-Generation flats) tend to cluster towards higher floor areas and resale prices. Smaller flat types (like 1-room and 2-room flats) cluster towards lower floor areas and resale prices
- Wide spread of resale prices for similar floor areas, especially in mid-sized flats (3-room, 4-room, and 5-room flats)
- Strong positive Pearson correlation between 'floor_area_sqm' and 'resale_price', moderate positive Pearson correlation between 'lease_commence_date' and 'resale_price'
- Strong positive Spearman Rank monotonic correlation between the 'floor_area' and the 'resale_price'. Moderate positive Spearman Rank monotonic relationship between the 'lease_commence_date' and the 'resale_price'. Weak positive Spearman Rank monotonic correlation between the 'storey_range' and the 'resale_price'

**Additional EDA steps** (see `eda.ipynb`): a 20% working sample is used for fast iteration (after removing duplicates on the full data); derived features (`remaining_lease_months`, `age_of_flat`, `price_per_sqm`) are created before the analysis (`price_per_sqm` is only used in the EDA because it is derived from the target); resale price is analysed over time and by town; `street_name` is frequency-encoded; influential outliers are identified with Cook's distance and kept because they are genuine transactions; encodings for `town_name` are compared (one-hot performs best for a linear model); and permutation importance is used to see which features are most predictive.

### Data Cleaning
- Storey ranges are converted to their average values because feature is stored as an object (string) and conversion maintains ordinal nature
- Remaining lease information is extracted and converted to total months because feature is stored as an object (string) with various inconsistent formats
- Year and month are extracted from the transaction date because feature is stored as an object (string) that contains both year and month format
- Missing town and flat model names are filled using ID-to-name mappings because missing values need to be accurately and systematically addressed, for machine learning algorithm use
- Negative `lease_commence_date` values are converted to positive values because they are data entry errors
- A data validation report (duplicates, missing values, unique values per categorical column) is logged before cleaning

### Feature Processing
- **Numerical Features**: `floor_area_sqm`,
`remaining_lease_months`, `lease_commence_date`, `year`
- Missing values are imputed with the median, then the features are standardized using `StandardScaler` because the numerical features are on different scales, which can adversely impact the performance of linear models and regularization if not addressed. Tree-based models are not affected by scaling
- **Nominal Features**: `month`, `town_name`, `flatm_name` - Missing values are imputed with the most frequent category, then encoded using `OneHotEncoder` because these features have no intrinsic order
- **Ordinal Features**: `flat_type` - Encoded using `OrdinalEncoder` with predefined categories because flat types can be considered to have a natural ranking based on their size and number of rooms
- **Passthrough Features**: `storey_range` - Used as-is after converting to numerical values because already been ordinally encoded
- The preprocessor outputs a dense array so that both linear and tree-based models can use it. It is part of every model pipeline, so it is fitted on the training data only (no data leakage)

## Model Choices and Evaluation

### Models Implemented and Justifications
1. **Linear Regression**: Basic model without regularization because predicting housing prices is a classic regression problem and it sets the baseline for comparison with other models
2. **Ridge Regression**: Linear regression with L2 regularization because it includes a regularization term to prevent overfitting. Since different input features will have varying levels of influence on the resale price, the reduction of coefficient size caused by the shrinkage penalty translates to each predictor feature having less influence on the final prediction
3. **Lasso Regression**: Linear regression with L1 regularization because it includes a regularization term to prevent overfitting. It is similar to Ridge Regression except it not only shrinks the coefficients towards 0 but can also force some coefficients to be exactly 0, effectively performing built-in feature selection by excluding irrelevant features from the model
4. **Random Forest**: An ensemble of decision trees because resale prices depend on non-linear interactions (for example between town, flat type and lease), which linear models cannot capture. It needs no feature scaling and is robust to outliers
5. **Histogram Gradient Boosting**: Trees built one after another, each correcting the errors of the previous ones, because boosting is usually among the most accurate methods for tabular data and is fast on large datasets

### Hyperparameter Tuning
- Grid search is performed for Ridge, Lasso and Histogram Gradient Boosting models because tuning model parameters can achieve the best possible performance. Grid Search CV's process is exhaustive but it ensures the best combination
- Parameters tuned include `alpha` and `fit_intercept` for Ridge and Lasso, and `learning_rate` and `max_iter` for Histogram Gradient Boosting, because the right combination of these settings ensures balance and should perform well on unseen data
- 5-fold cross-validation is used during tuning because averaged across all 5 folds will give a more reliable estimate of model performance. The mean and standard deviation of R², MAE and RMSE across the folds are logged for each tuned model
- The best model is selected on the validation set and then retrained on training + validation data before it is evaluated once on the test set

### Evaluation Metrics
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual prices
- **MSE (Mean Squared Error)**: Average squared difference between predicted and actual prices
- **RMSE (Root Mean Squared Error)**: Square root of MSE, in the same unit as the target
- **R² (Coefficient of Determination)**: Proportion of variance explained by the model

### Results
Validation set (the final test set is only used once for the best model):

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Random Forest | 22,506 | 31,829 | 0.953 |
| Histogram Gradient Boosting (tuned) | 23,883 | 33,138 | 0.949 |
| Histogram Gradient Boosting | 28,025 | 38,298 | 0.931 |
| Linear / Ridge / Lasso (baseline and tuned) | about 42,400 | about 55,030 | 0.858 |

The tree-based models clearly outperform the linear models, which means that the relationship between the features and the resale price is non-linear. The best model, Random Forest, scores **MAE 22,411, RMSE 31,817 and R² 0.953** on the test set after being retrained on training + validation data (the best linear model reached R² 0.865). Permutation importance shows that `floor_area_sqm` and `town_name` are by far the most important features, followed by `lease_commence_date` and `storey_range`.

## Deployment Considerations
- Code is structured into modular well-organized components making code easier to maintain, more scalable, and ready for real world applications
- The configuration is validated and file loading, cleaning and training errors are logged with clear messages
- Possible next steps: unit tests for the cleaning functions, saving the fitted model and metrics, and testing a log transform of the target