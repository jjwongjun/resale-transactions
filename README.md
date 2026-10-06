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

git clone https://github.com/yourusername/yourreponame.git
cd hdb-resale-price-prediction

2. Create and activate a conda environment:
    ```
    conda create -n hdbenv python=3.11
    conda activate hdbenv
    ```

3. Install the required packages:

pip install -r requirements.txt

## Pipeline Execution
To run the complete pipeline, execute:

python main.py

This will:
1. Load and clean the data
2. Preprocess features
3. Split data into training, validation, and test sets
4. Train baseline models
5. Perform hyperparameter tuning
6. Evaluate models and select the best one

## Logical Flow of the Pipeline

### 1. Configuration (config.yaml)
The pipeline starts with loading configuration
parameters from `src/config.yaml`, which includes:
- Data file path
- Target column name
- Feature categorization (numerical, nominal, ordinal)
- Train/validation/test split ratios
- Hyperparameter grid for model tuning

### 2. Data Preparation (`DataPreparation` class)
The DataPreparation' class handles:
- Removing duplicates
- Standardizing flat type names
- Converting storey ranges to numerical values
- Filling missing town and flat model names
- Extracting year and month from date
- Converting remaining lease information to months
- Creating a preprocessor for feature transformation

### 3. Model Training (`ModelTraining` class)
The ModelTraining' class manages:
- Splitting data into training, validation, and test
sets
- Training baseline models (Linear Regression, Ridge, Lasso)
- Hyperparameter tuning for Ridge and Lasso models
- Evaluating models using multiple metrics (MAE, MSE, RMSE, R²)
- Selecting the best model based on R² score

### 4. Main Execution (`main.py`)
The main script orchestrates the entire pipeline by:
- Loading configuration and data
- Initializing data preparation and model training
- Running the training and evaluation process
- Identifying and evaluating the best model on the test set

## Key Findings and Feature Handling

### Exploratory Data Analysis

Before diving into the model training process, an
exploratory data analysis (EDA) was conducted to
understand the data better and identify any potential
issues. Here are some key findings from the EDA:

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

### Data Cleaning
- Storey ranges are converted to their average values because feature is stored as an object (string) and conversion maintains ordinal nature
- Remaining lease information is extracted and converted to total months because feature is stored as an object (string) with various inconsistent formats
- Year and month are extracted from the transaction date because feature is stored as an object (string) that contains both year and month format
- Missing town and flat model names are filled using ID-to-name mappings because missing values need to be accurately and systematically addressed, for machine learning algorithm use

### Feature Processing
- **Numerical Features**: `floor_area_sqm`,
`remaining_lease_months`, `lease_commence_date`, `year`
- Standardized using `StandardScaler` because the numerical features have approximately normal distributions. In addition, the numerical features are on different scales, which can adversely impact the performance of machine learning algorithms if not addressed. Lastly, machine learning algorithms assume normally distributed data and are sensitive to feature scales, particularly those that rely on distance metrics
- **Nominal Features**: `month`, `town name`, `flatm_name` - Encoded using `OneHotEncoder` because these features have no intrinsic order
- **Ordinal Features**: `flat_type` - Encoded using `OrdinalEncoder` with predefined categories because flat types can be considered to have a natural ranking based on their size and number of rooms
- **Passthrough Features**: `storey_range` - Used as-is after converting to numerical values because already been ordinally encoded

## Model Choices and Evaluation

### Models Implemented and Justifications
1. **Linear Regression**: Basic model without regularization because predicting housing prices is a classic regression problem and it sets the baseline for comparison with other models
2. **Ridge Regression**: Linear regression with L2 regularization because it includes a regularization term to prevent overfitting. Since different input features will have varying levels of influence on the resale price, the reduction of coefficient size caused by the shrinkage penalty translates to each predictor feature having less influence on the final prediction
3. **Lasso Regression**: Linear regression with L1 regularization because it includes a regularization term to prevent overfitting. It is similar to Ridge Regression except it not only shrinks the coefficients towards 0 but can also force some coefficients to be exactly 0, effectively performing built-in feature selection by excluding irrelevant features from the model

### Hyperparameter Tuning
- Grid search is performed for Ridge and Lasso models because tuning model parameters can achieve the best possible performance. Grid Search CV's process is exhaustive but it ensures the best combination
- Parameters tuned include alpha' and fit_intercept because the right combination of these settings ensures balance and should perform well on unseen data
- 5-fold cross-validation is used during tuning because averaged across all 5 folds will give a more reliable estimate of model performance.

### Evaluation Metrics
- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and actual prices
- **MSE (Mean Squared Error)**: Average squared difference between predicted and actual prices
- **RMSE (Root Mean Squared Error)**: Square root of MSE, in the same unit as the target
- **R² (Coefficient of Determination)**: Proportion of variance explained by the model

## Deployment Considerations
- Code is structured into modular well-organized componenets making code easier to maintain, more scalable, and ready for real world applications