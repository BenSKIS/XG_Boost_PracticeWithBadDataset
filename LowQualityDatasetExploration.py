import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns

df =pd.read_csv(r'E:/Kaggle/Medical_Prediction_9.22/archive/medical_conditions_dataset.csv')


# Check if any missing values exist
print(df.isnull().values.any())
# Show total missing values per column
print(df.isnull().sum())
# Show percentage of missing values per column
print((df.isnull().sum() / len(df)) * 100)
# Display rows with missing values
print(df[df.isnull().any(axis=1)])

msno.matrix(df)

numerical_features = df.select_dtypes(include=['int64', 'float64']).columns


numerical_features

df['age_missing'] = df['age'].isnull().astype(int)
df['bmi_missing'] = df['bmi'].isnull().astype(int)
df['blood_pressure_missing'] = df['blood_pressure'].isnull().astype(int)
df['glucose_missing'] = df['glucose_levels'].isnull().astype(int)

# List of numerical features to apply mean imputation
numerical_features_with_missing = ['age', 'bmi', 'blood_pressure', 'glucose_levels']

for feature in numerical_features_with_missing:
    df[feature] = df[feature].fillna(df[feature].mean())

# Verify that missing values are imputed
print(df.isnull().sum())


from scipy.stats import shapiro

numerical_features
for i, column in enumerate(numerical_features):
    stat, p = shapiro(df[column])
    print(f'Statistics={stat}, p={p}')
    if p > 0.05:
        print(f"{column} is normally distributed")
    else:
        print(f"{column} is not normally distributed")

from scipy.stats import kurtosis

# Calculate kurtosis for a column in the dataframe
for i, column in enumerate(numerical_features):
    kurt_value = kurtosis(df[column], fisher=True)  # Fisher=True returns excess kurtosis
    print(f'Excess Kurtosis: {kurt_value}')
    if kurt_value > 0:
        print(f"{column} is leptokurtic (heavy tails)")
    elif kurt_value < 0:
        print(f"{column} is platykurtic (light tails)")
    else:
        print(f"{column} is mesokurtic (normal tails)")
        
        
from scipy.stats import skew

# Calculate skewness for a specific column in the dataframe
for i, column in enumerate(numerical_features):
    skew_value = skew(df[column])
    print(f'Skewness: {skew_value}')
    if skew_value > 0:
        print(f"{column} is positively skewed (right skewed)")
    elif skew_value < 0:
        print(f"{column} is negatively skewed (left skewed)")
    else:
        print(f"{column} is symmetric")

# Melt the DataFrame for better plotting of multiple features
df_melted = pd.melt(df[numerical_features])

# Create a subplot for each feature
fig, axes = plt.subplots(nrows=1, ncols=len(numerical_features), figsize=(15, 6))

# Plot each numerical feature in a separate subplot
for i, column in enumerate(numerical_features):
    sns.boxplot(y=column, data=df, ax=axes[i])  # Create a boxplot for each column
    axes[i].set_title(f'Boxplot of {column}')   # Set the title for each subplot

# Adjust layout so subplots do not overlap
plt.tight_layout()
#plt.show()



categorical_features = df.select_dtypes(include=['object']).columns
df[categorical_features] = df[categorical_features].astype('category')

dfd = df.drop(columns=['id','full_name'])
label = dfd['condition']
dfd = dfd.drop(columns=['condition'])

label.value_counts()






dfd_categorical_features = dfd.select_dtypes(include=['category']).columns
dfd_categorical_features
dfd.head()



from sklearn.preprocessing import LabelEncoder

# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Apply label encoding to each categorical feature
for col in dfd_categorical_features:
    dfd[col] = label_encoder.fit_transform(dfd[col])

# Check the results
print(dfd.dtypes)
print(dfd.head())

label = label_encoder.fit_transform(label)
features = dfd

from imblearn.over_sampling import SMOTE
from collections import Counter

# Assuming 'features' is your feature matrix and 'label' is your target vector
print("Original class distribution:", Counter(label))

# Apply SMOTE for oversampling the minority classes
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(features, label)

# Check the new class distribution
print("Resampled class distribution:", Counter(y_resampled))

features = X_resampled
label = y_resampled



from sklearn.preprocessing import MinMaxScaler

# List of columns to scale
columns_to_scale = ['age', 'bmi', 'blood_pressure', 'glucose_levels']

# Instantiate the scaler
scaler = MinMaxScaler()

# Apply the scaler to each column and replace the original values
for column in columns_to_scale:
    features[[column]] = scaler.fit_transform(features[[column]])


###Model Training
from sklearn.model_selection import train_test_split, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)

# Instantiate the XGBoost classifier
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# 5-fold Cross-Validation
cv_scores = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring='accuracy')

# Print cross-validation scores
print("Cross-validation scores: ", cv_scores)
print("Mean CV accuracy: ", cv_scores.mean())


# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

# Evaluate the model
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f'Training Accuracy: {train_accuracy}')
print(f'Test Accuracy: {test_accuracy}')


