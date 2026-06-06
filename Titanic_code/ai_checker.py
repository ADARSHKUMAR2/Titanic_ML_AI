import os
import json
import pandas as pd
import numpy as np
from openai import OpenAI
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import time
import pandas as pd
import json
from shared.config import Config
from shared.schemas import TitanicBatchResponse

# -------------------------------------------------------------------------
# 1. INITIALIZATION & SETUP
# -------------------------------------------------------------------------
# Replace with your actual API key or set it as an environment variable

print("Loading Titanic datasets...")
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

# Combine datasets temporarily for uniform feature processing
combined_df = pd.concat([train_df, test_df], ignore_index=True)

# -------------------------------------------------------------------------
# 2. AI FEATURE GENERATION (THE "REASONING" LAYER)
# -------------------------------------------------------------------------

def process_passenger_batch(batch_df: pd.DataFrame) -> list:
    """
    Passes a DataFrame chunk to OpenAI and returns a validated list of dictionaries 
    strictly matching the PassengerFeature schema.
    """
    # 1. Format the batch data cleanly for the prompt
    passenger_data_list = []
    for _, row in batch_df.iterrows():
        cabin = row['Cabin'] if pd.notna(row['Cabin']) else "Unknown"
        age = row['Age'] if pd.notna(row['Age']) else "Unknown"
        fare = row['Fare'] if pd.notna(row['Fare']) else "Unknown"
        family_size = int(row['SibSp'] + row['Parch'])

        passenger_data_list.append({
            "PassengerId": int(row['PassengerId']),
            "Name": row['Name'],
            "Sex": row['Sex'],
            "Age": age,
            "Pclass": int(row['Pclass']),
            "Fare": fare,
            "Cabin": cabin,
            "FamilySize": family_size
        })

    # 2. Build the system/user instruction set
    prompt = f"""
    You are an expert maritime historian analyzing Titanic passenger demographics.
    Analyze the following list of passengers and evaluate their survival profile based on historical dynamics:
    1. "Women and children first" protocols.
    2. Socio-economic privilege (Class, Fare).
    3. Spatial proximity to lifeboats if Cabin letter (A-G) is known.
    
    Passenger List:
    {json.dumps(passenger_data_list, indent=2)}
    """
    
    try:
        response = Config.github_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format=TitanicBatchResponse,
            temperature=0.1
        )
        
        # Access the pre-parsed Pydantic object directly
        parsed_data = response.choices[0].message.parsed
        
        if parsed_data:
            # Convert the Pydantic objects back to a list of standard dicts for Pandas compatibility
            return [passenger.model_dump() for passenger in parsed_data.passengers]
        else:
            raise ValueError("Failed to parse response into Pydantic model.")
            
    except Exception as e:
        print(f"Batch failed due to error: {e}. Executing baseline fallback.")
        # Type-safe fallback loop
        return [
            {"PassengerId": int(pid), "ai_survival_probability": 0.5, "estimated_social_tier": "Medium"} 
            for pid in batch_df['PassengerId']
        ]

# -------------------------------------------------------------------------
# Execution Loop (Processing in Chunks of 20)
# -------------------------------------------------------------------------
print("Beginning batch processing of dataset...")
batch_size = 20
all_ai_features = []

# Loop through the combined dataframe in increments of 20
for i in range(0, len(combined_df), batch_size):
    chunk = combined_df.iloc[i : i + batch_size]
    print(f"Processing rows {i} to {min(i + batch_size, len(combined_df))}...")
    
    # Process the batch via the API
    batch_results = process_passenger_batch(chunk)
    all_ai_features.extend(batch_results)
    
    # Optional short sleep step to prevent hitting aggressive rate limits on free/tier-1 API keys
    time.sleep(0.5)

# Convert results array straight into a dataframe
ai_features_df = pd.DataFrame(all_ai_features)

# Ensure ID columns share identical data types before joining
ai_features_df['PassengerId'] = ai_features_df['PassengerId'].astype(int)
combined_df['PassengerId'] = combined_df['PassengerId'].astype(int)

# Merge the new AI feature columns cleanly back into the master dataset
combined_df = pd.merge(combined_df, ai_features_df, on='PassengerId', how='left')

# Impute any missing metrics resulting from schema mismatch anomalies
combined_df['ai_survival_probability'] = combined_df['ai_survival_probability'].fillna(0.5)
combined_df['estimated_social_tier'] = combined_df['estimated_social_tier'].fillna("Medium")

# Rename columns to sync with the downstream ML model preprocessing
combined_df = combined_df.rename(columns={
    'ai_survival_probability': 'AI_Survival_Score',
    'estimated_social_tier': 'AI_Social_Tier'
})
print("AI feature generation complete. Moving to ML classification layers.")
# -------------------------------------------------------------------------
# 3. CLASSICAL ML PREPROCESSING
# -------------------------------------------------------------------------
print("Cleaning structured features...")
# Handle typical tabular missing values cleanly
combined_df['Age'] = combined_df['Age'].fillna(combined_df['Age'].median())
combined_df['Fare'] = combined_df['Fare'].fillna(combined_df['Fare'].median())
combined_df['Embarked'] = combined_df['Embarked'].fillna(combined_df['Embarked'].mode()[0])

# Map/Encode explicit categorical metrics for tree-based execution
le = LabelEncoder()
combined_df['Sex'] = le.fit_transform(combined_df['Sex'])
combined_df['Embarked'] = le.fit_transform(combined_df['Embarked'])
combined_df['AI_Social_Tier'] = le.fit_transform(combined_df['AI_Social_Tier'])

# Split back into definitive Train and Test frames
train_features = combined_df[combined_df['Survived'].notna()].copy()
test_features = combined_df[combined_df['Survived'].isna()].copy()

# Select ultimate feature set combining both paradigms
features = [
    'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 
    'AI_Survival_Score', 'AI_Social_Tier'
]

X = train_features[features]
y = train_features['Survived'].astype(int)
X_submission = test_features[features]

# Operational Validation Split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------------------------------------
# 4. ENSEMBLE STACKING ARCHITECTURE
# -------------------------------------------------------------------------
print("Training Stacked Ensemble (XGBoost + Random Forest -> Meta-Learner)...")

# Base Model 1: Gradient Boosting via XGBoost
xgb_base = XGBClassifier(
    n_estimators=120,
    learning_rate=0.04,
    max_depth=4,
    subsample=0.85,
    colsample_bytree=0.85,
    random_state=42,
    eval_metric='logloss'
)

# Base Model 2: Traditional Bagging via Random Forest
rf_base = RandomForestClassifier(
    n_estimators=150,
    max_depth=5,
    min_samples_split=4,
    random_state=42
)

# Stacking Configuration
base_learners = [
    ('xgb', xgb_base),
    ('rf', rf_base)
]
meta_learner = LogisticRegression()

stacking_classifier = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5, # Internal Stratified K-Fold validation to mitigate overfitting metadata
    n_jobs=-1
)

# Fit the stack on validation training subset
stacking_classifier.fit(X_train, y_train)

# Local evaluation
val_predictions = stacking_classifier.predict(X_val)
local_acc = accuracy_score(y_val, val_predictions)
print(f"\n>>> Local Validation Performance: {local_acc * 100:.2f}% <<<")

# -------------------------------------------------------------------------
# 5. FINAL SUBMISSION GENERATION
# -------------------------------------------------------------------------
print("Executing production training on full dataset...")
# Train model on 100% of the training data before final prediction
stacking_classifier.fit(X, y)

print("Generating predictions for Kaggle submission...")
submission_predictions = stacking_classifier.predict(X_submission)

submission_df = pd.DataFrame({
    "PassengerId": test_features["PassengerId"].astype(int),
    "Survived": submission_predictions
})

submission_df.to_csv("hybrid_ai_ml_submission.csv", index=False)
print("Pipeline complete. Output saved to 'hybrid_ai_ml_submission.csv'")