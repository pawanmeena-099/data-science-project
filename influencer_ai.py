import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def train_influencer_model():
    print("Generating realistic influencer data...")
    
    np.random.seed(42)
    n_samples = 500
    
    data = {
        'Followers': np.random.randint(10000, 1000000, n_samples),
        'Past_Avg_Likes': np.random.randint(1000, 50000, n_samples),
        'Reels_Posted_Per_Week': np.random.randint(1, 10, n_samples),
        'Sponsored_Posts_Ratio': np.random.uniform(0.1, 0.8, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    df['Expected_Engagement'] = (df['Past_Avg_Likes'] * 1.1) + (df['Followers'] * 0.05) + (df['Reels_Posted_Per_Week'] * 500)
    
    features = ['Followers', 'Past_Avg_Likes', 'Reels_Posted_Per_Week', 'Sponsored_Posts_Ratio']
    X = df[features]
    y = df['Expected_Engagement']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training the AI to understand engagement patterns...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Checking Model Accuracy
    predictions = model.predict(X_test)
    error = mean_absolute_error(y_test, predictions)
    print(f"Model Error Margin: +/- {int(error)} likes")
    print("-" * 40)
    
    new_influencer = pd.DataFrame({
        'Followers': [250000],
        'Past_Avg_Likes': [15000],
        'Reels_Posted_Per_Week': [4],
        'Sponsored_Posts_Ratio': [0.3]
    })
    
    predicted_engagement = model.predict(new_influencer)
    
    print("🎯 PREDICTION FOR NEW INFLUENCER:")
    print(f"Followers: 2.5 Lakhs | Past Avg Likes: 15K")
    print(f"Predicted Engagement for next post: {int(predicted_engagement[0])} likes/interactions")

if __name__ == "__main__":
    train_influencer_model()