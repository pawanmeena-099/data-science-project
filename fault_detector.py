import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_smart_grid_model():
    print("🔌 Generating Transmission Line Sensor Data...\n")
    
    np.random.seed(42)
    n_samples = 1000

    normal_voltage = np.random.normal(220, 5, int(n_samples * 0.8))
    normal_current = np.random.normal(10, 2, int(n_samples * 0.8))
    normal_status = ['Normal'] * int(n_samples * 0.8)
    
    fault_voltage = np.random.normal(150, 30, int(n_samples * 0.2))
    fault_current = np.random.normal(50, 15, int(n_samples * 0.2))
    fault_status = ['Fault Detected'] * int(n_samples * 0.2)

    voltages = np.concatenate([normal_voltage, fault_voltage])
    currents = np.concatenate([normal_current, fault_current])
    statuses = normal_status + fault_status
    
    df = pd.DataFrame({'Voltage_V': voltages, 'Current_A': currents, 'System_Status': statuses})
    
    df = df.sample(frac=1).reset_index(drop=True)

    X = df[['Voltage_V', 'Current_A']] 
    y = df['System_Status'] 
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("🧠 Training AI to recognize power anomalies...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"✅ AI Accuracy in detecting faults: {accuracy * 100:.2f}%\n")
    print("-" * 50)
    
    print("📡 TESTING LIVE SENSOR DATA:")
    
    test_1 = pd.DataFrame({'Voltage_V': [218], 'Current_A': [11]})
    print(f"Reading 1 (218V, 11A) -> AI Says: {model.predict(test_1)[0]}")
    
    test_2 = pd.DataFrame({'Voltage_V': [110], 'Current_A': [65]})
    print(f"Reading 2 (110V, 65A) -> AI Says: {model.predict(test_2)[0]}")

if __name__ == "__main__":
    train_smart_grid_model()