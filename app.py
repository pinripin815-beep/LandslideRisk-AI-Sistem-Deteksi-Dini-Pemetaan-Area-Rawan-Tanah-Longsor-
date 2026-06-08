import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. BIKIN DATA SIMULASI (100 BARIS AGAR RINGAN)
np.random.seed(42)
n_sampel = 100

data = {
    'lereng_curam': np.random.uniform(0, 50, n_sampel),      # Derajat kemiringan
    'curah_hujan': np.random.uniform(100, 400, n_sampel),    # mm / bulan
    'vegetasi_ndvi': np.random.uniform(0, 1, n_sampel)       # 0=gundul, 1=lebat
}
df = pd.DataFrame(data)

# Logika sederhana AI: Longsor jika lereng > 25 derajat DAN hujan > 250 mm
df['target_longsor'] = np.where(
    (df['lereng_curam'] > 25) & (df['curah_hujan'] > 250), 1, 0
)

# 2. BAGI DATA UNTUK TRAINING & TESTING
X = df.drop(columns=['target_longsor'])
y = df['target_longsor']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. LATIH MODEL AI
print("Sedang melatih Landslide AI...")
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 4. CEK AKURASI
prediksi = model.predict(X_test)
akurasi = accuracy_score(y_test, prediksi)

print(f"Pelatihan Selesai! Akurasi Model: {akurasi * 100:.2f}%")
print("\nContoh Prediksi Area Baru:")
area_baru = [[35, 300, 0.2]] # Lereng 35, Hujan 300, Vegetasi gundul
hasil = model.predict(area_baru)
print("Hasil -> Rawan Longsor" if hasil[0] == 1 else "Hasil -> Aman")
