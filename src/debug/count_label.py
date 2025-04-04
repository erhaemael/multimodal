import pandas as pd

# Path ke file CSV hasil preprocessing
csv_path = "data/preprocessed/WESAD.csv"

# Load dataset
df = pd.read_csv(csv_path)

# print(df.isna().sum())  # Cek jumlah NaN di setiap kolom
# print(df.describe())  # Statistik ringkasan untuk melihat distribusi data

# Hitung jumlah total label 0 dan 1
total_counts = df["Label"].value_counts()
print("Total distribusi label di seluruh dataset:")
print(total_counts)

# Jika ada kolom "Patient", hitung distribusi label per pasien
if "Patient" in df.columns:
    print("\nDistribusi label per pasien:")
    per_patient_counts = df.groupby("Patient")["Label"].value_counts().unstack(fill_value=0)
    print(per_patient_counts)
else:
    print("\nKolom 'Patient' tidak ditemukan dalam dataset.")
