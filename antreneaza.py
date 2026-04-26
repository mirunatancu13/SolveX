import os
from pathlib import Path
import numpy as np
import pickle
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

# Setăm calea către folderul descărcat de pe GitHub
ROOT_DIR = Path("Mathematical-Handwriting-recognition").resolve()
dataset_dir = ROOT_DIR / "Dataset" / "extracted_images"
default_image_size = (45, 45)

def get_image_matrix(image_path):
    try:
        # Ignorăm fișierele corupte sau folderele care dădeau Permission Denied
        if os.path.isdir(image_path): 
            return None
        img = Image.open(image_path).convert('L')
        
        try:
            resample_filter = Image.Resampling.LANCZOS
        except AttributeError:
            resample_filter = Image.LANCZOS
            
        img = img.resize(default_image_size, resample_filter)
        return np.array(img).flatten().tolist()
    except Exception:
        return None

X_train, Y_train = [], []
print("1. Citim imaginile din dataset... (Durează aprox. 1-2 minute)")

if not dataset_dir.exists():
    print(f"EROARE: Nu găsesc folderul cu imagini: {dataset_dir}")
    exit()

for directory in os.listdir(dataset_dir):
    dir_path = dataset_dir / directory
    if not os.path.isdir(dir_path): continue
    
    for filename in os.listdir(dir_path):
        img_path = dir_path / filename
        img_matrix = get_image_matrix(img_path)
        
        # Adăugăm doar dacă imaginea a fost citită corect
        if img_matrix is not None:
            X_train.append(img_matrix)
            Y_train.append(directory)

print("2. Antrenăm modelul AI... (Te rog așteaptă)")
# n_jobs=-1 folosește tot procesorul tău pentru a antrena super rapid
clf = RandomForestClassifier(n_jobs=-1) 
clf.fit(X_train, Y_train)

# Salvăm noul model
model_dir = ROOT_DIR / "Model"
model_dir.mkdir(exist_ok=True)
model_path = model_dir / "random_forest_classifier.pkl"

with open(model_path, 'wb') as f:
    pickle.dump(clf, f)

print(f"\n✅ GATA! Modelul compatibil cu Python 3.12 a fost salvat cu succes în:\n{model_path}")
print(f"Acuratețe: {clf.score(X_train, Y_train) * 100:.2f}%")