# 🔍 KAGGLE - DEBUG Repository Clone Problem

## Problem:
```
❌ Fehler beim Laden!
```

Das Repository wird geclont, aber die Dateien sind nicht vorhanden.

---

## 📌 ZELLE 1: Debug - Prüfe was geclont wurde

```python
import os
import subprocess

repo_path = "/kaggle/working/nanochat-deepall"

# Lösche altes Repository
if os.path.exists(repo_path):
    import shutil
    shutil.rmtree(repo_path)

# Clone mit Output
print("📥 Clone Repository...")
result = subprocess.run(
    ["git", "clone", "https://github.com/f4t1i/nanochat-deepall.git"],
    cwd="/kaggle/working",
    capture_output=True,
    text=True
)

print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)
print("Return Code:", result.returncode)

# Prüfe was existiert
print("\n📂 Was existiert:")
if os.path.exists(repo_path):
    print(f"✅ Repository Ordner existiert")
    print(f"   Inhalt: {os.listdir(repo_path)}")
    
    # Prüfe deepall Ordner
    deepall_path = os.path.join(repo_path, "deepall")
    if os.path.exists(deepall_path):
        print(f"✅ deepall Ordner existiert")
        print(f"   Dateien: {os.listdir(deepall_path)[:10]}")
    else:
        print(f"❌ deepall Ordner NICHT vorhanden!")
else:
    print(f"❌ Repository Ordner NICHT vorhanden!")

# Installiere Dependencies
print("\n📦 Installiere Dependencies...")
os.system("pip install -q torch transformers datasets tqdm numpy pandas scikit-learn")
print("✅ Dependencies installiert")
```

---

## 📌 ZELLE 2: Wenn Debug erfolgreich - Training starten

```python
import os
import sys

repo_path = "/kaggle/working/nanochat-deepall"

# Prüfe nochmal
if os.path.exists(f"{repo_path}/deepall"):
    print("✅ Repository OK!")
    
    sys.path.insert(0, repo_path)
    os.chdir(f"{repo_path}/deepall")
    
    print("🚀 Starte Training...")
    os.system("python kaggle_train.py")
else:
    print("❌ Repository Problem - Prüfe Zelle 1!")
```

---

## 🆘 Wenn immer noch Fehler:

Versuche diesen alternativen Weg:

```python
# Alternative: Direkter Download
import os
import urllib.request
import zipfile

print("📥 Download Repository als ZIP...")
url = "https://github.com/f4t1i/nanochat-deepall/archive/refs/heads/master.zip"
zip_path = "/kaggle/working/repo.zip"

urllib.request.urlretrieve(url, zip_path)
print("✅ ZIP heruntergeladen")

# Entpacke
print("📦 Entpacke...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall("/kaggle/working")

# Rename
os.rename("/kaggle/working/nanochat-deepall-master", "/kaggle/working/nanochat-deepall")
print("✅ Entpackt und umbenannt")

# Prüfe
if os.path.exists("/kaggle/working/nanochat-deepall/deepall"):
    print("✅ Repository erfolgreich!")
else:
    print("❌ Immer noch Fehler!")
```

---

## ✅ Wenn das funktioniert:

Dann Training starten mit Zelle 2!

