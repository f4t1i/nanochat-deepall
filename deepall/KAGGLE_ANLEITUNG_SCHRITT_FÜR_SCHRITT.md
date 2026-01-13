# 🚀 KAGGLE - SCHRITT FÜR SCHRITT ANLEITUNG

## ⚠️ WICHTIG - LESE DAS ZUERST!

Du musst **GENAU** diese Schritte befolgen:

1. **Öffne Kaggle Notebook**
2. **Erstelle NEUE Zelle** (nicht in bestehende Zelle schreiben!)
3. **Kopiere Code KOMPLETT** (nicht vermischen!)
4. **Führe aus** (Shift + Enter)
5. **Warte bis fertig** (grüner Haken)
6. **Dann nächste Zelle**

---

## 📌 SCHRITT 1: Neue Zelle erstellen

1. Gehe zu: https://www.kaggle.com/code
2. Klicke: **"New Notebook"**
3. Wähle: **GPU** (rechts oben)
4. Klicke: **"+ Code"** (neue Zelle)

---

## 📌 SCHRITT 2: Zelle 1 - Repository laden

**KOPIERE DIESEN CODE KOMPLETT:**

```python
import os
import shutil

repo_path = "/kaggle/working/nanochat-deepall"
if os.path.exists(repo_path):
    print("🗑️  Lösche altes Repository...")
    shutil.rmtree(repo_path)

print("📥 Clone Repository...")
os.system("cd /kaggle/working && git clone https://github.com/f4t1i/nanochat-deepall.git")

print("📦 Installiere Dependencies...")
os.system("pip install -q torch transformers datasets tqdm numpy pandas scikit-learn")

if os.path.exists(f"{repo_path}/deepall"):
    print("✅ Repository geladen!")
else:
    print("❌ Repository Problem!")
    exit()

deepall_path = f"{repo_path}/deepall"
training_file = f"{deepall_path}/training_data.txt"
if os.path.exists(training_file):
    with open(training_file, 'r') as f:
        lines = f.readlines()
    print(f"✅ Training Data vorhanden ({len(lines)} Zeilen)")

train_script = f"{deepall_path}/kaggle_train.py"
if os.path.exists(train_script):
    print(f"✅ kaggle_train.py vorhanden")

print("\n✅ Setup fertig!")
```

**Drücke: Shift + Enter**

**Ergebnis sollte sein:**
```
✅ Repository geladen!
✅ Training Data vorhanden (XXX Zeilen)
✅ kaggle_train.py vorhanden
✅ Setup fertig!
```

---

## 📌 SCHRITT 3: Zelle 2 - Training starten

**Klicke: "+ Code"** (neue Zelle)

**KOPIERE DIESEN CODE KOMPLETT:**

```python
import os
import sys

repo_path = "/kaggle/working/nanochat-deepall"
sys.path.insert(0, repo_path)
os.chdir(f"{repo_path}/deepall")

print("🚀 Starte Training...")
print("=" * 60)

os.system("python kaggle_train.py")
```

**Drücke: Shift + Enter**

**Ergebnis:** Training läuft (30 Min)

---

## 📌 SCHRITT 4: Zelle 3 - Ergebnisse prüfen

**Klicke: "+ Code"** (neue Zelle)

**KOPIERE DIESEN CODE KOMPLETT:**

```python
import os

output_path = "/kaggle/working/nanochat-deepall/deepall"
print("📁 Output Dateien:")

for f in os.listdir(output_path):
    if f.endswith(('.pt', '.pth')):
        full_path = os.path.join(output_path, f)
        size = os.path.getsize(full_path) / (1024*1024)
        print(f"  ✅ {f} ({size:.0f} MB)")

print("\n✅ Training fertig!")
```

**Drücke: Shift + Enter**

**Ergebnis:** Alle `.pt` Dateien werden angezeigt

---

## 💾 SCHRITT 5: Download

1. Rechts: **"Output"** Tab
2. Wähle die `.pt` Dateien
3. Klicke: **"Download"**

---

## ✅ FERTIG!

🎉 Dein Modell ist trainiert und heruntergeladen!

---

## 🆘 WENN FEHLER:

### Fehler: "SyntaxError"
→ Du hast Code vermischt!
→ Lösche die Zelle und kopiere nochmal KOMPLETT

### Fehler: "Repository Problem"
→ Zelle 1 nochmal ausführen

### Fehler: "Module not found"
→ Warte bis Zelle 1 fertig ist, dann Zelle 2

---

## ✅ WICHTIGSTE REGELN:

```
1. JEDE Zelle EINZELN
2. NICHT vermischen
3. KOMPLETT kopieren
4. Warten bis fertig
5. Dann nächste Zelle
```

**Wenn du diese Regeln befolgst, funktioniert es garantiert!** 🎉

