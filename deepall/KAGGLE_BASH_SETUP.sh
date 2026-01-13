#!/bin/bash

# ============================================================================
# KAGGLE NOTEBOOK SETUP - Repo laden und vorbereiten
# ============================================================================
# Kopiere diesen Code in die ERSTE ZELLE deines Kaggle Notebooks!
# ============================================================================

echo "🚀 Starte Kaggle Setup..."
echo ""

# 1. Installiere Git (falls nicht vorhanden)
echo "📦 Installiere Git..."
apt-get update -qq && apt-get install -y git > /dev/null 2>&1

# 2. Clone das Repository
echo "📥 Clone nanochat-deepall Repository..."
cd /kaggle/working
git clone https://github.com/f4t1i/nanochat-deepall.git
cd nanochat-deepall

# 3. Installiere Dependencies
echo "📚 Installiere Python Dependencies..."
pip install -q torch transformers datasets tqdm numpy pandas scikit-learn

# 4. Prüfe ob Dataset vorhanden ist
echo ""
echo "🔍 Prüfe Dataset..."
if [ -d "/kaggle/input" ]; then
    echo "✅ Input Verzeichnis gefunden"
    ls -la /kaggle/input/
else
    echo "⚠️  Kein Input Verzeichnis - Dataset muss hinzugefügt werden!"
fi

# 5. Prüfe ob Dateien vorhanden sind
echo ""
echo "📂 Prüfe Dateien..."
if [ -f "deepall/kaggle_train.py" ]; then
    echo "✅ kaggle_train.py gefunden"
else
    echo "❌ kaggle_train.py NICHT gefunden!"
fi

if [ -f "deepall/training_data.txt" ]; then
    echo "✅ training_data.txt gefunden"
else
    echo "⚠️  training_data.txt nicht gefunden"
fi

# 6. Zeige Struktur
echo ""
echo "📁 Repository Struktur:"
ls -la deepall/ | head -20

echo ""
echo "✅ Setup fertig!"
echo ""
echo "🎯 Nächste Schritte:"
echo "1. Prüfe ob dein Dataset als INPUT hinzugefügt ist"
echo "2. Führe die nächste Zelle aus: python deepall/kaggle_train.py"
echo ""

