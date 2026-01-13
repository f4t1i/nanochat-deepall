#!/bin/bash

# Push alle wichtigen Dateien zu GitHub

cd /home/deepall/nanochat

echo "🚀 Pushe alle Dateien zu GitHub..."
echo ""

# Füge alle Dateien hinzu
git add -A

# Commit
git commit -m "feat: Gesamter nanochat-deepall Ordner - Alle Dateien und Dokumentation

- Komplettes DeepMaster Fine-Tuning Setup
- Alle Trainingsdaten und Modelle
- Kaggle Training Scripts und Dokumentation
- Vollständige Dokumentation und Anleitungen
- Alle Konfigurationen und Skripte
- Ready für Production

Status: ✅ Alles gepusht zu GitHub"

# Push
echo ""
echo "📤 Pushe zu GitHub..."
git push origin master

echo ""
echo "✅ Fertig!"
git log --oneline -3

