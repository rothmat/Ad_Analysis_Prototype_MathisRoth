
echo "Starte Setup für virtuelles Environment..."

ENV_NAME="agentic_ai_env"

# 1. Virtuelle Umgebung anlegen
python3 -m venv $ENV_NAME

# 2. Aktivieren
source $ENV_NAME/bin/activate

# 3. Pip aktualisieren
pip install --upgrade pip

# 4. Abhängigkeiten installieren
pip install -r requirements.txt

echo "Umgebung '$ENV_NAME' aktiviert & requirements installiert."
