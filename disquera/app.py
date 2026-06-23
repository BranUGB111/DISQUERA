from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json
import os

app = Flask(__name__)

# ── Cargar y preparar datos ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "encuesta.csv")

df = pd.read_csv(DATA_PATH)

# Renombrar columnas a nombres cortos
df.columns = [
    "timestamp", "nombre", "genero_musical", "atraccion_cancion",
    "descubre_musica", "escucha_local", "necesidad_artistas",
    "participacion_evento", "colaboracion"
]

# Unificar variantes de colaboración
df["colaboracion"] = df["colaboracion"].replace(
    "Sí, me encantaría conocer gente nueva", "¡Sí, me encantaría conocer gente nueva!"
)

# ── Encoders globales para el modelo ────────────────────────────────────────
FEATURES = ["atraccion_cancion", "descubre_musica", "escucha_local",
            "necesidad_artistas", "participacion_evento", "colaboracion"]
TARGET = "genero_musical"

le_map = {}
df_model = df.copy()
for col in FEATURES + [TARGET]:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    le_map[col] = le

X = df_model[FEATURES]
y = df_model[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = round(accuracy_score(y_test, model.predict(X_test)) * 100, 1)

# ── Estadísticas para las gráficas ──────────────────────────────────────────
def get_stats():
    stats = {}
    for col in FEATURES + [TARGET]:
        counts = df[col].value_counts()
        stats[col] = {"labels": counts.index.tolist(), "values": counts.values.tolist()}
    return stats

# ── Importancia de variables ─────────────────────────────────────────────────
def get_feature_importance():
    imp = model.feature_importances_
    names = [col.replace("_", " ").title() for col in FEATURES]
    sorted_pairs = sorted(zip(names, imp), key=lambda x: x[1], reverse=True)
    return {
        "labels": [p[0] for p in sorted_pairs],
        "values": [round(p[1] * 100, 1) for p in sorted_pairs]
    }

# ── Opciones únicas para el formulario ──────────────────────────────────────
def get_form_options():
    opts = {}
    for col in FEATURES:
        opts[col] = sorted(df[col].dropna().unique().tolist())
    return opts

# ── Rutas ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    stats = get_stats()
    importance = get_feature_importance()
    form_opts = get_form_options()
    generos = df[TARGET].value_counts()
    top_genero = generos.index[0]
    top_pct = round(generos.iloc[0] / len(df) * 100, 1)
    return render_template(
        "index.html",
        stats=json.dumps(stats),
        importance=json.dumps(importance),
        form_opts=form_opts,
        accuracy=accuracy,
        top_genero=top_genero,
        top_pct=top_pct,
        total=len(df)
    )

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    try:
        row = []
        for col in FEATURES:
            val = data.get(col, "")
            le = le_map[col]
            if val not in le.classes_:
                return jsonify({"error": f"Valor no reconocido para '{col}': {val}"}), 400
            row.append(le.transform([val])[0])

        pred_encoded = model.predict([row])[0]
        pred_label = le_map[TARGET].inverse_transform([pred_encoded])[0]
        proba = model.predict_proba([row])[0]
        classes = le_map[TARGET].inverse_transform(range(len(proba)))
        proba_dict = {cls: round(float(p) * 100, 1) for cls, p in zip(classes, proba)}

        return jsonify({"prediccion": pred_label, "probabilidades": proba_dict})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
