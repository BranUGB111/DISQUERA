from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json, os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "encuesta.csv")

df = pd.read_csv(DATA_PATH)
df.columns = [
    "timestamp", "nombre", "genero_musical", "atraccion_cancion",
    "descubre_musica", "escucha_local", "necesidad_artistas",
    "participacion_evento", "colaboracion"
]
df["colaboracion"] = df["colaboracion"].replace(
    "Sí, me encantaría conocer gente nueva", "¡Sí, me encantaría conocer gente nueva!"
)

# ── Modelo: predice género musical ──────────────────────────────────────────
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

# ── Puntuación de candidato a la industria ──────────────────────────────────
# Reglas basadas en las respuestas de la encuesta (puntaje 0-100)
INDUSTRY_RULES = {
    "participacion_evento": {
        "Como artista": 40,
        "Ayudando en la organización o difusión.": 25,
        "Como espectador": 5
    },
    "colaboracion": {
        "¡Sí, me encantaría conocer gente nueva!": 30,
        "Tal vez, dependiendo del estilo musical.": 15,
        "Prefiero trabajar solo en mis proyectos por ahora.": 5
    },
    "escucha_local": {
        "Todo el tiempo! Siempre busco talento local": 20,
        "De vez en cuando, si me sale algo en redes.": 10,
        "Muy poco, no conozco a muchos.": 3
    },
    "necesidad_artistas": {
        "Mejores estudios de grabación y producción.": 10,
        "Asesoría en marketing y redes sociales.": 8,
        "Más espacios para tocar en vivo.": 7,
        "Apoyo de la gente compartiendo su música.": 6
    }
}

def industry_score(answers: dict) -> dict:
    """Calcula puntuación (0-100) de candidato a la industria."""
    score = 0
    breakdown = {}
    for col, rule in INDUSTRY_RULES.items():
        val = answers.get(col, "")
        pts = rule.get(val, 0)
        score += pts
        breakdown[col] = pts
    score = min(score, 100)

    if score >= 75:
        level = "Alto potencial"
        color = "#22c55e"
        desc  = "Este perfil muestra una fuerte inclinación hacia la industria musical. Sería un excelente candidato para colaborar con la disquera."
    elif score >= 45:
        level = "Potencial medio"
        color = "#f59e0b"
        desc  = "Existe interés genuino pero aún en desarrollo. Con la orientación correcta podría convertirse en un colaborador valioso."
    else:
        level = "Perfil oyente"
        color = "#7C3AED"
        desc  = "Este perfil corresponde principalmente a un consumidor de música. Puede ser parte importante del público objetivo de la disquera."

    return {"score": score, "level": level, "color": color, "desc": desc, "breakdown": breakdown}

# ── Estadísticas para gráficas ───────────────────────────────────────────────
def get_stats():
    stats = {}
    for col in FEATURES + [TARGET]:
        counts = df[col].value_counts()
        stats[col] = {"labels": counts.index.tolist(), "values": counts.values.tolist()}
    return stats

def get_feature_importance():
    imp = model.feature_importances_
    names = [col.replace("_", " ").title() for col in FEATURES]
    pairs = sorted(zip(names, imp), key=lambda x: x[1], reverse=True)
    return {"labels": [p[0] for p in pairs], "values": [round(p[1]*100,1) for p in pairs]}

def get_form_options():
    return {col: sorted(df[col].dropna().unique().tolist()) for col in FEATURES}

# ── Rutas ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    generos = df[TARGET].value_counts()
    # Potencial industria del dataset completo
    artistas_pct = round(
        df[df["participacion_evento"] == "Como artista"].shape[0] / len(df) * 100, 1)
    return render_template(
        "index.html",
        stats=json.dumps(get_stats()),
        importance=json.dumps(get_feature_importance()),
        form_opts=get_form_options(),
        accuracy=accuracy,
        top_genero=generos.index[0],
        top_pct=round(generos.iloc[0]/len(df)*100,1),
        total=len(df),
        artistas_pct=artistas_pct
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
                return jsonify({"error": f"Valor no reconocido: '{val}'"}), 400
            row.append(le.transform([val])[0])

        pred_encoded = model.predict([row])[0]
        pred_label   = le_map[TARGET].inverse_transform([pred_encoded])[0]
        proba        = model.predict_proba([row])[0]
        classes      = le_map[TARGET].inverse_transform(range(len(proba)))
        proba_dict   = {cls: round(float(p)*100,1) for cls, p in zip(classes, proba)}

        ind = industry_score(data)

        return jsonify({
            "prediccion":     pred_label,
            "probabilidades": proba_dict,
            "industria":      ind
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
