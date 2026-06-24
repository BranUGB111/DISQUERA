# 🎵 Disquera · Análisis de Gustos Musicales

Aplicación web Flask con Random Forest para predecir géneros musicales a partir de encuestas.

## Estructura del proyecto

```
disquera-app/
├── app.py              ← Servidor Flask + modelo Random Forest
├── requirements.txt    ← Dependencias Python
├── render.yaml         ← Config de despliegue en Render
├── templates/
│   └── index.html      ← Página web (charts + predictor)
├── static/
│   └── style.css       ← Estilos
└── data/
    └── encuesta.csv    ← Dataset de la encuesta
```

## Correr localmente

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Ejecutar servidor
python app.py

# 3. Abrir en el navegador
http://localhost:5000
```

## Desplegar en Render

1. Crear cuenta en https://render.com
2. Conectar tu repositorio de GitHub
3. New > Web Service > seleccionar el repositorio
4. Render detecta `render.yaml` automáticamente
5. Click en **Deploy** — listo en ~2 minutos

## Modelo

- **Algoritmo**: Random Forest Classifier (100 árboles)
- **Variable objetivo**: Género musical favorito
- **Variables predictoras**: 6 preguntas de la encuesta
- **División**: 80% entrenamiento / 20% prueba
