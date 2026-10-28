"""
app.py - Servicio web (API REST + página) del modelo MLP
Taller MLP: predicción de aprobación de estudiantes

Endpoints:
  GET  /            Página web con formulario para ingresar las variables
  POST /predict     Recibe JSON con las 6 variables y devuelve la predicción
  POST /train       Reentrena el modelo y devuelve sus métricas
  GET  /metrics     Métricas del modelo actual (matriz de confusión, reporte)
  GET  /health      Estado del servicio

Ejecutar local:  python app.py        (http://127.0.0.1:5000)
Producción:      gunicorn app:app --timeout 300
"""
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template_string, request

from train import FEATURES, RUTA_MODELO, entrenar

if not os.path.exists(RUTA_MODELO):
    print("No se encontró el modelo; entrenando...")
    entrenar()

artefactos = joblib.load(RUTA_MODELO)
app = Flask(__name__)


def predecir(datos: dict) -> dict:
    faltan = [f for f in FEATURES if f not in datos]
    if faltan:
        raise ValueError(f"Faltan variables: {faltan}")
    fila = pd.DataFrame([{f: datos[f] for f in FEATURES}]).apply(pd.to_numeric, errors="coerce")
    fila = fila.fillna(artefactos["medianas"])           # vacíos -> mediana
    X = artefactos["scaler"].transform(fila)
    modelo = artefactos["modelo"]
    clase = int(modelo.predict(X)[0])
    prob = float(modelo.predict_proba(X)[0][1])
    return {"prediccion": "Aprobado" if clase == 1 else "Reprobado",
            "clase": clase,
            "probabilidad_aprobar": round(prob, 4)}


@app.get("/health")
def health():
    return jsonify(status="ok", modelo=artefactos["metricas"]["mejores_hiperparametros"])


@app.get("/metrics")
def metrics():
    return jsonify(artefactos["metricas"])


@app.post("/predict")
def predict():
    datos = request.get_json(silent=True)
    if not isinstance(datos, dict):
        return jsonify(error="Envía un JSON con las variables: " + ", ".join(FEATURES)), 400
    try:
        return jsonify(predecir(datos))
    except ValueError as e:
        return jsonify(error=str(e)), 400


@app.post("/train")
def train():
    global artefactos
    metricas = entrenar(rapido=True)
    artefactos = joblib.load(RUTA_MODELO)
    return jsonify(mensaje="Modelo reentrenado", metricas=metricas)


PAGINA = """
<!doctype html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Predicción de aprobación – MLP</title>
<style>
 body{font-family:system-ui,Segoe UI,Arial,sans-serif;background:#f4f6fb;margin:0;color:#1d2433}
 main{max-width:860px;margin:auto;padding:24px 16px}
 h1{margin:.2em 0}.sub{color:#5b6475;margin-top:0}
 .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px}
 .card{background:#fff;border-radius:12px;padding:20px;box-shadow:0 1px 4px #0001}
 label{display:block;font-size:14px;font-weight:600;margin:12px 0 4px}
 input,select{width:100%;padding:8px;border:1px solid #cfd5e1;border-radius:8px;font-size:15px;box-sizing:border-box}
 button{margin-top:16px;padding:10px 18px;border:0;border-radius:8px;background:#3355dd;color:#fff;font-size:15px;cursor:pointer}
 button.sec{background:#6b7280}
 .res{font-size:28px;font-weight:700;margin:8px 0}.ok{color:#15803d}.no{color:#b91c1c}
 pre{background:#0f172a;color:#e2e8f0;padding:12px;border-radius:8px;overflow:auto;font-size:13px}
 table{border-collapse:collapse}td,th{border:1px solid #cfd5e1;padding:6px 12px;text-align:center}
</style></head><body><main>
<h1>🎓 ¿El estudiante aprobará?</h1>
<p class="sub">Perceptrón Multicapa (MLP) · accuracy en prueba: <b>{{ acc }}</b> · capas {{ capas }}</p>
<div class="grid">
 <form class="card" id="f">
  <label>Tareas (0-100)</label><input name="Tareas" type="number" step="any" min="0" max="100" value="65" required>
  <label>Exámenes (0-100)</label><input name="Examenes" type="number" step="any" min="0" max="100" value="60" required>
  <label>Asistencia (%)</label><input name="Asistencia" type="number" step="any" min="0" max="100" value="70" required>
  <label>Horas de estudio por semana</label><input name="Estudio_Horas" type="number" step="any" min="0" value="13.5" required>
  <label>Horas de sueño por día</label><input name="Sueno_Horas" type="number" step="any" min="0" max="24" value="6.5" required>
  <label>Actividades extracurriculares</label>
  <select name="Extracurriculares"><option value="0">No</option><option value="1">Sí</option></select>
  <button>Predecir</button>
 </form>
 <div class="card">
  <h3>Resultado</h3>
  <div id="res" class="res">—</div><div id="prob"></div>
  <h3>Matriz de confusión (prueba)</h3>
  <table><tr><th></th><th>Pred. Reprobado</th><th>Pred. Aprobado</th></tr>
   <tr><th>Real Reprobado</th><td>{{ cm[0][0] }}</td><td>{{ cm[0][1] }}</td></tr>
   <tr><th>Real Aprobado</th><td>{{ cm[1][0] }}</td><td>{{ cm[1][1] }}</td></tr></table>
  <button class="sec" id="tr" type="button">Reentrenar modelo</button>
  <div id="trmsg"></div>
 </div>
</div>
<div class="card" style="margin-top:16px">
 <h3>Uso del endpoint</h3>
 <pre>POST {{ base }}predict
Content-Type: application/json

{"Tareas": 90, "Examenes": 85, "Asistencia": 95,
 "Estudio_Horas": 20, "Sueno_Horas": 7.5, "Extracurriculares": 1}</pre>
 <p>Otros: <code>GET /metrics</code> · <code>POST /train</code> · <code>GET /health</code></p>
</div>
<script>
document.getElementById('f').onsubmit = async e => {
  e.preventDefault();
  const datos = Object.fromEntries(new FormData(e.target));
  for (const k in datos) datos[k] = Number(datos[k]);
  const r = await (await fetch('predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(datos)})).json();
  const el = document.getElementById('res');
  if (r.error) { el.textContent = r.error; el.className = 'res no'; return; }
  el.textContent = r.prediccion; el.className = 'res ' + (r.clase ? 'ok' : 'no');
  document.getElementById('prob').textContent = 'Probabilidad de aprobar: ' + (r.probabilidad_aprobar*100).toFixed(1) + '%';
};
document.getElementById('tr').onclick = async () => {
  const m = document.getElementById('trmsg'); m.textContent = 'Entrenando...';
  const r = await (await fetch('train', {method:'POST'})).json();
  m.textContent = 'Listo. Accuracy en prueba: ' + r.metricas.accuracy_prueba;
};
</script></main></body></html>
"""


@app.get("/")
def index():
    m = artefactos["metricas"]
    return render_template_string(
        PAGINA, acc=m["accuracy_prueba"], cm=m["matriz_confusion"],
        capas=m["mejores_hiperparametros"].get("hidden_layer_sizes"), base=request.host_url)


if __name__ == "__main__":
    app.run(debug=True)
