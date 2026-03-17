# Predicción de Riesgo Crediticio con Machine Learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-96.0%25-2ea44f?style=flat-square)]()
[![Accuracy](https://img.shields.io/badge/Accuracy-89.6%25-185FA5?style=flat-square)]()
[![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)](LICENSE)

Sistema de Machine Learning para predecir si un cliente presenta **alto riesgo crediticio**. El modelo final (Gradient Boosting) alcanza una precisión del **89.6%** con un AUC-ROC de **0.960**, permitiendo identificar clientes de riesgo con alta confiabilidad.

---

## Resultados

| Métrica | Valor |
|---------|-------|
| Accuracy | 89.6% |
| **AUC-ROC** | **96.0%** |
| Precision | 80.6% |
| Recall | 86.9% |
| F1-Score | 83.6% |

---

## Dataset

| Característica | Valor |
|----------------|-------|
| Registros | 5,000 clientes |
| Variables | 13 características |
| Target | Riesgo Alto (1) / Riesgo Bajo (0) |
| Distribución | 69.4% Bajo riesgo / 30.6% Alto riesgo |

### Variables

| Variable | Tipo | Descripción |
|----------|------|-------------|
| `edad` | Numérica | Edad del cliente (18-75 años) |
| `ingreso_anual` | Numérica | Ingreso anual en dólares |
| `deuda_total` | Numérica | Deuda total acumulada |
| `historial_crediticio` | Numérica | Años de historial crediticio |
| `num_prestamos_activos` | Numérica | Préstamos actualmente activos |
| `ratio_deuda_ingreso` | Numérica | Ratio deuda/ingreso (0-1) |
| `num_retrasos_12m` | Numérica | Retrasos de pago en últimos 12 meses |
| `empleo_estable` | Categórica | ¿Tiene empleo estable? (Sí/No) |
| `tipo_vivienda` | Categórica | Propia / Alquiler / Hipoteca |
| `num_tarjetas_credito` | Numérica | Número de tarjetas de crédito |
| `consultas_crediticias_6m` | Numérica | Consultas crediticias en 6 meses |
| `estado_civil` | Categórica | Soltero / Casado / Divorciado |

---

## Análisis Exploratorio

Hallazgos clave del EDA:

- **Ingreso anual**: Los clientes de riesgo alto tienen ingresos significativamente menores (mediana ~$35K vs ~$58K)
- **Ratio deuda/ingreso**: Factor crítico — clientes con ratio >0.5 tienen mayor probabilidad de riesgo alto
- **Retrasos de pago**: Fuerte correlación con riesgo. Clientes con 3+ retrasos tienen >75% de probabilidad de riesgo alto
- **Empleo estable**: Clientes sin empleo estable tienen ~42% de riesgo alto vs ~25% con empleo estable

![EDA](output/eda_riesgo_clientes.png)

---

## Preprocesamiento

| Paso | Descripción |
|------|-------------|
| Encoding | Label encoding para variables binarias, One-hot para categóricas |
| Escalado | StandardScaler para normalizar características numéricas |
| Balanceo | Oversampling de clase minoritaria (de 1,222 a 2,778 muestras) |
| División | 80% Train / 20% Test con estratificación |

---

## Modelos Evaluados

| Modelo | AUC-ROC | CV AUC (5-fold) |
|--------|---------|-----------------|
| **Gradient Boosting** 🥇 | **0.960** | **0.976** |
| Random Forest 🥈 | 0.957 | 0.982 |
| SVM 🥉 | 0.905 | 0.941 |
| Regresión Logística | 0.874 | 0.884 |

Modelo seleccionado: **Gradient Boosting** con `n_estimators=100`, `max_depth=5`, `learning_rate=0.1`.

---

## Evaluación del Modelo

### Matriz de confusión (test: 1,000 muestras)

|  | Predicción Bajo | Predicción Alto |
|--|-----------------|-----------------|
| **Real Bajo** | 630 (TN) | 64 (FP) |
| **Real Alto** | 40 (FN) ⚠️ | 266 (TP) |

Los **40 falsos negativos** son el caso más crítico: clientes de alto riesgo clasificados incorrectamente como bajo riesgo.

![Evaluación del modelo](output/evaluacion_modelo.png)

---

## Variables Más Importantes

| Ranking | Variable | Importancia |
|---------|----------|-------------|
| 1 | `ingreso_anual` | 37.2% |
| 2 | `ratio_deuda_ingreso` | 18.1% |
| 3 | `num_retrasos_12m` | 14.8% |
| 4 | `deuda_total` | 12.3% |
| 5 | `empleo_estable` | 6.5% |

---

## Instalación y uso

```bash
git clone https://github.com/tu-usuario/credit-risk-ml.git
cd credit-risk-ml
pip install -r requirements.txt
python ML_para_predicción_de_riesgo_de_clientes.py
```

Ejecutar el script genera el dataset, entrena los 4 modelos, guarda las visualizaciones y exporta el modelo en `output/`.

### Predecir un cliente nuevo

```python
import pickle

with open('output/modelo_riesgo_gb.pkl', 'rb') as f:
    assets = pickle.load(f)

modelo = assets['modelo']
scaler = assets['scaler']
encoder = assets['label_encoder']

cliente = {
    'edad': 35,
    'ingreso_anual': 60000,
    'deuda_total': 25000,
    'historial_crediticio': 8,
    'num_prestamos_activos': 1,
    'ratio_deuda_ingreso': 0.42,
    'num_retrasos_12m': 0,
    'empleo_estable': 'Si',
    'tipo_vivienda': 'Propia',
    'num_tarjetas_credito': 2,
    'consultas_crediticias_6m': 1,
    'estado_civil': 'Casado'
}

# Ver función predecir_riesgo_cliente() en el script para el preprocesamiento completo
```

### Ejemplos de predicción

| Cliente | Ingreso | Ratio D/I | Retrasos | Resultado |
|---------|---------|-----------|----------|-----------|
| Perfil saludable | $75,000 | 0.27 | 0 | ✅ RIESGO BAJO (0.4%) → APROBAR |
| Perfil de riesgo | $25,000 | 0.85 | 4 | ❌ RIESGO ALTO (99.7%) → RECHAZAR |
| Caso intermedio | $95,000 | 0.55 | 1 | ✅ RIESGO BAJO (10.3%) → APROBAR |

> **Recomendación**: usar umbral 0.5 para decisiones automáticas. Revisar manualmente casos con probabilidad entre 0.3 y 0.7.

---

## Estructura del proyecto

```
credit-risk-ml/
├── ML_para_predicción_de_riesgo_de_clientes.py
├── requirements.txt
├── README.md
└── output/
    ├── modelo_riesgo_gb.pkl
    ├── dataset_riesgo_clientes.csv
    ├── eda_riesgo_clientes.png
    ├── matriz_correlacion.png
    ├── evaluacion_modelo.png
    └── comparacion_modelos.png
```

---

## Conclusiones

1. El modelo alcanza un AUC-ROC de **0.960**, superando significativamente el azar
2. El **ingreso anual** es el predictor más importante (37.2%), seguido del ratio deuda/ingreso
3. El **balanceo de clases** mejoró el recall para detectar clientes de riesgo alto
4. Próximos pasos: datos reales, monitoreo continuo, variables adicionales (sector laboral, educación), despliegue como API

---

*Desarrollado con Python, scikit-learn, pandas y matplotlib.*
