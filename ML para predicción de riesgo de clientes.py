import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils import resample
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo
plt.style.use('ggplot')
sns.set_palette("husl")

# Crear carpeta de output
os.makedirs('output', exist_ok=True)

# ============================================================
# BLOQUE 1 - CREAR DATASET
# ============================================================
np.random.seed(42)
n_samples = 5000

data = {
    'edad': np.random.randint(18, 75, n_samples),
    'ingreso_anual': np.random.normal(50000, 25000, n_samples).clip(15000, 200000),
    'deuda_total': np.random.normal(30000, 20000, n_samples).clip(0, 150000),
    'historial_crediticio': np.random.randint(0, 30, n_samples),
    'num_prestamos_activos': np.random.poisson(2, n_samples),
    'ratio_deuda_ingreso': np.random.normal(0.4, 0.2, n_samples).clip(0, 1),
    'num_retrasos_12m': np.random.poisson(1, n_samples),
    'empleo_estable': np.random.choice(['Si', 'No'], n_samples, p=[0.7, 0.3]),
    'tipo_vivienda': np.random.choice(['Propia', 'Alquiler', 'Hipoteca'], n_samples, p=[0.4, 0.35, 0.25]),
    'num_tarjetas_credito': np.random.poisson(3, n_samples),
    'consultas_crediticias_6m': np.random.poisson(2, n_samples),
    'estado_civil': np.random.choice(['Soltero', 'Casado', 'Divorciado'], n_samples, p=[0.4, 0.5, 0.1]),
}

df = pd.DataFrame(data)

def calcular_riesgo(row):
    riesgo = 0
    if row['ratio_deuda_ingreso'] > 0.5:
        riesgo += 2
    if row['num_retrasos_12m'] > 2:
        riesgo += 3
    if row['ingreso_anual'] < 30000:
        riesgo += 2
    if row['historial_crediticio'] < 2:
        riesgo += 2
    if row['empleo_estable'] == 'No':
        riesgo += 1
    if row['consultas_crediticias_6m'] > 4:
        riesgo += 1
    if row['deuda_total'] > row['ingreso_anual'] * 0.8:
        riesgo += 2
    riesgo += np.random.randint(-1, 2)
    return 1 if riesgo >= 4 else 0

df['riesgo_alto'] = df.apply(calcular_riesgo, axis=1)

print("=" * 60)
print("DATASET DE RIESGO CREDITICIO - CLIENTES")
print("=" * 60)
print(f"Total de registros: {len(df):,}")
print(f"Variables: {df.shape[1]}")
print(f"\nDistribución de Riesgo:")
print(df['riesgo_alto'].value_counts())
print(f"\nPorcentaje de riesgo alto: {df['riesgo_alto'].mean()*100:.1f}%")
print("\nPrimeras filas:")
print(df.head(10))

# ============================================================
# BLOQUE 2 - EDA
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('ANÁLISIS EXPLORATORIO - DISTRIBUCIÓN DE VARIABLES', fontsize=14, fontweight='bold', y=1.02)

ax1 = axes[0, 0]
riesgo_counts = df['riesgo_alto'].value_counts()
colors = ['#2ecc71', '#e74c3c']
labels = ['Riesgo Bajo', 'Riesgo Alto']
ax1.pie(riesgo_counts.values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90, explode=(0, 0.05))
ax1.set_title('Distribución de Riesgo', fontweight='bold')

ax2 = axes[0, 1]
df.boxplot(column='ingreso_anual', by='riesgo_alto', ax=ax2)
ax2.set_title('Ingreso Anual por Riesgo', fontweight='bold')
ax2.set_xlabel('Riesgo (0=Bajo, 1=Alto)')
ax2.set_ylabel('Ingreso Anual ($)')

ax3 = axes[0, 2]
df.boxplot(column='ratio_deuda_ingreso', by='riesgo_alto', ax=ax3)
ax3.set_title('Ratio Deuda/Ingreso por Riesgo', fontweight='bold')
ax3.set_xlabel('Riesgo (0=Bajo, 1=Alto)')
ax3.set_ylabel('Ratio Deuda/Ingreso')

ax4 = axes[1, 0]
for riesgo, color in zip([0, 1], ['#2ecc71', '#e74c3c']):
    subset = df[df['riesgo_alto'] == riesgo]
    ax4.hist(subset['edad'], bins=20, alpha=0.6, label=f'Riesgo {riesgo}', color=color)
ax4.set_title('Distribución de Edad por Riesgo', fontweight='bold')
ax4.set_xlabel('Edad')
ax4.set_ylabel('Frecuencia')
ax4.legend()

ax5 = axes[1, 1]
ct = pd.crosstab(df['empleo_estable'], df['riesgo_alto'], normalize='index') * 100
ct.plot(kind='bar', ax=ax5, color=['#2ecc71', '#e74c3c'])
ax5.set_title('Empleo Estable vs Riesgo (%)', fontweight='bold')
ax5.set_xlabel('Empleo Estable')
ax5.set_ylabel('Porcentaje')
ax5.legend(['Riesgo Bajo', 'Riesgo Alto'])
ax5.tick_params(axis='x', rotation=0)

ax6 = axes[1, 2]
retraso_riesgo = df.groupby('num_retrasos_12m')['riesgo_alto'].mean() * 100
ax6.bar(retraso_riesgo.index, retraso_riesgo.values, color='#3498db', edgecolor='black')
ax6.set_title('Retrasos (12m) vs % Riesgo Alto', fontweight='bold')
ax6.set_xlabel('Número de Retrasos')
ax6.set_ylabel('% Riesgo Alto')

plt.tight_layout()
plt.savefig('output/eda_riesgo_clientes.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("✓ Visualización EDA guardada")

# ============================================================
# BLOQUE 3 - MATRIZ DE CORRELACIÓN
# ============================================================
fig, ax = plt.subplots(figsize=(12, 10))
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('MATRIZ DE CORRELACIÓN - VARIABLES NUMÉRICAS', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('output/matriz_correlacion.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("✓ Matriz de correlación guardada")

# ============================================================
# BLOQUE 4 - PREPROCESAMIENTO
# ============================================================
print("=" * 60)
print("PREPROCESAMIENTO DE DATOS")
print("=" * 60)

df_processed = df.copy()

print("\n1. Encoding de variables categóricas:")
le_empleo = LabelEncoder()
df_processed['empleo_estable'] = le_empleo.fit_transform(df_processed['empleo_estable'])
print(f"   - empleo_estable: {dict(zip(le_empleo.classes_, le_empleo.transform(le_empleo.classes_)))}")

df_processed = pd.get_dummies(df_processed, columns=['tipo_vivienda', 'estado_civil'], drop_first=True)
print(f"   - Variables dummy creadas: {[c for c in df_processed.columns if 'tipo_vivienda' in c or 'estado_civil' in c]}")

X = df_processed.drop('riesgo_alto', axis=1)
y = df_processed['riesgo_alto']

print(f"\n2. Dimensiones:")
print(f"   - Features (X): {X.shape}")
print(f"   - Target (y): {y.shape}")
print(f"   - Variables: {list(X.columns)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\n3. División Train/Test:")
print(f"   - Train: {X_train.shape[0]} muestras")
print(f"   - Test: {X_test.shape[0]} muestras")
print(f"   - Distribución train - Riesgo alto: {y_train.mean()*100:.1f}%")
print(f"   - Distribución test - Riesgo alto: {y_test.mean()*100:.1f}%")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"\n4. Escalado aplicado (StandardScaler)")

print(f"\n5. Balanceo de clases:")
print(f"   - Antes: Clase 0={sum(y_train==0)}, Clase 1={sum(y_train==1)}")

train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
train_df['riesgo_alto'] = y_train.values

df_majority = train_df[train_df['riesgo_alto'] == 0]
df_minority = train_df[train_df['riesgo_alto'] == 1]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=len(df_majority), random_state=42)
df_balanced = pd.concat([df_majority, df_minority_upsampled])

X_train_bal = df_balanced.drop('riesgo_alto', axis=1).values
y_train_bal = df_balanced['riesgo_alto'].values

print(f"   - Después: Clase 0={sum(y_train_bal==0)}, Clase 1={sum(y_train_bal==1)}")
print("\n✓ Preprocesamiento completado")

# ============================================================
# BLOQUE 5 - ENTRENAMIENTO DE MODELOS
# ============================================================
print("=" * 60)
print("ENTRENAMIENTO DE MODELOS DE CLASIFICACIÓN")
print("=" * 60)

modelos_resultados = {}

print("\n1. Regresión Logística...")
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_bal, y_train_bal)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
auc_lr = roc_auc_score(y_test, y_prob_lr)
cv_lr = cross_val_score(lr, X_train_bal, y_train_bal, cv=5, scoring='roc_auc').mean()
modelos_resultados['Regresión Logística'] = {'modelo': lr, 'auc': auc_lr, 'cv_auc': cv_lr, 'y_pred': y_pred_lr, 'y_prob': y_prob_lr}
print(f"   ✓ AUC-ROC: {auc_lr:.4f} | CV AUC: {cv_lr:.4f}")

print("\n2. Random Forest...")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_bal, y_train_bal)
y_pred_rf = rf.predict(X_test_scaled)
y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]
auc_rf = roc_auc_score(y_test, y_prob_rf)
cv_rf = cross_val_score(rf, X_train_bal, y_train_bal, cv=5, scoring='roc_auc').mean()
modelos_resultados['Random Forest'] = {'modelo': rf, 'auc': auc_rf, 'cv_auc': cv_rf, 'y_pred': y_pred_rf, 'y_prob': y_prob_rf}
print(f"   ✓ AUC-ROC: {auc_rf:.4f} | CV AUC: {cv_rf:.4f}")

print("\n3. Gradient Boosting...")
gb = GradientBoostingClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
gb.fit(X_train_bal, y_train_bal)
y_pred_gb = gb.predict(X_test_scaled)
y_prob_gb = gb.predict_proba(X_test_scaled)[:, 1]
auc_gb = roc_auc_score(y_test, y_prob_gb)
cv_gb = cross_val_score(gb, X_train_bal, y_train_bal, cv=5, scoring='roc_auc').mean()
modelos_resultados['Gradient Boosting'] = {'modelo': gb, 'auc': auc_gb, 'cv_auc': cv_gb, 'y_pred': y_pred_gb, 'y_prob': y_prob_gb}
print(f"   ✓ AUC-ROC: {auc_gb:.4f} | CV AUC: {cv_gb:.4f}")

print("\n4. Support Vector Machine...")
svm = SVC(probability=True, random_state=42, kernel='rbf')
svm.fit(X_train_bal, y_train_bal)
y_pred_svm = svm.predict(X_test_scaled)
y_prob_svm = svm.predict_proba(X_test_scaled)[:, 1]
auc_svm = roc_auc_score(y_test, y_prob_svm)
cv_svm = cross_val_score(svm, X_train_bal, y_train_bal, cv=5, scoring='roc_auc').mean()
modelos_resultados['SVM'] = {'modelo': svm, 'auc': auc_svm, 'cv_auc': cv_svm, 'y_pred': y_pred_svm, 'y_prob': y_prob_svm}
print(f"   ✓ AUC-ROC: {auc_svm:.4f} | CV AUC: {cv_svm:.4f}")

print("\n" + "=" * 60)
print("RESUMEN DE MODELOS")
print("=" * 60)
resultados_df = pd.DataFrame({
    'Modelo': list(modelos_resultados.keys()),
    'AUC-ROC': [modelos_resultados[m]['auc'] for m in modelos_resultados],
    'CV AUC (5-fold)': [modelos_resultados[m]['cv_auc'] for m in modelos_resultados]
})
resultados_df = resultados_df.sort_values('AUC-ROC', ascending=False)
print(resultados_df.to_string(index=False))

# ============================================================
# BLOQUE 6 - EVALUACIÓN DETALLADA
# ============================================================
best_model_name = 'Gradient Boosting'
best_model = modelos_resultados[best_model_name]['modelo']
best_pred = modelos_resultados[best_model_name]['y_pred']
best_prob = modelos_resultados[best_model_name]['y_prob']

print("=" * 60)
print(f"EVALUACIÓN DETALLADA - {best_model_name.upper()}")
print("=" * 60)
print("\nClassification Report:")
print(classification_report(y_test, best_pred, target_names=['Riesgo Bajo', 'Riesgo Alto']))

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(f'EVALUACIÓN DEL MODELO - {best_model_name}', fontsize=14, fontweight='bold', y=1.02)

ax1 = axes[0, 0]
for nombre, datos in modelos_resultados.items():
    fpr, tpr, _ = roc_curve(y_test, datos['y_prob'])
    ax1.plot(fpr, tpr, label=f"{nombre} (AUC={datos['auc']:.3f})", linewidth=2)
ax1.plot([0, 1], [0, 1], 'k--', label='Aleatorio (AUC=0.500)')
ax1.set_xlabel('Tasa de Falsos Positivos')
ax1.set_ylabel('Tasa de Verdaderos Positivos')
ax1.set_title('Curvas ROC - Comparación de Modelos', fontweight='bold')
ax1.legend(loc='lower right', fontsize=8)
ax1.grid(True, alpha=0.3)

ax2 = axes[0, 1]
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Riesgo Bajo', 'Riesgo Alto'],
            yticklabels=['Riesgo Bajo', 'Riesgo Alto'])
ax2.set_title('Matriz de Confusión', fontweight='bold')
ax2.set_ylabel('Real')
ax2.set_xlabel('Predicción')

ax3 = axes[1, 0]
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=True)
ax3.barh(feature_importance['feature'], feature_importance['importance'], color='#3498db')
ax3.set_title('Importancia de Características', fontweight='bold')
ax3.set_xlabel('Importancia')

ax4 = axes[1, 1]
ax4.hist(best_prob[y_test == 0], bins=30, alpha=0.6, label='Riesgo Bajo', color='#2ecc71', density=True)
ax4.hist(best_prob[y_test == 1], bins=30, alpha=0.6, label='Riesgo Alto', color='#e74c3c', density=True)
ax4.axvline(x=0.5, color='black', linestyle='--', label='Umbral (0.5)')
ax4.set_title('Distribución de Probabilidades Predichas', fontweight='bold')
ax4.set_xlabel('Probabilidad de Riesgo Alto')
ax4.set_ylabel('Densidad')
ax4.legend()

plt.tight_layout()
plt.savefig('output/evaluacion_modelo.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("\n✓ Visualización de evaluación guardada")

# ============================================================
# BLOQUE 7 - COMPARACIÓN DE MODELOS
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('COMPARACIÓN DE MODELOS', fontsize=14, fontweight='bold', y=1.02)

ax1 = axes[0]
modelos = list(modelos_resultados.keys())
auc_scores = [modelos_resultados[m]['auc'] for m in modelos]
colors = ['#e74c3c' if m == best_model_name else '#3498db' for m in modelos]
bars = ax1.bar(modelos, auc_scores, color=colors, edgecolor='black')
ax1.set_ylabel('AUC-ROC')
ax1.set_title('AUC-ROC por Modelo', fontweight='bold')
ax1.set_ylim(0.8, 1.0)
ax1.tick_params(axis='x', rotation=45)
for bar, score in zip(bars, auc_scores):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

ax2 = axes[1]
metrics = {
    'Accuracy': accuracy_score(y_test, best_pred),
    'Precision': precision_score(y_test, best_pred),
    'Recall': recall_score(y_test, best_pred),
    'F1-Score': f1_score(y_test, best_pred),
    'AUC-ROC': roc_auc_score(y_test, best_prob)
}
bars2 = ax2.bar(metrics.keys(), metrics.values(), color='#2ecc71', edgecolor='black')
ax2.set_ylabel('Score')
ax2.set_title(f'Métricas - {best_model_name}', fontweight='bold')
ax2.set_ylim(0, 1)
for bar, (name, score) in zip(bars2, metrics.items()):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('output/comparacion_modelos.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
print("✓ Comparación de modelos guardada")

# ============================================================
# BLOQUE 8 - FUNCIÓN DE PREDICCIÓN
# ============================================================
print("=" * 60)
print("FUNCIÓN DE PREDICCIÓN - NUEVOS CLIENTES")
print("=" * 60)

def predecir_riesgo_cliente(datos_cliente):
    cliente_df = pd.DataFrame([datos_cliente])
    cliente_df['empleo_estable'] = le_empleo.transform(cliente_df['empleo_estable'])
    cliente_df = pd.get_dummies(cliente_df, columns=['tipo_vivienda', 'estado_civil'], drop_first=True)
    for col in X.columns:
        if col not in cliente_df.columns:
            cliente_df[col] = 0
    cliente_df = cliente_df[X.columns]
    cliente_scaled = scaler.transform(cliente_df)
    prob = best_model.predict_proba(cliente_scaled)[0][1]
    pred = best_model.predict(cliente_scaled)[0]
    return {
        'riesgo_alto': bool(pred),
        'probabilidad_riesgo': prob,
        'nivel_riesgo': 'ALTO' if pred == 1 else 'BAJO',
        'recomendacion': 'RECHAZAR' if pred == 1 else 'APROBAR'
    }

clientes_prueba = [
    {'edad': 35, 'ingreso_anual': 75000, 'deuda_total': 20000, 'historial_crediticio': 10,
     'num_prestamos_activos': 1, 'ratio_deuda_ingreso': 0.27, 'num_retrasos_12m': 0,
     'empleo_estable': 'Si', 'tipo_vivienda': 'Propia', 'num_tarjetas_credito': 2,
     'consultas_crediticias_6m': 1, 'estado_civil': 'Casado'},
    {'edad': 28, 'ingreso_anual': 25000, 'deuda_total': 45000, 'historial_crediticio': 1,
     'num_prestamos_activos': 3, 'ratio_deuda_ingreso': 0.85, 'num_retrasos_12m': 4,
     'empleo_estable': 'No', 'tipo_vivienda': 'Alquiler', 'num_tarjetas_credito': 5,
     'consultas_crediticias_6m': 6, 'estado_civil': 'Soltero'},
    {'edad': 52, 'ingreso_anual': 95000, 'deuda_total': 150000, 'historial_crediticio': 25,
     'num_prestamos_activos': 2, 'ratio_deuda_ingreso': 0.55, 'num_retrasos_12m': 1,
     'empleo_estable': 'Si', 'tipo_vivienda': 'Hipoteca', 'num_tarjetas_credito': 3,
     'consultas_crediticias_6m': 2, 'estado_civil': 'Casado'}
]

print("\nEjemplos de predicción para nuevos clientes:")
print("-" * 60)
for i, cliente in enumerate(clientes_prueba, 1):
    resultado = predecir_riesgo_cliente(cliente)
    print(f"\n CLIENTE {i}:")
    print(f"   Ingreso: ${cliente['ingreso_anual']:,} | Deuda: ${cliente['deuda_total']:,}")
    print(f"   Ratio D/I: {cliente['ratio_deuda_ingreso']:.2f} | Retrasos: {cliente['num_retrasos_12m']}")
    print(f"   -> RIESGO: {resultado['nivel_riesgo']} ({resultado['probabilidad_riesgo']*100:.1f}%)")
    print(f"   -> RECOMENDACIÓN: {resultado['recomendacion']}")

print("\n✓ Función de predicción lista")

# ============================================================
# BLOQUE 9 - GUARDAR MODELO Y DATOS
# ============================================================
modelo_path = 'output/modelo_riesgo_gb.pkl'
with open(modelo_path, 'wb') as f:
    pickle.dump({
        'modelo': best_model,
        'scaler': scaler,
        'label_encoder': le_empleo,
        'feature_columns': list(X.columns)
    }, f)

df.to_csv('output/dataset_riesgo_clientes.csv', index=False)

print("=" * 60)
print("ARCHIVOS GUARDADOS")
print("=" * 60)
print(f"✓ Modelo: {modelo_path}")
print(f"✓ Dataset: output/dataset_riesgo_clientes.csv")
print(f"✓ Visualizaciones:")
print(f"  - output/eda_riesgo_clientes.png")
print(f"  - output/matriz_correlacion.png")
print(f"  - output/evaluacion_modelo.png")
print(f"  - output/comparacion_modelos.png")