# Estratificación de Riesgo de Ictus mediante Machine Learning
## Informe Técnico del Proyecto

## Equipo

| Nombre | Rol |
|---|---|
| Iris Amorim | Data Scientist & Product Owner |
| Jonathan Basales | Data Scientist & Scrum Master |
| Gema Yebenes | Data Scientist |

---

## Tabla de Contenidos

1. [Contexto del Proyecto](#1-contexto-del-proyecto)
2. [Dataset y Análisis Exploratorio (EDA)](#2-dataset-y-análisis-exploratorio-eda)
3. [Decisiones de Preprocesado](#3-decisiones-de-preprocesado)
4. [Gestión del Desbalance de Clases](#4-gestión-del-desbalance-de-clases)
5. [Estrategia de Modelado Clínico](#5-estrategia-de-modelado-clínico)
6. [Optimización de Hiperparámetros con Optuna](#6-optimización-de-hiperparámetros-con-optuna)
7. [Validación Estadística de Decisiones](#7-validación-estadística-de-decisiones)
8. [Modelo de Imagen — CNN EfficientNetB0](#8-modelo-de-imagen--cnn-efficientnetb0)
9. [Umbral de Decisión Clínica](#9-umbral-de-decisión-clínica)
10. [Estratificación del Riesgo en Cuatro Niveles](#10-estratificación-del-riesgo-en-cuatro-niveles)
11. [Corrección de Probabilidades por Desbalance](#11-corrección-de-probabilidades-por-desbalance)
12. [Integración Multimodal](#12-integración-multimodal)
13. [Tracking de Experimentos con MLflow](#13-tracking-de-experimentos-con-mlflow)
14. [Resultados Finales](#14-resultados-finales)
15. [Consideraciones Clínicas y Limitaciones](#15-consideraciones-clínicas-y-limitaciones)

---

## 1. Contexto del Proyecto

El ictus es una emergencia médica donde cada minuto cuenta. El objetivo de este proyecto es construir un **sistema de apoyo a la decisión clínica** que ayude a identificar precozmente qué pacientes tienen mayor riesgo, para que los profesionales sanitarios puedan priorizar la atención.

El sistema integra dos fuentes de información complementarias:

- **Datos clínicos**: variables demográficas y biométricas del paciente (edad, glucosa, hipertensión, etc.), procesadas por un modelo XGBoost.
- **Imagen médica (TC)**: tomografías computarizadas cerebrales, analizadas por una red neuronal convolucional (CNN).

El sistema no reemplaza al médico. Actúa como un segundo par de ojos que alerta cuando los datos apuntan a un riesgo elevado.

---

## 2. Dataset y Análisis Exploratorio (EDA)

### El dataset

Se trabajó con `stroke_dataset.csv`, que contiene registros de pacientes con 10 variables clínicas y demográficas, además de la variable objetivo `stroke` (1 = ha sufrido ictus, 0 = no).

### El principal problema: el desbalance de clases

Este es el hallazgo más determinante de todo el EDA y condiciona prácticamente todas las decisiones técnicas posteriores:

- **Clase 0** (sin ictus): ~95% de los pacientes
- **Clase 1** (ictus): ~5% de los pacientes
- **Ratio: 19 pacientes sanos por cada paciente con ictus**

Esto tiene una consecuencia directa: un modelo que prediga siempre "no ictus" tendría un 95% de accuracy, pero sería completamente inútil clínicamente. Por eso la **accuracy fue descartada como métrica** desde el inicio, sustituyéndola por AUC-ROC y Recall.

### Variables más relevantes para predecir el ictus

El análisis de distribuciones y correlaciones reveló una jerarquía clara entre predictores:

- **Edad (`age`)**: es el predictor más fuerte. Los pacientes con ictus presentan edades significativamente mayores. La relación no es lineal, lo que favorece el uso de modelos de árbol frente a modelos lineales.
- **Glucosa media (`avg_glucose_level`)**: valores elevados están claramente asociados al ictus. Presenta distribución con cola larga y outliers que, en contexto médico, pueden ser clínicamente reales (no errores de registro).
- **IMC (`bmi`)**: mayor dispersión y menor separación entre clases. Contribuye al modelo, pero con menos fuerza que las anteriores.
- **Hipertensión y enfermedad cardíaca**: variables binarias con relación directa con el riesgo.
- **Hábito tabáquico (`smoking_status`)**: existen diferencias entre categorías, aunque la categoría `Unknown` introdujo ruido que requirió tratamiento específico.

La ausencia de correlaciones lineales fuertes entre variables confirmó que los modelos no lineales (Random Forest, XGBoost) serían más adecuados que la Regresión Logística para capturar las relaciones complejas presentes en los datos.

### Decisiones de limpieza de datos

**`work_type = 'children'` → recodificado como `'not_applied'`**
Los menores no trabajan en ningún sentido convencional del término. Mantener esta categoría separada creaba un grupo de muy baja frecuencia que generaría ruido en el codificado one-hot. Unificarla con `not_applied` es semánticamente coherente y técnicamente más limpio.

**`smoking_status = 'Unknown'` en menores de 16 años → `'not_applied'`**
En menores, el valor `Unknown` no representa un hábito de tabaquismo desconocido, sino ausencia de información relevante para esa franja de edad. Recodificarlo evita que el modelo interprete incorrectamente este valor como un indicador de comportamiento.

### Dos versiones del dataset para experimentar

| Versión | Descripción | Uso |
|---|---|---|
| `full` | Todos los pacientes, incluidos menores de 16 años | Experimentos generales |
| `adults` | Solo pacientes con `age > 16` | Comprobar si los menores aportan ruido |

La elección entre versiones fue validada estadísticamente (ver Sección 7).

---

## 3. Decisiones de Preprocesado

### Por qué `imblearn.pipeline.Pipeline` y no `sklearn.pipeline.Pipeline`

Esta es una decisión técnica crítica que muchos proyectos pasan por alto. La versión de `imbalanced-learn` sabe que **SMOTE solo debe aplicarse durante el entrenamiento**, nunca al evaluar ni al hacer predicciones en producción. La versión estándar de sklearn no tiene esa lógica incorporada, lo que puede producir data leakage silencioso — el modelo parece mejor de lo que realmente es porque aprende de datos de test indirectamente.

El pipeline sigue siempre el mismo orden:

```
Preprocesado → [SMOTE, solo en train] → Modelo
```

### Transformaciones según el tipo de variable

| Tipo de variable | Transformación aplicada | Por qué |
|---|---|---|
| Numéricas con modelos lineales | `StandardScaler` (media 0, desv. 1) | La Regresión Logística es sensible a la escala de las variables |
| Numéricas con modelos de árbol | Sin transformación (`passthrough`) | XGBoost y Random Forest no necesitan escalado para funcionar correctamente |
| Categóricas | `OneHotEncoder(handle_unknown='ignore')` | Convierte texto en columnas binarias 0/1; el parámetro `ignore` evita errores si en producción llega una categoría no vista en entrenamiento |

### División de datos

División estratificada: 80% entrenamiento / 20% test, con `random_state=42` para garantizar reproducibilidad. La estratificación garantiza que el 5% de casos positivos se preserve en ambas particiones, evitando que el test set quede sin apenas casos de ictus para evaluar.

---

## 4. Gestión del Desbalance de Clases

Se evaluaron y compararon dos estrategias para que el modelo no ignorase la clase minoritaria:

### Estrategia A: SMOTE

SMOTE crea pacientes sintéticos de la clase minoritaria interpolando matemáticamente entre casos reales existentes. El resultado es un dataset de entrenamiento artificialmente balanceado.

- **Ventaja**: el modelo aprende con proporciones equilibradas entre clases.
- **Limitación**: los pacientes sintéticos son aproximaciones matemáticas, no registros reales. En fronteras de decisión complejas puede introducir ruido.

### Estrategia B: `scale_pos_weight` (nativo de XGBoost)

En lugar de crear datos nuevos, se indica al modelo que los errores sobre la clase minoritaria tienen un coste mayor. El valor se calcula directamente como:

```
scale_pos_weight = pacientes sin ictus / pacientes con ictus = 19.1
```

En cada fold de validación cruzada, este valor se recalcula sobre los datos específicos de ese fold para evitar leakage.

### Decisión final y su justificación estadística

El modelo ganador usa **`scale_pos_weight = 19.1` sin SMOTE**. La validación estadística (ver Sección 7) confirmó que SMOTE no aportó mejora significativa en ninguna métrica, por lo que añadirlo solo habría introducido complejidad innecesaria al pipeline.

---

## 5. Estrategia de Modelado Clínico

### Modelos comparados

Se entrenaron y evaluaron tres familias de modelos, cada una en versión baseline y optimizada con Optuna:

1. **Regresión Logística** — modelo lineal, sirve como referencia interpretable
2. **Random Forest** — ensemble de árboles de decisión independientes
3. **XGBoost** — gradient boosting; modelo ganador final

### Métricas utilizadas

Dado el desbalance severo, las métricas prioritarias fueron:

- **AUC-ROC** *(métrica principal)*: mide qué tan bien el modelo separa las dos clases, independientemente del umbral de decisión. Un valor de 1.0 sería perfecto; 0.5 equivale a adivinar al azar.
- **Recall** *(métrica secundaria)*: mide qué porcentaje de los ictus reales detecta el modelo. En cribado médico, no detectar un ictus (falso negativo) es mucho más grave que una falsa alarma (falso positivo).

### Validación cruzada con búsqueda de umbral sin data leakage

Se implementó una validación cruzada manual de 5 folds (`StratifiedKFold`) que resuelve un problema técnico importante: la función estándar `cross_validate` de sklearn no da acceso a las probabilidades por fold, y sin ellas no se puede buscar el umbral óptimo en validación sin contaminar el test set.

El proceso en cada uno de los 5 folds fue:

1. Entrenar el modelo sobre los datos de entrenamiento del fold.
2. Obtener probabilidades sobre los datos de validación del fold.
3. Buscar el umbral óptimo **solo con los datos de validación de ese fold**.
4. Registrar umbral y métricas de ese fold.

Al final de los 5 folds, se promedian los umbrales encontrados y ese promedio se aplica al test set. El test set **nunca participa** en la búsqueda del umbral. Esta distinción es fundamental para que la evaluación final sea honesta.

### Por qué F-beta (β=2) para buscar el umbral

El umbral óptimo dentro de cada fold se buscó maximizando F-beta con β=2. Esta métrica pondera el Recall **el doble** que la Precisión, reflejando explícitamente que en cribado médico no detectar un ictus es dos veces más costoso que generar una falsa alarma.

### Detección de overfitting

En cada experimento se calcularon las diferencias train-test para las métricas clave:

```
Δ AUC    = AUC_train - AUC_test
Δ Recall = Recall_train - Recall_test
Δ F1     = F1_train - F1_test
```

Estas diferencias se registraron en MLflow para monitorizar el overfitting de forma sistemática en todos los runs.

---

## 6. Optimización de Hiperparámetros con Optuna

### Qué es Optuna y por qué usarlo

Los modelos de machine learning tienen decenas de parámetros configurables (profundidad de los árboles, tasa de aprendizaje, regularización…). Probarlos todos manualmente sería inviable. Optuna automatiza esta búsqueda de forma inteligente: en lugar de probar combinaciones al azar, **aprende de los intentos anteriores** cuáles zonas del espacio de parámetros son más prometedoras (búsqueda bayesiana con TPE Sampler).

### Configuración

- **25 trials** por experimento
- **Objetivo**: maximizar AUC-ROC en validación cruzada
- El test set nunca es accesible durante la búsqueda de hiperparámetros

### Escala de experimentos

Se ejecutaron hasta **24 combinaciones** (3 modelos × 2 versiones del dataset × 2 configuraciones de SMOTE), cada una en versión baseline y versión Optuna, registradas todas en MLflow. El impacto real de Optuna fue validado estadísticamente (ver Sección 7).

---

## 7. Validación Estadística de Decisiones

Esta sección responde a una pregunta esencial: ¿las diferencias que observamos entre modelos y configuraciones son reales, o podrían deberse al azar?

Para responderla se aplicaron tres tests estadísticos, cada uno adecuado para el tipo de comparación que se quería hacer.

> **Cómo leer el p-valor**: indica la probabilidad de observar esa diferencia por puro azar. Si p < 0.05, la diferencia se considera estadísticamente significativa — es decir, hay menos de un 5% de probabilidad de que sea casualidad.

---

### Test 1 — McNemar: ¿XGBoost o Regresión Logística?

**¿Qué compara?** Los dos mejores modelos por AUC-ROC: XGBoost_optuna (AUC = 0.854) frente a LogisticRegression_optuna (AUC = 0.842).

McNemar es el test adecuado porque no compara porcentajes globales, sino los **desacuerdos entre modelos sobre los mismos pacientes**. Si un modelo falla donde el otro acierta, eso es una diferencia real. Si ambos fallan o ambos aciertan en los mismos casos, no hay diferencia sustancial.

De 997 pacientes evaluados:

| | LogReg acierta | LogReg falla |
|---|---|---|
| **XGBoost acierta** | 729 | 33 |
| **XGBoost falla** | 17 | 218 |

XGBoost acertó en 33 casos donde LogReg falló. LogReg acertó en 17 casos donde XGBoost falló.

```
Chi² = 4.50  |  p-valor = 0.034
✅ Los modelos difieren significativamente (p < 0.05)
```

**Conclusión**: XGBoost no solo tiene mejor AUC numéricamente — la diferencia es estadísticamente real. La elección de XGBoost como modelo clínico está respaldada por evidencia estadística, no solo por preferencia técnica.

---

### Test 2 — T-test: ¿Merece la pena usar SMOTE?

**¿Qué compara?** El rendimiento promedio de todos los experimentos con SMOTE activado frente a los experimentos sin SMOTE. El t-test compara medias de dos grupos para determinar si la diferencia es significativa.

| Métrica | Con SMOTE | Sin SMOTE | p-valor | ¿Diferencia significativa? |
|---|---|---|---|---|
| AUC | 0.811 | 0.815 | 0.681 | ❌ No |
| Recall | 0.552 | 0.533 | 0.895 | ❌ No |
| F1 | 0.194 | 0.193 | 0.966 | ❌ No |

```
Todos los p-valores >> 0.05
❌ SMOTE no aportó mejora estadísticamente significativa en ninguna métrica
```

**Conclusión**: SMOTE no empeora el modelo, pero tampoco lo mejora de forma demostrable. Dado que añade complejidad al pipeline y tiempo de entrenamiento, la decisión de no usarlo en el modelo final está técnicamente justificada.

---

### Test 3 — T-test: ¿Incluir menores de 16 años mejora el modelo?

**¿Qué compara?** Los experimentos entrenados con todos los pacientes (`full`) frente a los entrenados solo con adultos (`adults`).

| Métrica | Dataset Full | Dataset Adults | p-valor | ¿Diferencia significativa? |
|---|---|---|---|---|
| AUC | 0.823 | 0.803 | 0.088 | ❌ No |
| Recall | 0.565 | 0.520 | 0.754 | ❌ No |
| F1 | 0.195 | 0.192 | 0.933 | ❌ No |

```
Todos los p-valores > 0.05
❌ La diferencia entre versiones del dataset no es estadísticamente significativa
```

**Conclusión**: Aunque el dataset `full` puntúa marginalmente mejor en todas las métricas, la diferencia podría deberse al azar. Se optó por mantener el dataset completo para preservar toda la información clínica disponible, pero la decisión no es crítica para el rendimiento del modelo.

---

### Test 4 — Wilcoxon Signed-Rank: ¿Optuna mejora realmente sobre los modelos baseline?

**¿Qué compara?** El rendimiento de los modelos sin optimizar (baseline) frente a los modelos optimizados con Optuna.

Wilcoxon es el test apropiado aquí porque compara **pares emparejados** (mismo modelo, mismo dataset, misma configuración — solo cambia si fue optimizado o no). Es más robusto que el t-test cuando los datos no siguen una distribución normal perfecta.

| Métrica | Baseline | Optuna | p-valor | ¿Diferencia significativa? |
|---|---|---|---|---|
| AUC | 0.800 | 0.826 | **0.005** | ✅ Sí |
| Recall | 0.362 | 0.723 | **0.003** | ✅ Sí |
| F1 | 0.160 | 0.227 | 0.088 | ❌ No |

```
AUC:    p = 0.005  ✅ Optuna mejora significativamente el AUC
Recall: p = 0.003  ✅ Optuna mejora significativamente el Recall (+36 puntos porcentuales)
F1:     p = 0.088  ❌ La mejora en F1 no alcanza significación estadística
```

**Conclusión**: La optimización con Optuna no es un adorno técnico — produce mejoras reales y estadísticamente demostradas, especialmente en Recall. El salto de 0.36 a 0.72 en Recall es la diferencia entre detectar 1 de cada 3 ictus o detectar más de 7 de cada 10. En términos clínicos, eso es sustancial.

---

### Resumen de decisiones respaldadas estadísticamente

| Decisión tomada | Test aplicado | Veredicto |
|---|---|---|
| Elegir XGBoost sobre Regresión Logística | McNemar | ✅ Diferencia significativa (p = 0.034) |
| No usar SMOTE en el modelo final | T-test | ✅ Confirmado: SMOTE no aporta mejora demostrable |
| Mantener dataset completo (`full`) | T-test | ⚠️ Diferencia no significativa; se mantiene por criterio clínico |
| Usar Optuna en lugar de modelos baseline | Wilcoxon | ✅ Mejora significativa en AUC (p = 0.005) y Recall (p = 0.003) |

---

## 8. Modelo de Imagen — CNN EfficientNetB0

### El dataset de imágenes

Se trabajó con 2.501 tomografías cerebrales en formato `.jpg`, con tamaño uniforme de 650×650 píxeles:

| Clase | Imágenes | Porcentaje |
|---|---|---|
| Normal | 1.551 | 62% |
| Stroke | 950 | 38% |
| **Total** | **2.501** | |

El desbalance es de 1.63:1, mucho más manejable que el 19:1 del modelo clínico. La gran cantidad de píxeles negros (fondo de las TC) era igual en ambas clases, lo que confirmó que la diferencia entre clases es sutil y localizada en zonas específicas del cerebro — de ahí la necesidad de una CNN en lugar de estadísticas simples.

### División del dataset de imagen

| Partición | Imágenes | Normal | Stroke | Función |
|---|---|---|---|---|
| Train | 1.750 | 1.085 | 665 | El modelo aprende aquí |
| Validación | 375 | 233 | 142 | Controlar que no memorice |
| Test | 376 | 233 | 143 | Evaluación final — no se toca hasta el final |

La división fue estratificada para mantener la misma proporción de clases en los tres conjuntos.

### Arquitectura: Transfer Learning en dos fases

Con solo 2.501 imágenes, entrenar una red desde cero tiene limitaciones claras. Se aplicó **Transfer Learning** con EfficientNetB0, una arquitectura preentrenada con millones de imágenes de ImageNet.

La analogía: es como contratar a un radiólogo con 10 años de experiencia y enseñarle específicamente a detectar ictus, en lugar de formar a alguien desde cero sin ningún conocimiento previo de imágenes médicas.

**Fase 1 — Adaptar solo la capa clasificadora (15 epochs, `lr = 1e-4`)**

El backbone de EfficientNet se congela — no aprende. Solo se entrena la pequeña capa de decisión final. El backbone ya sabe extraer características visuales; solo necesitamos enseñarle a usarlas para este problema concreto.

```
Epoch 1  →  val_AUC: 0.711   val_Recall: 0.457
Epoch 15 →  val_AUC: 0.899   val_Recall: 0.852   ← mejor epoch Fase 1
```

**Fase 2 — Fine-tuning de las últimas 20 capas (30 epochs, `lr = 1e-5`)**

Se descongelan las capas más profundas del backbone y se ajustan con un learning rate 10 veces más pequeño. Un learning rate alto destruiría los pesos preentrenados valiosos; uno pequeño los ajusta suavemente al dominio de las TC cerebrales.

```
Epoch 1  →  val_AUC: 0.862   val_Recall: 0.803   (parte desde Fase 1)
Epoch 30 →  val_AUC: 0.934   val_Recall: 0.880   ← mejor epoch Fase 2
```

### Detalle técnico crítico: el rango de entrada de las imágenes

EfficientNetB0 fue diseñado esperando imágenes en rango **[0, 255]**. Tiene su propia normalización interna. Si se le pasan imágenes ya normalizadas a **[0, 1]**, los features extraídos son incorrectos y el modelo no puede aprender nada útil. Este error produjo `val_recall = 0.0` en todos los epochs en los intentos iniciales.

La solución fue mantener **dos pipelines de carga separados**: uno para el baseline CNN (normalizado a [0, 1]) y otro para EfficientNet (sin normalizar, rango [0, 255]). Este dato queda registrado en el `metadata.json` del modelo para que cualquier sistema que lo cargue lo use correctamente.

### Grad-CAM: ¿qué zonas del cerebro mira el modelo para decidir?

Grad-CAM es una técnica de explicabilidad que genera un mapa de calor sobre la imagen original, señalando qué zonas influyen más en la decisión del modelo. En medicina, no basta con que el modelo acierte — hay que poder justificar por qué.

Si el mapa de calor señala las zonas cerebrales clínicamente relevantes (y no el fondo negro o los bordes de la imagen), la decisión puede considerarse clínicamente justificable. La implementación requirió una solución técnica específica: EfficientNet está anidado como una subcapa dentro del modelo externo, lo que impide el acceso directo a los gradientes intermedios. Se resolvió dividiendo el modelo en dos submodelos y conectándolos mediante `tf.GradientTape()`.

### Resultados finales en test set

| Modelo | AUC | Recall | Precisión | F1 | Estabilidad |
|---|---|---|---|---|---|
| Baseline CNN (desde cero) | 0.954 | 0.972 | 0.665 | 0.790 | Baja (recall oscilante entre epochs) |
| **EfficientNetB0** | **0.962** | **0.902** | **0.827** | **0.863** | **Alta** |

**Modelo seleccionado: EfficientNetB0**, por su equilibrio superior entre Recall, Precisión y estabilidad durante el entrenamiento. El baseline alcanzó un Recall más alto (0.972) pero con Precisión muy baja (0.665) y comportamiento inestable, lo que lo hace poco fiable en producción.

---

## 9. Umbral de Decisión Clínica

### El problema del umbral por defecto (0.5)

Por defecto, los modelos clasifican como positivo todo lo que supera el 50% de probabilidad. Este umbral no es adecuado aquí por dos razones:

1. Las probabilidades del modelo clínico están distorsionadas por el desbalance de entrenamiento.
2. En cribado médico, es clínicamente correcto aceptar más falsas alarmas (falsos positivos) a cambio de no perder ningún ictus real (falsos negativos).

### Umbral seleccionado

```
Threshold = 0.5517  (55.17%)
Recall resultante: 80%
```

Este umbral fue determinado mediante el **índice de Youden**, que maximiza la suma de Sensibilidad y Especificidad. Se buscó durante la validación cruzada, nunca sobre el test set.

### La inconsistencia que se detectó y corrigió

El uso de umbrales arbitrarios anteriores (30% y 60%) creaba una zona de inconsistencia con potencial impacto clínico real:

> Un paciente con probabilidad del modelo del 57% era clasificado como **positivo** (≥ 55.17%) por el algoritmo, pero la aplicación lo mostraba como **riesgo moderado** (< 60%). El médico vería una categorización de bajo riesgo para un paciente que el modelo ya había marcado como caso positivo.

En un entorno de triaje, esta discrepancia podría traducirse en una decisión clínica equivocada. La redefinición de los niveles de estratificación eliminó esta zona problemática.

---

## 10. Estratificación del Riesgo en Cuatro Niveles

La estratificación final alinea los niveles de riesgo con el umbral del modelo, eliminando cualquier inconsistencia entre la predicción del algoritmo y la categoría mostrada al clínico:

| Nivel | Probabilidad del modelo | Acción clínica recomendada |
|---|---|---|
| 🟢 **Riesgo bajo** | < 40% | Seguimiento clínico rutinario |
| 🟡 **Riesgo moderado** | 40% – 55% | Vigilancia y reevaluación |
| 🟠 **Riesgo alto** | 55% – 75% | Evaluación neurológica prioritaria |
| 🔴 **Riesgo crítico** | > 75% | Activación inmediata del protocolo de ictus |

El límite inferior de "Riesgo alto" (55%) coincide exactamente con el threshold de clasificación del modelo (0.5517), garantizando que todo paciente que el modelo clasifica como positivo aparezca al menos en la categoría de riesgo alto.

---

## 11. Corrección de Probabilidades por Desbalance

### Por qué el 57% del modelo no significa un 57% de riesgo real

El modelo clínico fue entrenado con un peso artificial de 19.1 para la clase minoritaria (`scale_pos_weight`). Esto desplaza las probabilidades de salida hacia arriba. Un paciente al que el modelo asigna 57% de probabilidad no tiene un 57% de probabilidad epidemiológica real de sufrir un ictus — ese número está amplificado por el diseño del entrenamiento.

Para estimar la probabilidad real se aplica la siguiente corrección:

```
P_real ≈ P_modelo / (P_modelo + (1 - P_modelo) × 19.1)
```

### Tabla de conversión

| Probabilidad del modelo | Probabilidad real estimada |
|---|---|
| 55% | ~6% |
| 75% | ~14% |
| 90% | ~32% |

Esta corrección es fundamental para comunicar el riesgo de forma honesta a los profesionales sanitarios. Las probabilidades del modelo deben entenderse como **indicadores de riesgo relativo**, no como estimaciones epidemiológicas absolutas.

---

## 12. Integración Multimodal

### El problema: dos modelos que no hablan el mismo idioma

Los dos modelos operan en escalas probabilísticas distintas:

- El modelo clínico (XGBoost) tiene probabilidades amplificadas por el desbalance.
- El modelo de imagen (CNN) tiene mejor calibración (entrenamiento más balanceado, ratio 1.63:1).

Combinarlos directamente sin ajuste previo equivaldría a mezclar dos termómetros con escalas distintas.

### Proceso de integración

**Paso 1 — Normalización por umbral**: cada probabilidad se reescala en relación a su propio umbral de decisión, llevándolas a una escala comparable antes de combinarlas.

**Paso 2 — Combinación ponderada por AUC**: se da más peso al modelo que mejor discrimina:

| Modelo | Peso en combinación | AUC base |
|---|---|---|
| XGBoost (clínico) | 40% | 0.849 |
| CNN (imagen TC) | 60% | 0.962 |

### Lógica de activación del sistema híbrido

El modelo de imagen no se activa para todos los pacientes — solo cuando el modelo clínico lo justifica. Esto ahorra recursos y centra la TC donde realmente aporta valor:

```
XGBoost evalúa los datos clínicos del paciente
    ├── p < 0.30  →  BAJO RIESGO     (confianza alta, sin necesidad de TC)
    ├── p > 0.80  →  ALTO RIESGO     (confianza alta, sin necesidad de TC)
    └── 0.30–0.80 →  ZONA INCIERTA   → activar CNN con imagen TC
                         ├── Combinación ponderada (40% XGB + 60% CNN)
                         └── |p_xgb - p_cnn| > 0.25  →  REVISIÓN MANUAL
```

Cuando los dos modelos discrepan significativamente (diferencia > 25 puntos porcentuales), el sistema no decide solo — escala el caso a revisión médica manual.

---

## 13. Tracking de Experimentos con MLflow

MLflow registró automáticamente cada experimento entrenado, garantizando reproducibilidad y trazabilidad completa del proceso.

### Qué se registró por cada run

- **Parámetros**: hiperparámetros del modelo, versión del dataset, uso de SMOTE, autor del experimento
- **Métricas de rendimiento**: AUC, Recall, F1, Precision
- **Métricas de overfitting**: Δ AUC, Δ Recall, Δ F1 (diferencia train-test) para detectar memorización
- **Artefactos**: modelo serializado en disco, matrices de confusión
- **Tags**: autor, tipo de modelo, versión del dataset, tipo de optimización (baseline / optuna)

### Trabajo colaborativo

Cada miembro del equipo etiquetó sus runs con su nombre. Esto permitió comparar contribuciones, reproducir cualquier experimento anterior con exactamente los mismos parámetros y mantener la trazabilidad del proceso de decisión colectivo a lo largo del proyecto.

---

## 14. Resultados Finales

### Modelo clínico seleccionado: XGBoost_optuna

| Métrica | Valor |
|---|---|
| AUC-ROC | **0.849** |
| Recall | **0.80** |
| Precisión | 0.165 |
| F1-score | 0.274 |
| PR-AUC | 0.172 |
| Threshold | 0.5517 |
| scale_pos_weight | 19.1 |

La precisión baja (16.5%) es **esperada y aceptable** en este contexto. En cribado poblacional con 5% de prevalencia, incluso un modelo excelente generará más falsas alarmas que verdaderos positivos. Lo clínicamente relevante es que detecta el 80% de los ictus reales.

El PR-AUC de 0.172 refleja la dificultad intrínseca del problema con 5% de prevalencia y es coherente con los benchmarks publicados para datasets similares.

### Modelo de imagen seleccionado: EfficientNetB0

| Métrica | Valor |
|---|---|
| AUC-ROC | **0.962** |
| Recall | **0.902** |
| Precisión | 0.827 |
| F1-score | 0.863 |
| Threshold | 0.5 |
| Rango de entrada esperado | [0, 255] |

### Comparativa global

| Modelo | Datos de entrada | AUC | Recall | Precisión |
|---|---|---|---|---|
| XGBoost_optuna | Variables clínicas (10 features) | 0.849 | 0.80 | 0.165 |
| EfficientNetB0 | Tomografía computarizada | 0.962 | 0.902 | 0.827 |

### Features del modelo clínico

```
gender, age, hypertension, heart_disease, ever_married,
work_type, Residence_type, avg_glucose_level, bmi, smoking_status
```

---

## 15. Consideraciones Clínicas y Limitaciones

### Uso apropiado del sistema

El sistema está diseñado como herramienta de apoyo — no como diagnóstico automático. Sus resultados deben interpretarse junto a la evaluación clínica y otras pruebas diagnósticas. El criterio médico siempre prevalece sobre la predicción del modelo.

### Limitaciones conocidas

| Limitación | Detalle |
|---|---|
| Probabilidades no calibradas | Las salidas del modelo clínico no representan probabilidades absolutas reales debido al desbalance de entrenamiento; requieren corrección para interpretarse epidemiológicamente |
| Generalización | El modelo se validó sobre un dataset específico; requiere validación prospectiva en población real antes de cualquier uso clínico |
| Dependencia del umbral | El threshold 0.5517 se optimizó sobre los datos disponibles; puede requerir ajuste en poblaciones con prevalencia diferente |
| CNN como caja negra | Las redes convolucionales no son transparentes; Grad-CAM ofrece explicabilidad parcial pero no equivale a interpretabilidad médica formal |
| Dataset de imagen | Si las TC no provienen de entornos clínicos reales, los resultados de la CNN deben validarse en condiciones hospitalarias antes de su uso |

### Líneas de trabajo futuro

- **Calibración del modelo clínico**: aplicar calibración isotónica o de Platt para que las probabilidades sean directamente interpretables como riesgo absoluto.
- **Augmentation más agresiva en CNN**: añadir rotaciones pequeñas (±10°) y zoom (±10%) podría mejorar la generalización del modelo de imagen a nuevas procedencias de TC.
- **Focal loss para el baseline CNN**: resolvería la inestabilidad del Recall entre epochs observada durante el entrenamiento, al estar diseñada específicamente para clases difíciles.
- **Umbral optimizado para la CNN**: el threshold actual de 0.5 puede ajustarse sobre la curva PR para maximizar Recall sin disparar en exceso las falsas alarmas; valores en el rango 0.35–0.40 son candidatos prometedores.
- **Calibración conjunta del sistema híbrido**: los thresholds del sistema multimodal deben ajustarse conjuntamente usando las curvas PR de ambos modelos sobre el mismo test set.
- **Validación prospectiva hospitalaria**: paso imprescindible antes de cualquier despliegue clínico real.

