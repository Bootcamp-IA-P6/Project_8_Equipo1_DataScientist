# Estratificación de riesgo de ictus mediante modelos predictivos multimodales

## 1. Introducción

El presente sistema tiene como objetivo apoyar la identificación precoz de pacientes con riesgo de ictus mediante la integración de modelos de aprendizaje automático aplicados a datos clínicos y de imagen.

Se emplean dos modelos principales:

- Modelo clínico basado en XGBoost, entrenado sobre variables epidemiológicas y biométricas.
- Modelo de imagen basado en red neuronal convolucional (CNN, arquitectura EfficientNetB0) aplicado a imágenes de tomografía computarizada (TC).

Ambos modelos presentan características diferenciales en términos de calibración, rendimiento y tipo de información procesada.

---

## 2. Características de los modelos

| Modelo   | Tipo de datos        | AUC   | Sensibilidad | Precisión |
|----------|---------------------|-------|--------------|-----------|
| XGBoost  | Variables clínicas  | 0.849 | 0.80         | 0.165     |
| CNN      | Imagen TC           | 0.962 | 0.902        | 0.827     |

El modelo clínico ha sido entrenado en un contexto de desbalance significativo entre clases (`scale_pos_weight = 19.1`), reflejando una baja prevalencia de ictus en la población analizada (~5%).

Como consecuencia, las probabilidades generadas por dicho modelo **no deben interpretarse como probabilidades absolutas calibradas**, sino como puntuaciones de riesgo relativas.

Por el contrario, el modelo de imagen presenta una mejor calibración probabilística, debido a un conjunto de entrenamiento más balanceado.

---

## 3. Umbral de decisión clínica

El modelo XGBoost utiliza un umbral óptimo de clasificación:

- **Threshold = 0.5517 (55.17%)**

Este umbral ha sido determinado mediante optimización basada en el índice de Youden, priorizando una alta sensibilidad diagnóstica (80%).

Desde el punto de vista clínico, esta elección es coherente con estrategias de cribado, donde:

> La minimización de falsos negativos (casos no detectados) es prioritaria frente a la reducción de falsos positivos.

---

## 4. Limitaciones de umbrales fijos convencionales

El uso inicial de umbrales arbitrarios (30% y 60%) generaba inconsistencias entre la salida del modelo y la categorización clínica del riesgo.

En particular, existía un intervalo en el cual:

- El modelo clasificaba al paciente como positivo (≥55.17%)
- La aplicación lo categorizaba como riesgo moderado (<60%)

Esta discrepancia supone un riesgo potencial en entornos de triaje clínico.

---

## 5. Ajuste de estratificación basado en el modelo

Se ha redefinido la clasificación del riesgo alineándola con el umbral del modelo:

- **Riesgo alto:** ≥ 55%  
- **Riesgo moderado:** 40% – 55%  
- **Riesgo bajo:** < 40%  

Este ajuste elimina inconsistencias y mejora la coherencia entre predicción algorítmica e interpretación clínica.

---

## 6. Corrección por desbalance y estimación de riesgo real

Debido al desbalance de clases en el entrenamiento del modelo clínico, se ha incorporado una aproximación para estimar la probabilidad real de ictus:

```

P_real ≈ P_modelo / (P_modelo + (1 - P_modelo) * 19.1)

```

Esto implica que:

| Probabilidad del modelo | Probabilidad real estimada |
|------------------------|---------------------------|
| 55%                    | ~6%                       |
| 75%                    | ~14%                      |
| 90%                    | ~32%                      |

Por tanto, las probabilidades del modelo deben interpretarse como indicadores de riesgo relativo y no como estimaciones directas de probabilidad absoluta.

---

## 7. Estratificación final del riesgo

Con el fin de mejorar la toma de decisiones clínicas, se establece una clasificación en cuatro niveles:

- **Riesgo bajo (<40%)**  
  Seguimiento clínico rutinario.

- **Riesgo moderado (40–55%)**  
  Vigilancia clínica y reevaluación.

- **Riesgo alto (55–75%)**  
  Evaluación neurológica prioritaria.

- **Riesgo crítico (>75%)**  
  Activación inmediata de protocolo de ictus.

Esta estratificación permite una mejor diferenciación de los niveles de urgencia asistencial.

---

## 8. Integración de modelos multimodales

Dado que los modelos clínico y de imagen presentan diferentes escalas de probabilidad, se ha implementado un proceso de normalización previo a su combinación.

Cada probabilidad se ajusta en relación a su umbral de decisión, permitiendo su comparación en una escala común.

Posteriormente, se realiza una combinación ponderada basada en el rendimiento de cada modelo (AUC):

- Modelo clínico (XGBoost): 40%  
- Modelo de imagen (CNN): 60%  

Este enfoque permite aprovechar de forma equilibrada la información clínica y radiológica.

---

## 9. Consideraciones para uso clínico

- El sistema está diseñado como herramienta de apoyo a la decisión clínica, no como sustituto del juicio médico.
- La baja precisión del modelo clínico (16.5%) es esperable en contextos de cribado poblacional.
- Los resultados deben interpretarse en conjunto con la evaluación clínica y otras pruebas diagnósticas.

---

## 10. Conclusión

La redefinición de los umbrales de riesgo y la integración adecuada de modelos multimodales permiten mejorar la coherencia, interpretabilidad y aplicabilidad clínica del sistema.

Este enfoque favorece la detección precoz de casos de ictus, optimizando la priorización asistencial y reduciendo el riesgo de omisión diagnóstica.