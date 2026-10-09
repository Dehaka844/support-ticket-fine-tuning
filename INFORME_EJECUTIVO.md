# Informe ejecutivo --- Clasificador de tickets con Fine-Tuning

## Objetivo

Construir un clasificador de intenciones para tickets de soporte técnico
y comparar fine-tuning completo de BERT con una alternativa eficiente
basada en LoRA. Se generó un dataset sintético de 5.000 tickets en seis
categorías, dividido en entrenamiento (3.500), validación (750) y test
(750).

## Mejor modelo y configuración

Se selecciona **BERT Full (`bert-base-uncased`)** para la inferencia del
prototipo. Configuración principal: longitud máxima de 128 tokens,
learning rate `2e-5`, batch de entrenamiento 16, scheduler lineal,
weight decay `0.01`, FP16, semilla 42, pesos de clase y early stopping.
El entrenamiento terminó en la época 3.

## Resultados cuantitativos

  Métrica                             BERT Full       BERT + LoRA
  -------------------------------- ------------ -----------------
  Accuracy en test sintético              100 %             100 %
  F1 macro                               1,0000            1,0000
  F1 `technical`                         1,0000            1,0000
  F1 `cancellation`                      1,0000            1,0000
  Parámetros entrenables             \~109,49 M   2,68 M (2,39 %)
  Tiempo de entrenamiento medido        38,96 s          113,81 s
  Pico de memoria GPU medido         2341,16 MB        1741,60 MB

Ambos modelos clasificaron correctamente los 750 ejemplos sintéticos de
test. LoRA entrenó solo el 2,39 % de los parámetros y registró menor uso
máximo de memoria GPU, aunque fue más lento en esta ejecución.

## Análisis de errores y automatización

En una prueba adicional de 10 ejemplos nuevos, BERT Full acertó 7 (70
%). Los tres errores correspondieron a confusiones entre `general` y
`feature_request`, `cancellation` y `billing`, y `technical` y
`account`. Dos errores presentaron confianza muy alta, por lo que la
confianza softmax no debe interpretarse como garantía de acierto.

Con umbral de confianza \> 0,8, el análisis del test sintético asignó
automáticamente 750/750 tickets (100 %) y no derivó ninguno a revisión
humana. **Esta tasa solo describe el test sintético y no debe
presentarse como una previsión real de producción.**

## ROI proyectado

El enunciado estima el coste manual en 15.000 USD mensuales. Aplicando
hipotéticamente el 100 % de automatización observado, el ahorro bruto
sería de 15.000 USD al mes o 180.000 USD al año. Esta cifra es un
escenario teórico, no un ROI validado: no incluye infraestructura,
supervisión, mantenimiento, revisión humana ni costes derivados de
errores.

## Limitaciones y próximos pasos

La principal limitación es el carácter sintético y relativamente
sencillo del dataset. La diferencia entre el 100 % del test y el 70 % de
la prueba adicional evidencia que las métricas de test no bastan para
afirmar generalización. Antes de producción se recomienda evaluar con
tickets históricos reales anonimizados, mejorar ejemplos y etiquetas
ambiguas, calibrar el umbral de confianza, medir errores de
automatización y ejecutar un piloto con supervisión humana.

**Conclusión:** el reto demuestra el proceso técnico de fine-tuning,
comparación con LoRA y construcción de inferencia. BERT Full es la
opción elegida para este prototipo por el menor tiempo de entrenamiento
observado; LoRA destaca por su eficiencia en parámetros entrenables y
memoria medida. No se recomienda despliegue autónomo sin validación
adicional.
