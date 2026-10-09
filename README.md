# Fine-Tuning de un Clasificador de Tickets de Soporte

Proyecto de clasificación de intenciones de soporte técnico mediante
fine-tuning de BERT y comparación con LoRA. Se utiliza un dataset
sintético de 5.000 tickets distribuidos en seis categorías. El proyecto
incluye preparación de datos, entrenamiento, evaluación, análisis de
errores e inferencia con un umbral de confianza para derivar tickets a
revisión humana.

> **Limitación principal:** los resultados sobre el conjunto de test son
> perfectos, pero el dataset es sintético y contiene patrones
> relativamente fáciles de aprender. En una prueba adicional de 10
> ejemplos nuevos, el modelo BERT Full acertó 7. Estos resultados no
> deben interpretarse como rendimiento garantizado sobre tickets reales.

## Objetivos

-   Crear y dividir un dataset sintético en train/validation/test
    (70/15/15).
-   Entrenar un clasificador mediante fine-tuning tradicional de BERT.
-   Comparar el entrenamiento completo con una adaptación LoRA.
-   Evaluar accuracy, F1 macro, métricas por clase y matriz de
    confusión.
-   Examinar ejemplos ambiguos y documentar errores.
-   Proporcionar inferencia reutilizable y derivación por umbral de
    confianza.
-   Estimar automatización y ROI teórico, indicando sus limitaciones.

## Tecnologías

-   Python
-   PyTorch con CUDA
-   Hugging Face Transformers
-   PEFT / LoRA
-   pandas, NumPy y scikit-learn
-   NVIDIA RTX 5070 para entrenamiento

## Categorías y datos

El dataset contiene 5.000 ejemplos:

  Categoría                 Total   Porcentaje
  ------------------- ----------- ------------
  `technical`               1.500         30 %
  `account`                 1.200         24 %
  `billing`                   900         18 %
  `general`                   700         14 %
  `feature_request`           450          9 %
  `cancellation`              250          5 %
  **Total**             **5.000**    **100 %**

División estratificada:

  Partición         Ejemplos   Porcentaje
  --------------- ---------- ------------
  Entrenamiento        3.500         70 %
  Validación             750         15 %
  Test                   750         15 %

Se utilizaron pesos de clase en el entrenamiento de BERT Full para
compensar el desbalance. Los pesos calculados sobre entrenamiento
fueron: `technical=0.5556`, `account=0.6944`, `billing=0.9259`,
`general=1.1905`, `feature_request=1.8519` y `cancellation=3.3333`.

## Experimentos

### BERT Full Fine-Tuning

-   Modelo base: `bert-base-uncased`.
-   Longitud máxima: 128 tokens.
-   Learning rate: `2e-5`.
-   Batch de entrenamiento: 16; evaluación: 32.
-   Máximo de 5 épocas; early stopping con paciencia 2.
-   Scheduler lineal, weight decay `0.01`, semilla 42 y FP16.
-   El entrenamiento se detuvo en la época 3.
-   Tiempo de entrenamiento registrado en la última ejecución: **38,96
    s**.
-   Pico de memoria GPU medido: **2341,16 MB**.
-   Parámetros entrenables: aproximadamente **109,49 M**.

### BERT + LoRA

-   Modelo base: `bert-base-uncased`.
-   Configuración LoRA: `r=16`, `lora_alpha=32`, `lora_dropout=0.05`.
-   Módulos objetivo: `query`, `key`, `value`, `dense`.
-   Parámetros entrenables: **2.683.398**, aproximadamente **2,39 %**
    del total.
-   Tiempo de entrenamiento medido: **113,81 s**.
-   Pico de memoria GPU medido: **1741,60 MB**.

### Comparación de resultados

  Métrica                                 BERT Full   BERT + LoRA
  ------------------------------------ ------------ -------------
  Accuracy en test                           1,0000        1,0000
  F1 macro en test                           1,0000        1,0000
  F1 `technical`                             1,0000        1,0000
  F1 `cancellation`                          1,0000        1,0000
  Parámetros entrenables                 \~109,49 M       2,683 M
  Porcentaje entrenable                       100 %        2,39 %
  Tiempo de entrenamiento registrado        38,96 s      113,81 s
  Pico de memoria GPU                    2341,16 MB    1741,60 MB

Los tiempos corresponden a ejecuciones concretas en el equipo utilizado
y pueden variar entre ejecuciones. LoRA entrenó muchos menos parámetros
y consumió menos memoria GPU medida, pero tardó más en este experimento.
Ambos enfoques alcanzaron las mismas métricas en el test sintético.

**Modelo elegido para inferencia:** `models/bert_full`, por su menor
tiempo de entrenamiento observado y por no necesitar la configuración de
adaptadores LoRA en la función de inferencia.

## Análisis de errores

En el conjunto de test de 750 ejemplos, BERT Full obtuvo accuracy y F1
macro de 1,0000 y una matriz de confusión diagonal sin errores.

Para examinar la generalización se probaron 10 textos adicionales
escritos para esta evaluación. El modelo acertó 7/10 (70 %):

  ---------------------------------------------------------------------------
  Texto resumido   Esperado         Predicho                        Confianza
  ---------------- ---------------- ------------------- ---------------------
  Consulta sobre   `general`        `feature_request`                  0,8881
  documentos                                            
  necesarios para                                       
  registrarse                                           

  Cobro por un     `cancellation`   `billing`                          0,9765
  servicio que se                                       
  desea cancelar                                        

  Web no carga y   `technical`      `account`                          0,9890
  el usuario no                                         
  puede acceder a                                       
  su cuenta                                             
  ---------------------------------------------------------------------------

Estos ejemplos revelan ambigüedades entre categorías. Además, dos
errores tuvieron una confianza superior a 0,97: la confianza softmax no
equivale a una probabilidad calibrada de acierto. Un umbral de confianza
por sí solo no elimina todos los errores.

Mejoras recomendadas: - Evaluar con tickets reales anonimizados y
etiquetados por personas. - Añadir ejemplos diversos y difíciles, en
particular de `cancellation`. - Revisar y armonizar las reglas de
etiquetado para mensajes con varias intenciones. - Calibrar el umbral
con un conjunto de validación real y medir precision/recall de la ruta
automática. - Monitorizar errores y deriva de datos tras el despliegue.

## Inferencia y automatización

`src/inference.py` proporciona `TicketClassifier`, que carga el modelo
una vez y expone `predict(text)`. La función devuelve categoría,
confianza y decisión. Con el umbral configurado, confianza **mayor que
0,8** implica `automatic`; confianza **igual o menor que 0,8** implica
`human_review`.

En los 750 ejemplos del test sintético, el análisis obtuvo: -
Automatizados: 750 (100 %). - Derivados a revisión humana: 0 (0 %). -
Confianza media: 0,9944; mínima: 0,9871; máxima: 0,9957.

Esta cifra es el porcentaje de automatización **sobre ese test
sintético**, no una previsión validada para producción. La prueba
adicional de 10 ejemplos mostró que existen errores fuera de la
distribución sintética del test, incluso con alta confianza.

## ROI teórico

El enunciado del reto plantea un coste de clasificación manual de
**15.000 USD al mes**. Si se aplicara directamente el 100 % observado en
el test sintético, el ahorro bruto teórico sería:

-   Ahorro mensual: 15.000 USD.
-   Ahorro anual: 180.000 USD.

Es un escenario ilustrativo, no un ahorro esperado ni garantizado. No
incluye costes de infraestructura, supervisión, mantenimiento, revisión
humana, errores de clasificación ni cambios operativos. Para estimar ROI
real hacen falta datos de producción y un piloto controlado.

## Estructura relevante

``` text
fine-tuning-proyecto/
├── data/
│   ├── support_tickets.csv
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── src/
│   ├── data_utils.py
│   ├── train.py
│   ├── train_lora.py
│   ├── error_analysis.py
│   ├── inference.py
│   └── automation_analysis.py
├── models/
│   ├── bert_full/
│   └── bert_lora/
├── results/
│   ├── lora_metrics.json
│   └── automation_analysis.csv
├── README.md
└── INFORME_EJECUTIVO.md
```

La estructura anterior enumera los archivos principales utilizados
durante el trabajo; si algún archivo de resultados no existe en el
estado actual del repositorio, debe omitirse o generarse antes de
publicarlo.

## Ejecución

Con el entorno virtual activado y las dependencias instaladas:

``` bash
python src/data_utils.py
python src/train.py
python src/train_lora.py
python src/error_analysis.py
python src/inference.py
python src/automation_analysis.py
```

Los comandos de generación de datos y entrenamiento pueden sobrescribir
datos o modelos existentes. Si solo se quiere probar inferencia, basta
con ejecutar `python src/inference.py` después de disponer de
`models/bert_full`.

## Limitaciones

1.  Dataset sintético; no representa toda la variedad del lenguaje real
    de clientes.
2.  Test sintético perfecto, pero solo 70 % de acierto en los 10
    ejemplos adicionales seleccionados manualmente.
3.  No se ha demostrado que el umbral 0,8 garantice decisiones seguras.
4.  El ROI es teórico y se basa en una tasa de automatización que aún no
    se ha validado con datos reales.
5.  Las mediciones de tiempo y memoria dependen del hardware, versión de
    librerías y configuración.

## Conclusión

El proyecto demuestra el flujo técnico completo de preparación de datos,
fine-tuning tradicional, LoRA, evaluación, análisis cualitativo e
inferencia. BERT Full se selecciona como modelo de inferencia para este
prototipo; LoRA demuestra una reducción sustancial de parámetros
entrenables y de memoria GPU medida. Antes de un uso real, el siguiente
paso prioritario es evaluar y calibrar el sistema con tickets reales
representativos.
