# Saber 11 e internet fijo en Colombia

Proyecto de **Herramientas de Big Data** (Universidad de La Sabana). Cruza los resultados del Saber 11 (ICFES, 2017, 2019, 2020 y 2022; ~2,19 millones de estudiantes) con el acceso a internet fijo por municipio (MinTIC) y estudia qué tanto explican el hogar y el internet municipal el puntaje.

**El análisis completo está en [`analisis_saber11_internet.ipynb`](analisis_saber11_internet.ipynb)**, que GitHub muestra con todas las salidas. El mapa interactivo está en [`resultados/mapa_saber11_internet.html`](resultados/mapa_saber11_internet.html) (descárgalo y ábrelo en el navegador).

## Herramientas
Arquitectura medallón (Bronze / Silver / Gold) en Parquet · PySpark (broadcast joins, persist, particionamiento, salting) · DuckDB · Polars · Folium · MongoDB (`2dsphere`, aggregation pipelines) · scikit-learn y Spark MLlib.

## Datos (datos.gov.co)
| Dataset | Fuente | Uso |
|---|---|---|
| `kgxf-xxbe` | ICFES | Saber 11 consolidado |
| `rnvb-vnyh` | ICFES | Saber 11 2020-2 |
| `n48w-gutb` | MinTIC | Accesos de internet fijo por municipio |
| `tvzi-ba2g` | MinTIC | Población DANE 2023 y nombres de municipios |
| `pqwj-3fi4` | MinSalud | Coordenadas de los municipios |

Los datos crudos (~200 MB) no están en el repositorio: el cuaderno los descarga de la API la primera vez y los guarda en `datos/cache/`.

## Cómo correrlo
Requisitos: Python 3.13 (PySpark aún no soporta 3.14), Java 17 o superior y, para la sección de MongoDB, Docker.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name pbd-bigdata --display-name "Python (PBD Big Data)"
docker run -d --name pbd-mongo -p 27017:27017 mongo:7   # opcional, para la sección 9
```

Luego abre el cuaderno, elige el kernel **Python (PBD Big Data)** y ejecuta todo. La primera vez tarda más por la descarga.

## Archivos
- `analisis_saber11_internet.ipynb`: el cuaderno con el EDA, los análisis y las conclusiones.
- `saber11_internet.py`: descarga, limpieza y cruce de datos (lo usa el cuaderno; también se puede correr solo con `python saber11_internet.py --anios 2017 2019 2020 2022 --analizar`).
- `build_notebook.py`: genera el cuaderno.
- `datos/`: tablas municipales finales y R² de los modelos.
- `resultados/`: gráficas y mapa.
