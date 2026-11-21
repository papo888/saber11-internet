#!/usr/bin/env python3
"""
Saber 11 x Internet fijo municipal (MinTIC) - descarga, limpieza, cruce y analisis.

Fuentes (datos.gov.co, API Socrata):
  - kgxf-xxbe  Resultados unicos Saber 11 (consolidado; no trae el 2020-2)
  - rnvb-vnyh  Saber 11 2020-2
  - n48w-gutb  Internet Fijo Accesos por tecnologia y segmento (2017 a 2023-T3)
  - tvzi-ba2g  Accesos Fijos a Internet por municipio y poblacion (se usa solo la poblacion DANE 2023)

Uso:
  python saber11_internet.py                       # descarga y cruza 2019 y 2022
  python saber11_internet.py --analizar            # ademas corre el analisis
  python saber11_internet.py --anios 2017 2019 2020 2022 --analizar
  python saber11_internet.py --poblacion pob.csv   # columnas cod_mcpio, anio, poblacion
"""
from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE = "https://www.datos.gov.co/resource"
DS_SABER = "kgxf-xxbe"
DS_SABER_2020_2 = "rnvb-vnyh"
DS_INTERNET = "n48w-gutb"
DS_POBLACION = "tvzi-ba2g"

DIR_DATOS = Path("datos")
DIR_CACHE = DIR_DATOS / "cache"

# Periodos del consolidado por anio. En 2019 y 2022 el ICFES publica un periodo "4"
# que junta los dos semestres. 2018 y 2021 solo traen calendario B, por eso no estan.
PERIODOS = {
    2017: ["20171", "20172"],
    2019: ["20191", "20194"],
    2020: ["20201"],  # el 2020-2 viene en un dataset aparte
    2022: ["20221", "20224"],
}

COLS_SABER = [
    "periodo", "estu_consecutivo", "estu_cod_reside_mcpio", "cole_cod_mcpio_ubicacion",
    "punt_global", "fami_estratovivienda", "fami_educacionmadre", "fami_educacionpadre",
    "fami_tieneinternet", "fami_tienecomputador", "cole_naturaleza", "cole_area_ubicacion",
    "estu_genero",
]

ESTRATO = {f"Estrato {i}": i for i in range(1, 7)}
ESTRATO["Sin Estrato"] = 0

# Anios de estudio aproximados
EDUCACION = {
    "Ninguno": 0,
    "Primaria incompleta": 3,
    "Primaria completa": 5,
    "Secundaria (Bachillerato) incompleta": 8,
    "Secundaria (Bachillerato) completa": 11,
    "Técnica o tecnológica incompleta": 12,
    "Técnica o tecnológica completa": 14,
    "Educación profesional incompleta": 13.5,
    "Educación profesional completa": 16,
    "Postgrado": 18,
    # "No sabe" y "No Aplica" quedan como NaN
}

SI_NO = {"Si": 1, "No": 0}

VARS_HOGAR = ["estrato", "sin_estrato", "educ_madre", "educ_padre", "internet_casa", "computador_casa"]
VARS_MUNICIPIO = ["penetracion", "pct_fibra", "vel_prom_mbps", "pct_25mbps", "pct_sat_inal"]


# --------------------------------------------------------------------------- API

def soql(dataset: str, params: dict, fmt: str = "csv", reintentos: int = 5) -> pd.DataFrame:
    url = f"{BASE}/{dataset}.{fmt}"
    for i in range(reintentos):
        try:
            r = requests.get(url, params=params, timeout=300)
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text), dtype=str)
        except (requests.RequestException, pd.errors.ParserError) as e:
            if i == reintentos - 1:
                raise
            espera = 5 * (i + 1)
            print(f"   error ({e}); reintento en {espera}s", file=sys.stderr)
            time.sleep(espera)
    raise RuntimeError("inalcanzable")


def descargar_paginado(dataset: str, select: str, where: str, pagina: int = 50000) -> pd.DataFrame:
    partes, offset = [], 0
    while True:
        df = soql(dataset, {"$select": select, "$where": where, "$order": ":id",
                            "$limit": pagina, "$offset": offset})
        partes.append(df)
        offset += len(df)
        print(f"   {offset:,} filas", end="\r", flush=True)
        if len(df) < pagina:
            break
    print()
    return pd.concat(partes, ignore_index=True)


def cache(nombre: str, funcion, refrescar: bool = False) -> pd.DataFrame:
    ruta = DIR_CACHE / f"{nombre}.csv.gz"
    if ruta.exists() and not refrescar:
        return pd.read_csv(ruta, dtype=str)
    DIR_CACHE.mkdir(parents=True, exist_ok=True)
    df = funcion()
    df.to_csv(ruta, index=False)
    return df


def cod5(s: pd.Series) -> pd.Series:
    """Codigo DANE de municipio a 5 digitos con cero inicial ('5001', '5001.0' -> '05001')."""
    s = s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    s = s.where(s.str.fullmatch(r"\d{4,5}"), None)
    return s.str.zfill(5)


# --------------------------------------------------------------------------- Saber 11

def bajar_saber(anio: int, refrescar: bool) -> pd.DataFrame:
    if anio not in PERIODOS:
        raise SystemExit(f"Anio {anio} no soportado para Saber 11. Opciones: {sorted(PERIODOS)}")
    select = ",".join(COLS_SABER)
    periodos = ",".join(f"'{p}'" for p in PERIODOS[anio])

    print(f"Saber 11 {anio}: descargando periodos {PERIODOS[anio]}")
    df = cache(f"saber11_{anio}",
               lambda: descargar_paginado(DS_SABER, select, f"periodo in({periodos})"), refrescar)
    if anio == 2020:
        print("Saber 11 2020-2: descargando dataset aparte")
        df2 = cache("saber11_2020_2",
                    lambda: descargar_paginado(DS_SABER_2020_2, select, "1=1"), refrescar)
        df = pd.concat([df, df2], ignore_index=True)

    n = len(df)
    df = df.drop_duplicates("estu_consecutivo")
    if n != len(df):
        print(f"   {n - len(df):,} filas duplicadas eliminadas (quedan {len(df):,} estudiantes)")
    df["anio"] = anio
    return df


def limpiar_saber(df: pd.DataFrame, mcpio: str) -> pd.DataFrame:
    out = pd.DataFrame({
        "anio": df["anio"].astype(int),
        "periodo": df["periodo"],
        "estu_consecutivo": df["estu_consecutivo"],
        "cod_mcpio": cod5(df["estu_cod_reside_mcpio" if mcpio == "residencia" else "cole_cod_mcpio_ubicacion"]),
        "punt_global": pd.to_numeric(df["punt_global"], errors="coerce"),
        "estrato": df["fami_estratovivienda"].map(ESTRATO),
        "educ_madre": df["fami_educacionmadre"].map(EDUCACION),
        "educ_padre": df["fami_educacionpadre"].map(EDUCACION),
        "internet_casa": df["fami_tieneinternet"].map(SI_NO),
        "computador_casa": df["fami_tienecomputador"].map(SI_NO),
        "colegio_oficial": (df["cole_naturaleza"] == "OFICIAL").astype(float).where(df["cole_naturaleza"].notna()),
        "colegio_rural": (df["cole_area_ubicacion"] == "RURAL").astype(float).where(df["cole_area_ubicacion"].notna()),
        "mujer": df["estu_genero"].map({"F": 1, "M": 0}),
    })
    out["sin_estrato"] = (out["estrato"] == 0).astype(float).where(out["estrato"].notna())

    # Aviso si aparecen etiquetas que el mapeo no conoce (p. ej. si el ICFES las cambia)
    for col, mapa, ignorar in [("fami_estratovivienda", ESTRATO, set()),
                               ("fami_educacionmadre", EDUCACION, {"No sabe", "No Aplica"}),
                               ("fami_tieneinternet", SI_NO, set())]:
        raras = set(df[col].dropna().unique()) - set(mapa) - ignorar
        if raras:
            print(f"   OJO: valores sin mapear en {col}: {sorted(raras)}")

    sin_puntaje = out["punt_global"].isna().sum()
    if sin_puntaje:
        print(f"   {sin_puntaje:,} estudiantes sin puntaje global (se descartan)")
    out = out.dropna(subset=["punt_global"])
    return out[out["punt_global"] > 0]   # puntaje 0 = examen anulado


# --------------------------------------------------------------------------- Internet

def bajar_internet(anio: int, trimestre: int, refrescar: bool) -> pd.DataFrame:
    """Agregados por municipio calculados por la API (pocas filas en vez de millones)."""
    acc = "no_de_accesos::number"
    # Algunos proveedores reportan la velocidad en kbps como si fueran Mbps (p. ej. 12288 = 12 Mbps).
    # Todo valor > 1000 se interpreta como kbps y se divide entre 1024.
    v = "replace(velocidad_bajada, ',', '.')::number"
    vel = f"case({v} > 1000, {v} / 1024, true, {v})"
    select = ", ".join([
        "cod_municipio",
        f"sum({acc}) as accesos",
        f"sum(case(tecnologia like '%FIBER%' OR tecnologia like '%FIBRA%', {acc}, true, 0)) as acc_fibra",
        f"sum({acc} * {vel}) as acc_x_vel",
        f"sum(case({vel} >= 25, {acc}, true, 0)) as acc_25mbps",
        f"sum(case(tecnologia in('SATELITAL','WIFI','WIMAX','OTRAS TECNOLOGÍAS INALÁMBRICAS'), {acc}, true, 0)) as acc_sat_inal",
    ])
    where = f"anno='{anio}' AND trimestre='{trimestre}'"
    print(f"Internet {anio}-T{trimestre}: agregando por municipio en la API")
    df = cache(f"internet_{anio}_T{trimestre}",
               lambda: soql(DS_INTERNET, {"$select": select, "$where": where,
                                          "$group": "cod_municipio", "$limit": 50000}), refrescar)
    if df.empty:
        raise SystemExit(f"No hay datos de internet para {anio}-T{trimestre}.")
    df["anio"] = anio
    return df


def bajar_poblacion(refrescar: bool) -> pd.DataFrame:
    print("Poblacion DANE 2023 (la que trae MinTIC)")
    df = cache("poblacion_2023", lambda: soql(DS_POBLACION, {
        "$select": "codigo_dane_municipio, poblacion_dane",
        "$where": "anno=2023 AND trimestre=4", "$limit": 5000}), refrescar)
    return pd.DataFrame({"cod_mcpio": cod5(df["codigo_dane_municipio"]),
                         "poblacion": pd.to_numeric(df["poblacion_dane"], errors="coerce")})


def indicadores_internet(raw: pd.DataFrame, pob: pd.DataFrame, pob_por_anio: pd.DataFrame | None) -> pd.DataFrame:
    d = raw.copy()
    for c in ["accesos", "acc_fibra", "acc_x_vel", "acc_25mbps", "acc_sat_inal"]:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)
    d["cod_mcpio"] = cod5(d["cod_municipio"])
    d = d.dropna(subset=["cod_mcpio"])

    if pob_por_anio is not None:
        d = d.merge(pob_por_anio, on=["cod_mcpio", "anio"], how="left")
    else:
        d = d.merge(pob, on="cod_mcpio", how="left")

    tot = d["accesos"].replace(0, np.nan)
    d["penetracion"] = 100 * d["accesos"] / d["poblacion"]           # accesos por 100 habitantes
    d["pct_fibra"] = 100 * d["acc_fibra"] / tot
    d["vel_prom_mbps"] = d["acc_x_vel"] / tot                          # ponderada por accesos
    d["pct_25mbps"] = 100 * d["acc_25mbps"] / tot
    d["pct_sat_inal"] = 100 * d["acc_sat_inal"] / tot
    return d[["cod_mcpio", "anio", "accesos", "poblacion"] + VARS_MUNICIPIO]


# --------------------------------------------------------------------------- Cruce

def cruzar(est: pd.DataFrame, net: pd.DataFrame) -> pd.DataFrame:
    df = est.merge(net, on=["cod_mcpio", "anio"], how="left", indicator=True)
    sin = df[df["_merge"] == "left_only"]
    print(f"\nCruce: {len(df) - len(sin):,} de {len(df):,} estudiantes con datos de internet "
          f"({100 * (1 - len(sin) / len(df)):.1f}%)")
    if len(sin):
        top = sin.groupby(["anio", "cod_mcpio"], dropna=False).size().sort_values(ascending=False)
        print(f"   {top.index.get_level_values(1).nunique()} municipios sin cruzar; los que mas estudiantes pierden:")
        print(top.head(10).to_string())
    sin_pob = df["poblacion"].isna() & (df["_merge"] == "both")
    if sin_pob.any():
        print(f"   {sin_pob.sum():,} estudiantes en municipios sin poblacion (penetracion queda NaN)")
    return df.drop(columns="_merge")


def tabla_municipal(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["cod_mcpio", "anio"])
    m = g.agg(n_estudiantes=("punt_global", "size"),
              punt_global_prom=("punt_global", "mean"),
              estrato_prom=("estrato", "mean"),
              educ_madre_prom=("educ_madre", "mean"),
              pct_internet_casa=("internet_casa", "mean"),
              pct_computador_casa=("computador_casa", "mean"),
              pct_colegio_oficial=("colegio_oficial", "mean"),
              pct_colegio_rural=("colegio_rural", "mean"))
    for c in ["pct_internet_casa", "pct_computador_casa", "pct_colegio_oficial", "pct_colegio_rural"]:
        m[c] *= 100
    extra = g[["accesos", "poblacion"] + VARS_MUNICIPIO].first()
    return m.join(extra).reset_index()


# --------------------------------------------------------------------------- Analisis

def analizar(df: pd.DataFrame, muestra: int, semilla: int = 42) -> None:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d = df.dropna(subset=["punt_global"]).copy()
    if muestra and len(d) > muestra:
        d = d.sample(muestra, random_state=semilla)
    anios = pd.get_dummies(d["anio"], prefix="anio", drop_first=True, dtype=float)
    d = pd.concat([d, anios], axis=1)
    vars_anio = list(anios.columns)

    tr, te = train_test_split(d, test_size=0.25, random_state=semilla)
    y_tr, y_te = tr["punt_global"], te["punt_global"]

    # Techo: R2 que se lograria conociendo el promedio exacto de cada municipio-anio.
    # Ninguna variable municipal puede explicar mas que esto.
    prom = tr.groupby(["cod_mcpio", "anio"])["punt_global"].mean().rename("prom")
    techo = te.join(prom, on=["cod_mcpio", "anio"])["prom"].fillna(y_tr.mean())
    r2_techo = r2_score(y_te, techo)

    specs = {
        "Solo hogar": VARS_HOGAR + vars_anio,
        "Solo internet municipal": VARS_MUNICIPIO + vars_anio,
        "Hogar + internet municipal": VARS_HOGAR + VARS_MUNICIPIO + vars_anio,
    }
    modelos = {
        "Lineal": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression()),
        "Boosting": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=semilla),
    }

    print(f"\n=== Analisis: {len(d):,} estudiantes (train {len(tr):,} / test {len(te):,}), anios {sorted(d['anio'].unique())}")
    print(f"Techo municipal (promedio municipio-anio): R2 = {r2_techo:.3f}")
    print("Todas las especificaciones incluyen efectos fijos de anio.\n")

    filas, ajustados = [], {}
    for nm, ctor in modelos.items():
        for ns, cols in specs.items():
            m = ctor().fit(tr[cols], y_tr)
            r2 = r2_score(y_te, m.predict(te[cols]))
            filas.append({"modelo": nm, "especificacion": ns, "R2_test": r2})
            ajustados[(nm, ns)] = m
    res = pd.DataFrame(filas).pivot(index="especificacion", columns="modelo", values="R2_test")
    res = res.loc[list(specs)]
    print("R2 en test:")
    print(res.round(3).to_string())

    base = pd.DataFrame(filas).set_index(["modelo", "especificacion"])["R2_test"]
    print("\nAporte de cada grupo (cuanto cae el R2 del modelo completo al quitarlo):")
    for nm in modelos:
        full = base[(nm, "Hogar + internet municipal")]
        print(f"  {nm:8s}  quitar internet municipal: -{full - base[(nm, 'Solo hogar')]:.3f}   "
              f"quitar hogar: -{full - base[(nm, 'Solo internet municipal')]:.3f}")

    # Coeficientes estandarizados del lineal completo (puntos de Saber 11 por 1 desviacion estandar)
    cols = specs["Hogar + internet municipal"]
    lin = ajustados[("Lineal", "Hogar + internet municipal")]
    coef = pd.Series(lin[-1].coef_, index=cols)
    print("\nLineal completo - puntos por 1 desviacion estandar de cada variable:")
    print(coef.drop(vars_anio).sort_values(key=abs, ascending=False).round(2).to_string())

    res.to_csv(DIR_DATOS / "resultados_r2.csv")
    print(f"\nGuardado: {DIR_DATOS / 'resultados_r2.csv'}")


# --------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--anios", type=int, nargs="+", default=[2019, 2022])
    ap.add_argument("--trimestre", type=int, default=3,
                    help="Trimestre de internet a usar (default 3: el examen de calendario A es en ago-sep)")
    ap.add_argument("--municipio", choices=["residencia", "colegio"], default="residencia",
                    help="Municipio del estudiante para el cruce (default: residencia)")
    ap.add_argument("--poblacion", type=Path,
                    help="CSV con cod_mcpio, anio, poblacion (p. ej. proyecciones DANE por anio)")
    ap.add_argument("--analizar", action="store_true")
    ap.add_argument("--muestra", type=int, default=400000,
                    help="Estudiantes a usar en el analisis (0 = todos)")
    ap.add_argument("--refrescar", action="store_true", help="Ignorar la cache y volver a descargar")
    a = ap.parse_args()

    DIR_DATOS.mkdir(exist_ok=True)

    pob_anio = None
    if a.poblacion:
        pob_anio = pd.read_csv(a.poblacion, dtype={"cod_mcpio": str})
        pob_anio["cod_mcpio"] = cod5(pob_anio["cod_mcpio"])
        pob_anio = pob_anio[["cod_mcpio", "anio", "poblacion"]]
    pob = bajar_poblacion(a.refrescar)

    est = pd.concat([limpiar_saber(bajar_saber(y, a.refrescar), a.municipio) for y in a.anios],
                    ignore_index=True)
    net_raw = pd.concat([bajar_internet(y, a.trimestre, a.refrescar) for y in a.anios], ignore_index=True)
    net = indicadores_internet(net_raw, pob, pob_anio)

    df = cruzar(est, net)
    mun = tabla_municipal(df)

    tag = "_".join(map(str, a.anios))
    f1, f2 = DIR_DATOS / f"saber11_internet_estudiantes_{tag}.csv.gz", DIR_DATOS / f"saber11_internet_municipios_{tag}.csv"
    df.to_csv(f1, index=False)
    mun.to_csv(f2, index=False)
    print(f"\nGuardados:\n  {f1}  ({len(df):,} estudiantes)\n  {f2}  ({len(mun):,} municipio-anio)")

    if a.analizar:
        analizar(df, a.muestra)


if __name__ == "__main__":
    main()
