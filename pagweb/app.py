import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar los datasets (limpio y sucio)
df_raw = pd.read_csv('../datos/AB_NYC_2019.csv')  # Dataset sucio
df_clean = pd.read_csv('../datos/Limpio_ABNYC_2019.csv')  # Dataset limpio

# Titulo de la aplicación
st.title('Airbnb NYC')

# Crear un menú lateral para navegar entre las secciones
menu = ["Introducción", "Comparación General", "Distribución de Precios y Reseñas", 
        "Eliminación de Columnas", "Filtrado para Análisis de Datos", 
        "Discretización de Variables", "Mapa de Alojamiento"]

selection = st.sidebar.radio("Ir a:", menu)

# ======================================================
# SECCIÓN 1: Introducción
# ======================================================

if selection == "Introducción":
    st.header("Introducción")
    st.markdown("""
    En esta pagina web desarrollada en streamlit se compara un dataset **sucio** y un dataset **limpio** del conjunto de datos de **Airbnb NYC 2019**. 
    El objetivo es mostrar cómo los procesos de **limpieza** y **transformación** de datos afectan a la calidad y utilidad de la información.
    Se proporcionan visualizaciones de las distribuciones de datos **antes y después de la limpieza** para comparar el impacto de las transformaciones.
    """)

# ======================================================
# SECCIÓN 2: COMPARACIÓN GENERAL
# ======================================================

if selection == "Comparación General":
    st.header("1. Comparación general: Dataset sucio vs limpio")
    st.markdown("""
    En esta sección, compararemos las principales diferencias entre el dataset original (sucio) y el dataset limpio, mostrando:
    - Número de columnas
    - Valores nulos
    """)

    # Información básica de los datasets
    st.subheader("Información básica del Dataset")
    st.write(f"Dimensiones del dataset sucio: {df_raw.shape}")
    st.write(f"Dimensiones del dataset limpio: {df_clean.shape}")

    # Mostrar valores nulos antes y después de la limpieza
    st.subheader("Valores nulos antes y después de la limpieza")
    nulls_before = df_raw.isnull().sum()
    nulls_after = df_clean.isnull().sum()

    compare_nulls = pd.DataFrame({
        'Antes': nulls_before,
        'Después': nulls_after
    }).fillna(0)

    st.write(compare_nulls)

# ======================================================
# SECCIÓN 3: DISTRIBUCIÓN DE PRECIOS Y RESEÑAS
# ======================================================

if selection == "Distribución de Precios y Reseñas":
    st.header("2. Distribución de precios y reseñas")
    st.subheader("Distribución de precios: Antes y después de la limpieza")

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Antes de la limpieza
    sns.histplot(df_raw['price'], kde=True, color="blue", ax=ax[0])
    ax[0].set_title("Distribución de precios (Antes)")
    ax[0].set_xlim(0, 1000)
    ax[0].set_xlabel("Precio (USD)")
    ax[0].set_ylabel("Frecuencia")

    # Después de la limpieza
    sns.histplot(df_clean['price'], kde=True, color="green", ax=ax[1])
    ax[1].set_title("Distribución de precios (Después)")
    ax[1].set_xlim(0, 1000)
    ax[1].set_xlabel("Precio (USD)")
    ax[1].set_ylabel("Frecuencia")

    st.pyplot(fig)

    st.subheader("Distribución de 'reviews_per_month' (Antes y después de la limpieza)")

    fig_reviews, ax_reviews = plt.subplots(1, 2, figsize=(12, 6))

    # Antes de la limpieza
    sns.histplot(df_raw['reviews_per_month'].fillna(0), kde=True, color="gray", ax=ax_reviews[0])
    ax_reviews[0].set_title("Distribución de reviews_per_month (Antes)")
    ax_reviews[0].set_xlim(0, 15)
    ax_reviews[0].set_xlabel("Reviews por mes")
    ax_reviews[0].set_ylabel("Frecuencia")

    # Después de la limpieza
    sns.histplot(df_clean['reviews_per_month'], kde=True, color="orange", ax=ax_reviews[1])
    ax_reviews[1].set_title("Distribución de reviews_per_month (Después)")
    ax_reviews[1].set_xlim(0, 15)
    ax_reviews[1].set_xlabel("Reviews por mes")
    ax_reviews[1].set_ylabel("Frecuencia")

    st.pyplot(fig_reviews)

# ======================================================
# SECCIÓN 4: ELIMINACIÓN DE COLUMNAS IRRELEVANTES
# ======================================================

if selection == "Eliminación de Columnas":
    st.header("3. Eliminación de columnas irrelevantes")
    st.markdown("""
    En esta sección se eliminó la columna **`last_review`** por considerarse redundante y por la gran cantidad de valores faltantes que presentaba. 
    Esta columna registraba la fecha de la última reseña, pero los valores nulos correspondían a los anuncios sin reseñas.
    """)

    # Comparación gráfica antes y después de eliminar la columna
    st.subheader("Impacto de la eliminación de la columna 'last_review'")

    # Histograma de 'last_review' antes de la limpieza (en el dataset sucio)
    fig_last_review, ax_last_review = plt.subplots(1, 2, figsize=(12, 6))

    # Antes de la limpieza (usamos df_raw porque en df_clean la columna ya fue eliminada)
    sns.histplot(df_raw['last_review'].isnull(), kde=False, color="red", ax=ax_last_review[0])
    ax_last_review[0].set_title("Antes - Valores nulos en 'last_review'")
    ax_last_review[0].set_xlabel("Valores nulos")
    ax_last_review[0].set_ylabel("Frecuencia")

    # Después de la limpieza (en df_clean la columna ya no existe, así que no es necesario graficar aquí)
    sns.histplot([False] * len(df_clean), kde=False, color="green", ax=ax_last_review[1])
    ax_last_review[1].set_title("Después - 'last_review' eliminada")
    ax_last_review[1].set_xlabel("No hay datos")
    ax_last_review[1].set_ylabel("Frecuencia")

    st.pyplot(fig_last_review)

# ======================================================
# SECCIÓN 5: FILTRADO PARA ANÁLISIS DE DATOS
# ======================================================

if selection == "Filtrado para Análisis de Datos":
    st.header("4. Filtrado para Análisis de Datos")
    st.markdown("""
    Se aplicaron filtros esenciales para asegurar la validez de la columna **`price`** y segmentar el dataset con distintas columnas. 
    Se verificó que no existan precios iguales o menores a $0. Además, se segmentó el dataset para analizar la distribución de precios (promedio, mínimo y máximo) 
    para el grupo de vecindarios **Manhattan** con tipo de alojamiento **Entire home/apt**.
    """)

    # Comparación de precios para el grupo Manhattan con tipo Entire home/apt
    manhattan_data = df_clean[(df_clean['neighbourhood_group'] == 'Manhattan') & 
                              (df_clean['room_type'] == 'Entire home/apt')]

    st.subheader("Análisis de precios en Manhattan para 'Entire home/apt'")

    fig_manhattan_price, ax_manhattan_price = plt.subplots(figsize=(8, 4))
    sns.histplot(manhattan_data['price'], kde=True, color="purple", ax=ax_manhattan_price)
    ax_manhattan_price.set_title("Distribución de precios en Manhattan (Entire home/apt)")
    ax_manhattan_price.set_xlabel("Precio (USD)")
    ax_manhattan_price.set_ylabel("Frecuencia")

    st.pyplot(fig_manhattan_price)

# ======================================================
# SECCIÓN 6: DISCRETIZACIÓN DE VARIABLES NUMÉRICAS
# ======================================================

if selection == "Discretización de Variables":
    st.header("5. Discretización de Variables Numéricas")
    st.markdown("""
    En esta fase, se transformaron variables numéricas continuas en variables categóricas para simplificar el análisis y facilitar la interpretación de los patrones.
    Las columnas clave como **`price`** y **`minimum_nights`** se agruparon en rangos predefinidos (ej. 'Económico', 'Moderado', 'Corta_Estancia'). 
    Esto permite analizar los datos en grupos lógicos y reducir el impacto de valores atípicos.
    """)

    st.subheader("Distribución de precios discretizados")
    fig_disc_price, ax_disc_price = plt.subplots(figsize=(6, 4))

    price_bins = [0, 50, 100, 200, 500, 10000]
    price_labels = ['Muy_Economico', 'Economico', 'Moderado', 'Caro', 'Muy_Caro']
    df_clean['price_category'] = pd.cut(df_clean['price'], bins=price_bins, labels=price_labels, right=False)

    sns.countplot(x="price_category", data=df_clean, palette="mako", ax=ax_disc_price)
    ax_disc_price.set_title("Frecuencia de categorías de precios discretizados")
    ax_disc_price.set_xlabel("Categoría de precio")
    ax_disc_price.set_ylabel("Frecuencia")

    st.pyplot(fig_disc_price)

# ======================================================
# SECCIÓN 7: MAPA DE ALOJAMIENTOS
# ======================================================

if selection == "Mapa de Alojamiento":
    st.header("(PLUS) Mapa de Alojamiento")
    st.markdown("""
    En esta sección, mostraremos un mapa interactivo con la ubicación de todos los alojamientos en el dataset. Cada punto en el mapa representa un Airbnb.
    """)
    # Mostrar mapa con latitudes y longitudes
    st.subheader("Ubicación de los Airbnb en Nueva York")
    st.map(df_clean[['latitude', 'longitude']])

