import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from io import StringIO
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor


# Cargar los datasets (limpio y sucio)
df_raw = pd.read_csv('../datos/AB_NYC_2019.csv')  # Dataset sucio
df_clean = pd.read_csv('../datos/Limpio_ABNYC_2019.csv')  # Dataset limpio

# Titulo de la aplicación
st.title('Airbnb NYC')

# Crear un menú lateral para navegar entre las secciones
menu = ["Introducción", "Comparación General", "Distribución de Precios y Reseñas", 
        "Eliminación de Columnas", "Filtrado para Análisis de Datos", 
        "Discretización de Variables", "Mapa de Alojamiento", "Modelado y Evaluación"]

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

# ======================================================
# SECCIÓN 8: MODELADO Y EVALUACIÓN
# ======================================================
if selection == "Modelado y Evaluación":
    st.header("6. Modelado y Evaluación")

    # Usamos el dataset limpio como base
    df = df_clean.copy()

    # --- SUB-MENÚ DE NAVEGACIÓN ---
    opciones_submenu = [
        "6.1 Tipos de datos y columnas",
        "6.2 Distribución de variables",
        "6.3 Relaciones (scatter y mapa)",
        "6.4 Heatmap Correlación",
        "6.5 Preparación y Split",
        "6.6 Regresión Lineal Múltiple",
        "6.7 k-NN Regressor",
        "6.8 Comparativa final"
    ]
    
    sub_seccion = st.sidebar.radio("Navegación Modelado:", opciones_submenu)
    
    # --- Procesamiento 6.4: Mapeos ---
    neighbourhood_map = {'Bronx': 1, 'Staten Island': 2, 'Queens': 3, 'Brooklyn': 4, 'Manhattan': 5}
    room_map = {'Shared room': 1, 'Private room': 2, 'Entire home/apt': 3}

    # Creamos las columnas mapeadas (necesarias para los modelos)
    df['neighbourhood_group_map'] = df['neighbourhood_group'].map(neighbourhood_map)
    df['room_type_map'] = df['room_type'].map(room_map)

    if 'reviews_per_month' in df.columns:
        df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    
    # Limpieza de nulos en mapeos
    df = df.dropna(subset=['neighbourhood_group_map', 'room_type_map'])

    features_to_predict = [
        'neighbourhood_group_map', 'room_type_map', 'longitude', 'latitude'
    ]
    target = 'price'

    # --- Procesamiento 6.6: Split ---
    X = df[features_to_predict]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Procesamiento 6.7: Regresión Lineal (Entrenamiento) ---
    modelo_lr = LinearRegression()
    modelo_lr.fit(X_train, y_train)
    y_pred_lr = modelo_lr.predict(X_test)
    
    # --- Procesamiento 6.8: k-NN (Búsqueda y Entrenamiento) ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    param_grid = {'n_neighbors': np.arange(3, 22, 2)}
    knn_gscv = GridSearchCV(KNeighborsRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
    knn_gscv.fit(X_train_scaled, y_train)
    best_k = knn_gscv.best_params_['n_neighbors']
    
    best_knn = KNeighborsRegressor(n_neighbors=best_k)
    best_knn.fit(X_train_scaled, y_train)
    y_pred_knn = best_knn.predict(X_test_scaled)

    if sub_seccion == "6.1 Tipos de datos y columnas":
        st.subheader("6.1 Tipos de datos y columnas")
        buf = StringIO()
        df.info(buf=buf)
        st.text(buf.getvalue())

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        st.write("**Columnas numéricas:**", numeric_cols)
        st.write("**Columnas categóricas:**", categorical_cols)

    elif sub_seccion == "6.2 Distribución de variables":
        st.subheader("6.2 Distribución de variables")
        
        # --- Introducción General ---
        st.markdown("""
        ### **Distribución de los datos**
        El análisis de distribución permite identificar patrones, sesgos y posibles anomalías dentro del dataset.
        """)
        st.markdown("---")

        # ==========================================
        # 1. PRICE
        # ==========================================
        st.markdown("#### Price")
        st.info("La mayoría de los alojamientos cuestan entre 50 y 150 USD, mientras que una minoría alcanza valores muy altos (300–500 USD). Esto indica presencia de outliers y una distribución no normal.")
        
        if 'price' in df.columns:
            fig = plt.figure(figsize=(8, 4))
            sns.histplot(df['price'].dropna(), kde=True)
            plt.title('Distribución de Price')
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")

        # ==========================================
        # 2. UBICACIÓN (Latitud y Longitud)
        # ==========================================
        st.markdown("#### Latitude y Longitude")
        st.info("Ambas muestran picos marcados que reflejan la concentración de alojamientos en zonas específicas (principalmente Manhattan y Brooklyn). La forma de las curvas indica agrupación geográfica natural.")

        col1, col2 = st.columns(2)
        
        with col1:
            if 'latitude' in df.columns:
                fig = plt.figure(figsize=(6, 4))
                sns.histplot(df['latitude'].dropna(), kde=True)
                plt.title('Distribución de Latitude')
                st.pyplot(fig)
                plt.close(fig)
        
        with col2:
            if 'longitude' in df.columns:
                fig = plt.figure(figsize=(6, 4))
                sns.histplot(df['longitude'].dropna(), kde=True)
                plt.title('Distribución de Longitude')
                st.pyplot(fig)
                plt.close(fig)

        st.markdown("---")

        # ==========================================
        # 3. MINIMUM NIGHTS
        # ==========================================
        st.markdown("#### Minimum Nights")
        st.info("Distribución altamente concentrada en valores pequeños (1–7 noches). Existen valores extremos (100–1200 noches), considerados atípicos por su baja frecuencia.")

        if 'minimum_nights' in df.columns:
            fig = plt.figure(figsize=(8, 4))
            sns.histplot(df['minimum_nights'].dropna(), kde=True)
            plt.title('Distribución de Minimum Nights')
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")

        # ==========================================
        # 4. RESEÑAS (Number of Reviews y Reviews per Month)
        # ==========================================
        st.markdown("#### Reviews")
        st.info("**Number of Reviews:** La mayoría de los alojamientos tienen entre 0 y 20 reseñas.\n\n**Reviews per Month:** Alta concentración en valores cercanos a cero.")

        col3, col4 = st.columns(2)

        with col3:
            if 'number_of_reviews' in df.columns:
                fig = plt.figure(figsize=(6, 4))
                sns.histplot(df['number_of_reviews'].dropna(), kde=True)
                plt.title('Number of Reviews')
                st.pyplot(fig)
                plt.close(fig)
        
        with col4:
            if 'reviews_per_month' in df.columns:
                fig = plt.figure(figsize=(6, 4))
                sns.histplot(df['reviews_per_month'].dropna(), kde=True)
                plt.title('Reviews per Month')
                st.pyplot(fig)
                plt.close(fig)

        st.markdown("---")

        # ==========================================
        # 5. DISPONIBILIDAD
        # ==========================================
        st.markdown("#### Availability 365")
        st.info("Distribución particular con alojamientos con 0 días disponibles y otros con 365 días disponibles. El resto se distribuye de forma uniforme entre ambos extremos.")

        if 'availability_365' in df.columns:
            fig = plt.figure(figsize=(8, 4))
            sns.histplot(df['availability_365'].dropna(), kde=True)
            plt.title('Distribución de Availability 365')
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")

        # ==========================================
        # 6. VARIABLES CATEGÓRICAS (Neighbourhood y Room Type)
        # ==========================================
        st.subheader("Variables Categóricas")

        # Neighbourhood Group
        st.markdown("#### Neighbourhood Group")
        st.info("Manhattan y Brooklyn concentran la mayor parte de los alojamientos.")
        if 'neighbourhood_group' in df.columns:
            fig = plt.figure(figsize=(8, 4))
            sns.histplot(df['neighbourhood_group'].dropna())
            plt.title('Frecuencia de Neighbourhood Group')
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("#### Neighbourhood ")
        st.info("Algunos tienen miles de alojamientos, pero la mayoría presenta muy pocos.")
        if 'neighbourhood' in df.columns:
            fig = plt.figure(figsize=(10, 5))
            top_hoods = df['neighbourhood'].value_counts().index
            sns.countplot(data=df[df['neighbourhood'].isin(top_hoods)], x='neighbourhood', order=top_hoods)
            plt.xticks(rotation=90)
            plt.title('Neighbourhoods')
            st.pyplot(fig)
            plt.close(fig)

        # Room Type
        st.markdown("#### Room Type")
        st.info("Entire home/apt y Private room dominan el mercado.")
        if 'room_type' in df.columns:
            fig = plt.figure(figsize=(8, 4))
            sns.histplot(df['room_type'].dropna())
            plt.title('Frecuencia de Room Type')
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")
        
    elif sub_seccion == "6.3 Relaciones (scatter y mapa)":
        st.subheader("6.3 Relaciones")
        
        st.warning("""
        * **Price vs Number of Reviews:** Los precios bajos suelen tener más reseñas, mientras que los alojamientos caros tienen pocas.
        * **Price vs Availability:** No se observa una relación clara entre disponibilidad y precio, cualquier nivel de disponibilidad puede tener precios bajos o altos.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Precio vs Reseñas**")
            fig1 = plt.figure()
            sns.scatterplot(data=df, x="number_of_reviews", y="price")
            st.pyplot(fig1)
            plt.close(fig1)
        
        with col2:
            st.markdown("**Precio vs Disponibilidad**")
            fig2 = plt.figure()
            sns.scatterplot(data=df, x="availability_365", y="price")
            st.pyplot(fig2)
            plt.close(fig2)

        st.markdown("**Mapa de Alojamientos por Precio**")
        fig3 = plt.figure(figsize=(8, 6))
        sc = plt.scatter(df["longitude"], df["latitude"], c=df["price"], s=8, alpha=0.3, cmap='viridis')
        plt.colorbar(sc, label="Precio")
        plt.xlabel("Longitud"); plt.ylabel("Latitud")
        st.pyplot(fig3)
        plt.close(fig3)
        
    elif sub_seccion == "6.4 Heatmap Correlación":
            st.subheader("Heatmap de Correlación")

            st.markdown("""
            ### Análisis de Correlación Enfocado al Precio
            
            * **room_type_map** con 0.68. Tipo de habitación influye mucho en el precio.
            * **Longitude (-0.44):** La ubicación geográfica es clave, existe una relación negativa considerable.
            * **Latitude (0.14):** Complementa a la longitud para ubicar zonas.
            * **Calculated Host Listings Count (-0.11):** Hosts con muchas propiedades tienden a manejar precios un poco más bajos.
            """)
            # -------------------------

            fig_corr = plt.figure(figsize=(10, 8))
            # Seleccionamos solo numéricas para evitar errores
            numeric_cols2 = df.select_dtypes(include=['int64', 'float64'])
            
            # Creamos el heatmap
            sns.heatmap(numeric_cols2.corr(method='spearman'), annot=True, cmap='coolwarm', fmt=".2f")
            plt.title("Matriz de Correlación (Spearman)")
            st.pyplot(fig_corr)
            plt.close(fig_corr)

    elif sub_seccion == "6.5 Preparación y Split":
        st.subheader("6.5 Split de Datos")
        st.write(f"**Datos de entrenamiento:** {X_train.shape[0]} registros")
        st.write(f"**Datos de prueba:** {X_test.shape[0]} registros")
        
        st.markdown("#### Relación Variables vs Precio (Train/Test)")
        st.markdown("""
        **Observaciones Generales:**
        * **Patrones de Distribución:** Se observa que la mayoría de las ofertas se concentran en rangos de precios bajos y medios, con una menor densidad en precios altos (outliers).
        * **Consistencia Train-Test:** La superposición visual de los puntos rojos y azules en todas las gráficas confirma que el conjunto de prueba es representativo del conjunto de entrenamiento, validando nuestra estrategia de división de datos.
        * **Tendencias:** Las visualizaciones permiten identificar preliminarmente si existe una correlación visual directa (positiva, negativa o nula) entre cada característica (como disponibilidad o número de reseñas) y el precio final.
        """)
        opcion_var = st.selectbox("Elige variable para visualizar:", features_to_predict)
        
        # Generación de la gráfica
        fig_rel = plt.figure(figsize=(8, 5))
        plt.scatter(X_train[opcion_var], y_train, color='r', label='Entrenamiento', alpha=0.3, s=15)
        plt.scatter(X_test[opcion_var], y_test, color='b', label='Prueba', alpha=0.3, s=15)
        plt.legend()
        plt.xlabel(opcion_var)
        plt.ylabel('Precio')
        st.pyplot(fig_rel)
        plt.close(fig_rel)


    elif sub_seccion == "6.6 Regresión Lineal Múltiple":
        st.subheader("6.6 Regresión Lineal Múltiple")
        
        # Cálculos de métricas
        mae = mean_absolute_error(y_test, y_pred_lr)
        mse = mean_squared_error(y_test, y_pred_lr)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred_lr)

        # Visualización de métricas
        col1, col2 = st.columns(2)
        col1.metric("R² (Precisión)", f"{r2:.4f}")
        col2.metric("RMSE (Error)", f"{rmse:.2f}")
        
        st.write(f"**MAE:** {mae:.2f} | **MSE:** {mse:.2f}")
        st.markdown("""
        ---
        #### Interpretación de Resultados del Modelo (Regresión Lineal Múltiple)

        1.  **Evaluación de Errores y Ajuste**
            * **R² (0.0605):** El modelo logra explicar apenas el **6.05%** de la variabilidad de los precios. Este es un resultado bajo que indica que la relación lineal entre las variables seleccionadas y el precio es débil.
            * **RMSE (266.51):** El error estándar de la estimación es de aproximadamente **$266.51 USD**. Esto indica una dispersión considerable en las predicciones respecto a los valores reales.
            * **MAE (77.60):** En términos absolutos, el modelo se equivoca en promedio por **$77.60 USD** por noche.

        2. **Interpretación de Coeficientes**
            * **Coeficiente de Intersección (-32,077.16):** El ajuste matemático comienza con un valor negativo sumamente grande. Esto sugiere que el modelo lineal tiene dificultades para encontrar un punto de partida realista sin las variables independientes, o que la escala de los datos requiere revisión.
            * **Coeficientes de las variables:** Los coeficientes obtenidos **[19.55, 99.24, -372.24, 107.31]** muestran magnitudes variadas. Destaca un coeficiente negativo fuerte (-372.24) que reduce drásticamente el precio, mientras que otros aportan valor positivo. Esto se da por el rango de precios que hay en los datos.
        """)

        # Gráficos de diagnóstico
        st.subheader("Diagnóstico de Residuales")
        residuos = y_test - y_pred_lr
        
        fig_diag = plt.figure(figsize=(10, 8))
        
        # Histograma
        plt.subplot(2, 2, 1)
        sns.histplot(residuos, kde=True)
        plt.title('Histograma Residuales')
        
        # Q-Q Plot
        plt.subplot(2, 2, 2)
        sm.qqplot(residuos, line='s', ax=plt.gca())
        plt.title('Q-Q Plot')
        
        # Scatter Residuales vs Ajustados
        plt.subplot(2, 2, 3)
        sns.scatterplot(x=y_pred_lr, y=residuos)
        plt.axhline(0, c='r', ls='--')
        plt.title('Residuales vs Ajustados')
        
        plt.tight_layout()
        st.pyplot(fig_diag)
        plt.close(fig_diag)


    elif sub_seccion == "6.7 k-NN Regressor":
        st.subheader("6.7 k-NN Regressor")
        
        # Mostrar el mejor parámetro encontrado
        st.success(f"Valor óptimo de k encontrado: {best_k}")
        
        # --- GRÁFICA DE ERROR VS K ---
        mean_errors = -knn_gscv.cv_results_['mean_test_score']
        k_values = param_grid['n_neighbors']
        
        fig_k = plt.figure(figsize=(8, 4))
        plt.plot(k_values, mean_errors, marker='o', ls='--', color='purple')
        plt.axvline(x=best_k, color='red', linestyle='-', label=f'Mejor k ({best_k})')
        plt.xlabel('Número de Vecinos (k)')
        plt.ylabel('Error Cuadrático Medio (MSE)')
        plt.title('Validación Cruzada para encontrar k óptimo')
        plt.legend()
        plt.grid(True, alpha=0.3)
        st.pyplot(fig_k)
        plt.close(fig_k)

        st.info("La gráfica justifica nuestra elección y asegura que el modelo final es el más robusto para la evaluación de métricas.")

        # --- MÉTRICAS DEL MODELO k-NN ---
        st.markdown("### Rendimiento del Modelo k-NN")
        
        r2_knn = r2_score(y_test, y_pred_knn)
        rmse_knn = np.sqrt(mean_squared_error(y_test, y_pred_knn))
        
        col1, col2 = st.columns(2)
        col1.metric("R² k-NN", f"{r2_knn:.4f}")
        col2.metric("RMSE k-NN", f"{rmse_knn:.2f}")

        st.markdown(f"""
        ---
        El $\\text{{R}}^2$ de ${r2_knn:.4f}$ y el $\\text{{RMSE}}$ de ${rmse_knn:.2f}$ establecen el desempeño del modelo k-NN. 
        
        Aunque este rendimiento no supera al de la Regresión Lineal en precisión, sí valida la lógica de similitud local de k-NN y subraya que el precio de un alojamiento en NYC es parcialmente dependiente del consenso de las propiedades más similares que lo rodean (los "vecinos"), lo cual es crucial para analizar patrones en mercados dinámicos.
        """)

    elif sub_seccion == "6.8 Comparativa final":
        st.subheader("6.8 Comparativa Final")
        
        # --- 1. CÁLCULO DE TODAS LAS MÉTRICAS ---
        r2_lr = r2_score(y_test, y_pred_lr)
        rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred_lr))
        mae_lr = mean_absolute_error(y_test, y_pred_lr) # Necesario para la tabla
        
        r2_knn = r2_score(y_test, y_pred_knn)
        rmse_knn = np.sqrt(mean_squared_error(y_test, y_pred_knn))
        mae_knn = mean_absolute_error(y_test, y_pred_knn) # Necesario para la conclusión

        # --- 2. TABLA COMPARATIVA ---
        metricas = pd.DataFrame({
            'Modelo': ['Regresión Lineal', f'k-NN (k={best_k})'],
            'R²': [r2_lr, r2_knn],
            'RMSE': [rmse_lr, rmse_knn],
            'MAE': [mae_lr, mae_knn]
        })
        
        st.table(metricas)
        
        # --- 3. GRÁFICAS DE BARRAS (MÉTRICAS) ---
        fig_comp = plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        sns.barplot(data=metricas, x='Modelo', y='R²')
        plt.title("Comparación R²")
        
        plt.subplot(1, 2, 2)
        sns.barplot(data=metricas, x='Modelo', y='RMSE')
        plt.title("Comparación RMSE")
        
        st.pyplot(fig_comp)
        plt.close(fig_comp)

        # --- 4. GRÁFICAS SCATTER (REAL VS PREDICHO) ---
        st.markdown("---")
        st.markdown("### Análisis Visual de Predicciones")

        fig_scatter = plt.figure(figsize=(14, 6))

        # Subplot 1: Regresión Lineal
        plt.subplot(1, 2, 1)
        # Nota: Usamos y_pred_lr en lugar de y_predic para consistencia
        plt.scatter(y_test, y_pred_lr, alpha=0.3, color='red', label='Predicciones')
        plt.plot([0, 1000], [0, 1000], '--k', lw=2, label='Perfecto') # Línea ideal
        plt.title(f'Regresión Lineal (R² = {r2_lr:.4f})')
        plt.xlabel('Precio Real')
        plt.ylabel('Precio Predicho')
        plt.xlim(0, 1000); plt.ylim(0, 1000) 
        plt.legend()

        # Subplot 2: k-NN
        plt.subplot(1, 2, 2)
        plt.scatter(y_test, y_pred_knn, alpha=0.3, color='blue', label='Predicciones')
        plt.plot([0, 1000], [0, 1000], '--k', lw=2, label='Perfecto') # Línea ideal
        plt.title(f'k-NN (R² = {r2_knn:.4f})')
        plt.xlabel('Precio Real')
        plt.ylabel('Precio Predicho')
        plt.xlim(0, 1000); plt.ylim(0, 1000)
        plt.legend()

        st.pyplot(fig_scatter)
        plt.close(fig_scatter)

        # --- 5. TEXTO DE INTERPRETACIÓN Y CONCLUSIÓN ---
        st.markdown(f"""
        La línea negra punteada representa predicciones perfectas. Puntos cercanos a esta línea indican mejores predicciones.
        Los gráficos de dispersión (Valor Real vs. Valor Predicho) muestran la dispersión de los modelos:

        - **Regresión Lineal**: Las predicciones están muy concentradas alrededor de una línea horizontal (la media), lo que indica una subestimación sistemática en precios altos y una baja capacidad para ajustarse a los patrones de precios.
        - **K-NN**: Muestra una mayor dispersión, lo que significa que logra capturar mejor algunos patrones no lineales en los datos, resultando en predicciones marginalmente mejores. La línea negra punteada representa las predicciones perfectas. La lejanía de los puntos a esta línea en ambos modelos confirma que el $R^2$ es muy bajo.

        ---
        ### Conclusión

        **K-NN (k={best_k}) es el modelo ganador** comparado con la Regresión Lineal, presentando:
        - Mayor capacidad explicativa (**R² = {r2_knn:.4f}**)
        - Menor error de predicción (**RMSE = ${rmse_knn:.2f}**)
        - Mayor precisión promedio (**MAE = ${mae_knn:.2f}**)

        **Observación Importante:**
        A pesar de que K-NN supera a la Regresión Lineal, ambos modelos presentan un **R² bajo (< 0.1)**. Esto sugiere que:
        1.  Las variables actuales no son suficientes para explicar la complejidad del precio.
        2.  Existe una gran variabilidad en los datos que estos modelos no están capturando (posible presencia de *outliers* o necesidad de segmentación).
        3.  Se recomendaría explorar ingeniería de características adicional o modelos más complejos para mejorar la predicción.
        """)