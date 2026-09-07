import streamlit as st
import pandas as pd
import joblib
import random


# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Monitoreo de Estabilidad Hidráulica",
    page_icon="⚙️",
    layout="wide"
)


# ============================================================
# CARGA DEL MODELO
# ============================================================

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_hidraulico_rf.pkl")


modelo = cargar_modelo()

pipeline = modelo["pipeline"]
features = modelo["features"]
defaults = modelo["defaults"]
minimums = modelo["minimums"]
maximums = modelo["maximums"]


# ============================================================
# FUNCIONES
# ============================================================

def generar_datos_aleatorios():
    """Genera valores aleatorios dentro de los rangos del entrenamiento."""

    for feature in features:

        minimo = float(minimums[feature])
        maximo = float(maximums[feature])

        if minimo == maximo:
            valor = minimo
        else:
            valor = random.uniform(minimo, maximo)

        st.session_state[f"input_{feature}"] = valor


def restablecer_valores():
    """Restablece todos los campos a la mediana del entrenamiento."""

    for feature in features:
        st.session_state[f"input_{feature}"] = float(defaults[feature])


# ============================================================
# INICIALIZACIÓN DE VALORES
# ============================================================

for feature in features:

    key = f"input_{feature}"

    if key not in st.session_state:
        st.session_state[key] = float(defaults[feature])


# ============================================================
# ENCABEZADO
# ============================================================

st.title("⚙️ Monitoreo de Estabilidad de un Sistema Hidráulico")

st.markdown(
    """
    Esta aplicación utiliza un modelo de **Random Forest** optimizado
    mediante validación cruzada para clasificar la condición de
    estabilidad de un sistema hidráulico.

    **Clase 0:** Inestable  
    **Clase 1:** Estable
    """
)

st.divider()


# ============================================================
# INFORMACIÓN DEL MODELO
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Modelo", "Random Forest")

with col2:
    st.metric("Variables", len(features))

with col3:
    st.metric("Balanceo", "SMOTE")

with col4:
    st.metric("F1 en test", "97.76%")


st.divider()


# ============================================================
# GENERACIÓN DE DATOS
# ============================================================

st.subheader("Datos de entrada")

st.info(
    "Puede introducir los valores manualmente o utilizar el botón "
    "de generación aleatoria. Los valores aleatorios se generan "
    "dentro de los rangos observados en los datos de entrenamiento."
)

col1, col2 = st.columns(2)

with col1:

    st.button(
        "🎲 Generar datos aleatorios",
        use_container_width=True,
        on_click=generar_datos_aleatorios
    )

with col2:

    st.button(
        "↩️ Restablecer valores",
        use_container_width=True,
        on_click=restablecer_valores
    )


# ============================================================
# VARIABLES DE ENTRADA
# ============================================================

grupos = {}

for feature in features:

    prefijo = feature.split("_")[0]

    if prefijo not in grupos:
        grupos[prefijo] = []

    grupos[prefijo].append(feature)


for grupo, variables in grupos.items():

    with st.expander(
        f"Variables {grupo}",
        expanded=True
    ):

        columnas = st.columns(2)

        for i, feature in enumerate(variables):

            with columnas[i % 2]:

                st.number_input(
                    feature,
                    key=f"input_{feature}",
                    format="%.6f"
                )


# ============================================================
# PREDICCIÓN
# ============================================================

st.divider()

if st.button(
    "🔍 Predecir estabilidad",
    type="primary",
    use_container_width=True
):

    entrada = pd.DataFrame(
        [
            [
                st.session_state[f"input_{feature}"]
                for feature in features
            ]
        ],
        columns=features
    )

    prediccion = pipeline.predict(entrada)[0]

    probabilidades = pipeline.predict_proba(entrada)[0]

    prob_inestable = probabilidades[0]
    prob_estable = probabilidades[1]


    # --------------------------------------------------------
    # RESULTADO
    # --------------------------------------------------------

    st.subheader("Resultado de la predicción")

    if prediccion == 1:

        st.success(
            "✅ El modelo clasifica el sistema como ESTABLE."
        )

    else:

        st.error(
            "⚠️ El modelo clasifica el sistema como INESTABLE."
        )


    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Probabilidad de inestabilidad",
            f"{prob_inestable:.2%}"
        )

    with col2:

        st.metric(
            "Probabilidad de estabilidad",
            f"{prob_estable:.2%}"
        )


    st.progress(
        float(prob_estable),
        text=f"Probabilidad de estabilidad: {prob_estable:.2%}"
    )


# ============================================================
# INFORMACIÓN DEL PROYECTO
# ============================================================

st.divider()

st.caption(
    "Proyecto Integrador - CRISP-DM | Despliegue de Modelo "
    "de Aprendizaje de Máquina"
)
