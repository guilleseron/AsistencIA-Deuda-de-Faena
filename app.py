import streamlit as st
import pandas as pd
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Configuración de página
st.set_page_config(page_title="Consulta de Deuda SAGyP", layout="wide")

# Inicialización del modelo LLM vía LangChain
def init_llm():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        st.error("Error: Variable de entorno GOOGLE_API_KEY no configurada.")
        st.stop()
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

# Carga de datos
@st.cache_data
def load_data(file_path):
    try:
        # Se fuerza el tipo string para identificadores
        dtypes = {
            'establecimiento': str,
            'cuit_titular': str,
            'matricula': str,
            'dte': str
        }
        df = pd.read_excel(file_path, dtype=dtypes)
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo de datos: {e}")
        st.stop()

def main():
    st.title("Consulta de Deuda de Información")
    st.markdown("### Resolución Nº 40/2026 SAGyP")

    llm = init_llm()
    df = load_data("data/deuda.xlsx")

    # Formulario de entrada de datos
    with st.form("consulta_form"):
        cuit_input = st.text_input("Ingrese CUIT (11 cifras, sin guiones):", max_chars=11)
        
        col1, col2 = st.columns(2)
        with col1:
            matricula_input = st.text_input("Ingrese Matrícula RUCA (Opcional si ingresa Establecimiento):", max_chars=6)
        with col2:
            establecimiento_input = st.text_input("Ingrese N° Establecimiento RUCA (Opcional si ingresa Matrícula):", max_chars=6)
        
        submit = st.form_submit_button("Consultar Deuda")

    if submit:
        # Validaciones de entrada
        if not cuit_input or len(cuit_input) != 11 or not cuit_input.isdigit():
            st.warning("Debe ingresar una CUIT válida de 11 dígitos numéricos.")
            return
            
        if not matricula_input and not establecimiento_input:
            st.warning("Debe ingresar la Matrícula RUCA o el Número de Establecimiento.")
            return

        # Filtrado determinista mediante Pandas
        filtro_cuit = df['cuit_titular'] == cuit_input
        
        if matricula_input and establecimiento_input:
            filtro_secundario = (df['matricula'] == matricula_input) | (df['establecimiento'] == establecimiento_input)
        elif matricula_input:
            filtro_secundario = (df['matricula'] == matricula_input)
        else:
            filtro_secundario = (df['establecimiento'] == establecimiento_input)

        resultados = df[filtro_cuit & filtro_secundario]

        # Evaluación de resultados
        if resultados.empty:
            st.success("El usuario detallado no presenta deuda de información con referencia a la Resolución Nº 40/2026 SAGyP")
        else:
            st.error("Se registran deudas de información para el usuario consultado.")
            
            # Selección de columnas requeridas
            columnas_salida = ['dte', 'cabezas', 'especie', 'fecha_ingreso']
            df_salida = resultados[columnas_salida].copy()
            
            # Limpieza visual de fechas
            df_salida['fecha_ingreso'] = pd.to_datetime(df_salida['fecha_ingreso']).dt.strftime('%d/%m/%Y')
            
            # Exposición de datos en formato tabular
            st.dataframe(df_salida, use_container_width=True, hide_index=True)
            
            # Integración de LangChain para síntesis
            prompt_template = PromptTemplate.from_template(
                """
                Actúa como un asistente administrativo formal. 
                Basado en los siguientes datos tabulares que representan deudas de un usuario, redacta un breve párrafo 
                resumiendo la cantidad total de cabezas adeudadas agrupadas por especie.
                
                Datos de deuda:
                {datos_deuda}
                
                Respuesta:
                """
            )
            
            chain = prompt_template | llm
            datos_contexto = df_salida.to_csv(index=False)
            
            with st.spinner("Generando síntesis mediante IA..."):
                respuesta_llm = chain.invoke({"datos_deuda": datos_contexto})
                st.info(respuesta_llm.content)

if __name__ == "__main__":
    main()