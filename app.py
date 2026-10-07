import streamlit as st
import pandas as pd
import os
import glob
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuración de página
st.set_page_config(page_title="Gestión SIF y Deuda DNCCA", layout="wide")

# Inicialización del modelo LLM
def init_llm():
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except KeyError:
        st.error("Error: Credencial GOOGLE_API_KEY no configurada.")
        st.stop()
        
    if not api_key or len(api_key) < 10:
         st.error("Error: Credencial inválida.")
         st.stop()

    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite", 
        temperature=0,
        api_key=api_key
    )

# Carga de datos tabulares (Deuda)
@st.cache_data
def load_data():
    try:
        file_id = "1z0PpKAbw37rG-koBo4SieWAquFETP3ce"
        url_descarga = f"https://drive.google.com/uc?export=download&id={file_id}"
        dtypes = {
            'establecimiento': str,
            'cuit_titular': str,
            'matricula': str,
            'dte': str
        }
        return pd.read_excel(url_descarga, dtype=dtypes)
    except Exception as e:
        st.error(f"Error al cargar la base de datos operativa: {e}")
        st.stop()

# Carga de datos documentales y motor RAG (Normativa)
@st.cache_resource
def inicializar_motor_normativo():
    directorio_data = "data"
    archivos_txt = glob.glob(os.path.join(directorio_data, "*.txt"))
    
    if not archivos_txt:
        return None
        
    documentos = []
    for archivo in archivos_txt:
        try:
            loader = TextLoader(archivo, encoding="utf-8")
            documentos.extend(loader.load())
        except Exception:
            continue
            
    if not documentos:
        return None
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    docs_divididos = text_splitter.split_documents(documentos)
    
    # Vectorización local para optimización de recursos
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs_divididos, embeddings)
    
    return vector_store.as_retriever(search_kwargs={"k": 3})

def obtener_cadena_rag(retriever, llm):
    plantilla_prompt = """
    Actúa como un asistente técnico de la Dirección Nacional de Control Comercial Agropecuario (DNCCA).
    Tu función es responder consultas sobre las obligaciones, plazos y penalizaciones del Sistema Integral de Faena.
    
    REGLAS ESTRICTAS DE OPERACIÓN:
    1. Responde ÚNICAMENTE utilizando la información contenida en el "Contexto normativo" provisto.
    2. Si la respuesta a la consulta no se encuentra explícitamente en el contexto, o si la pregunta excede los términos documentados, DEBES responder textualmente con la siguiente directiva, sin agregar preámbulos ni disculpas:
    "La información solicitada no se encuentra en la base documental actual. Ante cualquier duda o requerimiento específico, comuníquese con el área de Gestión de la Información de la Dirección Nacional de Control Comercial Agropecuario (DNCCA) a los teléfonos (011) 4349-2722/29/30, en el horario de 07:00 a 20:00 hs."
    
    Contexto normativo:
    {context}
    
    Consulta del usuario:
    {question}
    
    Respuesta:
    """
    prompt = PromptTemplate.from_template(plantilla_prompt)
    
    def formatear_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
        
    cadena_rag = (
        {"context": retriever | formatear_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return cadena_rag

def main():
    st.title("Sistema Integral de Faena")
    
    llm = init_llm()
    
    # Creación de sistema de pestañas
    tab_operativa, tab_normativa = st.tabs(["Consulta de Deuda (Res. 40/2026)", "Asistencia Normativa SIF"])
    
    # ---------------------------------------------------------
    # PESTAÑA 1: CONSULTA DE DEUDA
    # ---------------------------------------------------------
    with tab_operativa:
        st.markdown("### Verificación de Obligaciones de Información")
        df = load_data()

        with st.form("consulta_form"):
            cuit_input = st.text_input("Ingrese CUIT (11 cifras, sin guiones):", max_chars=11)
            
            col1, col2 = st.columns(2)
            with col1:
                matricula_input = st.text_input("Ingrese Matrícula RUCA (Opcional si ingresa Establecimiento):", max_chars=6)
            with col2:
                establecimiento_input = st.text_input("Ingrese N° Establecimiento RUCA (Opcional si ingresa Matrícula):", max_chars=6)
            
            submit = st.form_submit_button("Consultar Deuda")

        if submit:
            if not cuit_input or len(cuit_input) != 11 or not cuit_input.isdigit():
                st.warning("Debe ingresar una CUIT válida de 11 dígitos numéricos.")
            elif not matricula_input and not establecimiento_input:
                st.warning("Debe ingresar la Matrícula RUCA o el Número de Establecimiento.")
            else:
                filtro_cuit = df['cuit_titular'] == cuit_input
                
                if matricula_input and establecimiento_input:
                    filtro_secundario = (df['matricula'] == matricula_input) | (df['establecimiento'] == establecimiento_input)
                elif matricula_input:
                    filtro_secundario = (df['matricula'] == matricula_input)
                else:
                    filtro_secundario = (df['establecimiento'] == establecimiento_input)

                resultados = df[filtro_cuit & filtro_secundario]

                if resultados.empty:
                    st.success("El usuario detallado no presenta deuda de información con referencia a la Resolución Nº 40/2026 SAGyP.")
                else:
                    st.error("Se registran deudas de información para el usuario consultado.")
                    
                    columnas_salida = ['dte', 'cabezas', 'especie', 'fecha_ingreso']
                    df_salida = resultados[columnas_salida].copy()
                    df_salida['fecha_ingreso'] = pd.to_datetime(df_salida['fecha_ingreso']).dt.strftime('%d/%m/%Y')
                    
                    st.dataframe(df_salida, use_container_width=True, hide_index=True)
                    
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
                    
                    chain_resumen = prompt_template | llm | StrOutputParser()
                    datos_contexto = df_salida.to_csv(index=False)
                    
                    with st.spinner("Generando síntesis de deuda..."):
                        respuesta_resumen = chain_resumen.invoke({"datos_deuda": datos_contexto})
                        st.info(respuesta_resumen)

    # ---------------------------------------------------------
    # PESTAÑA 2: ASISTENCIA NORMATIVA
    # ---------------------------------------------------------
    with tab_normativa:
        st.markdown("### Consultas de Obligaciones y Plazos Legales")
        
        retriever = inicializar_motor_normativo()
        
        if retriever is None:
            st.warning("No se encontraron archivos normativos (TXT) en el directorio de datos. El módulo de consulta se encuentra inactivo.")
        else:
            cadena_rag = obtener_cadena_rag(retriever, llm)
            
            consulta_legal = st.text_area("Describa su consulta respecto al Sistema Integral de Faena:", height=100)
            
            if st.button("Procesar Consulta"):
                if not consulta_legal.strip():
                    st.warning("Por favor, ingrese un texto válido para la consulta.")
                else:
                    with st.spinner("Buscando normativas y generando respuesta..."):
                        respuesta_normativa = cadena_rag.invoke(consulta_legal)
                        st.info(respuesta_normativa)

if __name__ == "__main__":
    main()