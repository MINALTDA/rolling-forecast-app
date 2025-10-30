import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Importar modelos locales
try:
    from models import ModeloMediaMovil, SuavizacaoExponencial, ModeloARIMA, preparar_datos
except ImportError:
    st.error("Error importing models. Please check models.py file.")

st.set_page_config(
    page_title="Rolling Forecast Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header principal
st.markdown('<h1 class="main-header">🎯 Rolling Forecast Tool</h1>', unsafe_allow_html=True)
st.markdown("### 📊 Herramienta con 3 Modelos Estadísticos + Lógica de Lançamento")
st.markdown("---")

# Sidebar para configuración
with st.sidebar:
    st.header("⚙️ Configuración")
    
    # Upload de archivos
    st.subheader("📁 Cargar Archivos")
    uploaded_resumo = st.file_uploader(
        "Arquivo Resumo", 
        type=['xlsx', 'csv'],
        help="Archivo principal con datos de productos y clientes"
    )
    uploaded_logicas = st.file_uploader(
        "Arquivo Lógicas", 
        type=['xlsx', 'csv'],
        help="Archivo con lógicas por mes y clase"
    )
    uploaded_relaciones = st.file_uploader(
        "Arquivo Relações", 
        type=['xlsx', 'csv'],
        help="Archivo con factores por cliente y año"
    )
    
    # Configuración de fecha base
    st.subheader("📅 Configuración Temporal")
    fecha_base = st.date_input(
        "Data Base", 
        datetime.now(),
        help="Fecha base para el cálculo del forecast"
    )
    
    # Selección de modelos
    st.subheader("🔧 Modelos a Ejecutar")
    modelo_media = st.checkbox("📈 Media Móvil (Atual)", True)
    modelo_suavizacao = st.checkbox("📊 Suavização Exponencial", True)
    modelo_arima = st.checkbox("🔬 ARIMA", True)
    
    # Parámetros para nuevos modelos
    st.subheader("⚙️ Parámetros")
    
    if modelo_suavizacao:
        alpha = st.slider(
            "Alpha (Suavização)", 
            0.1, 0.9, 0.3, 0.1,
            help="Factor de suavización (0.1 = más suave, 0.9 = más reactivo)"
        )
    else:
        alpha = 0.3
    
    if modelo_arima:
        st.write("**Parámetros ARIMA (p,d,q):**")
        col1, col2, col3 = st.columns(3)
        with col1:
            p = st.selectbox("p (AR)", [0, 1, 2, 3], 1)
        with col2:
            d = st.selectbox("d (I)", [0, 1, 2], 1)
        with col3:
            q = st.selectbox("q (MA)", [0, 1, 2, 3], 1)
        arima_params = (p, d, q)
    else:
        arima_params = (1, 1, 1)
    
    st.markdown("---")
    st.info("💡 **Nota:** Todos los modelos usan la misma lógica de lançamento cuando es aplicable.")

def main():
    if uploaded_resumo and uploaded_logicas and uploaded_relaciones:
        
        # Cargar datos
        try:
            with st.spinner("Cargando archivos..."):
                # Detectar tipo de archivo y cargar
                if uploaded_resumo.name.endswith('.xlsx'):
                    df_resumo = pd.read_excel(uploaded_resumo)
                else:
                    df_resumo = pd.read_csv(uploaded_resumo)
                
                if uploaded_logicas.name.endswith('.xlsx'):
                    df_logicas = pd.read_excel(uploaded_logicas)
                else:
                    df_logicas = pd.read_csv(uploaded_logicas)
                
                if uploaded_relaciones.name.endswith('.xlsx'):
                    df_relaciones = pd.read_excel(uploaded_relaciones)
                else:
                    df_relaciones = pd.read_csv(uploaded_relaciones)
            
            st.markdown('<div class="success-box">✅ <strong>Archivos cargados exitosamente!</strong></div>', unsafe_allow_html=True)
            
            # Mostrar información de los archivos
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📊 Resumo", f"{df_resumo.shape[0]} filas")
            with col2:
                st.metric("⚙️ Lógicas", f"{df_logicas.shape[0]} filas")
            with col3:
                st.metric("🔗 Relações", f"{df_relaciones.shape[0]} filas")
            
            # Mostrar preview de datos
            with st.expander("👀 Preview dos Dados"):
                tab1, tab2, tab3 = st.tabs(["📊 Resumo", "⚙️ Lógicas", "🔗 Relações"])
                
                with tab1:
                    st.dataframe(df_resumo.head(10), use_container_width=True)
                
                with tab2:
                    st.dataframe(df_logicas.head(10), use_container_width=True)
                
                with tab3:
                    st.dataframe(df_relaciones.head(10), use_container_width=True)
            
            # Validar que al menos un modelo esté seleccionado
            if not any([modelo_media, modelo_suavizacao, modelo_arima]):
                st.warning("⚠️ Por favor, selecciona al menos un modelo para ejecutar.")
                return
            
            # Botón para ejecutar forecast
            if st.button("🚀 Executar Forecast", type="primary", use_container_width=True):
                with st.spinner("Procesando modelos... Esto puede tomar unos minutos."):
                    try:
                        resultados = executar_forecast(
                            df_resumo, df_logicas, df_relaciones, fecha_base,
                            modelo_media, modelo_suavizacao, modelo_arima,
                            alpha, arima_params
                        )
                        
                        if resultados:
                            mostrar_resultados(resultados)
                        else:
                            st.error("❌ No se pudieron generar resultados. Verifica los datos de entrada.")
                    
                    except Exception as e:
                        st.error(f"❌ Error durante el procesamiento: {str(e)}")
                        st.exception(e)
                
        except Exception as e:
            st.error(f"❌ Error al cargar archivos: {str(e)}")
            st.info("💡 Verifica que los archivos tengan el formato correcto.")
    
    else:
        st.info("📤 **Por favor, carga todos los archivos necesarios para comenzar.**")
        
        # Mostrar información sobre los archivos requeridos
        st.markdown("### 📋 Archivos Requeridos:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **📊 Arquivo Resumo**
            - Datos principales de productos
            - Columnas de fechas con valores históricos
            - Información de clientes y clases
            """)
        
        with col2:
            st.markdown("""
            **⚙️ Arquivo Lógicas**
            - Lógicas por clase y mes
            - Configuración de forecast
            - Reglas de lançamento
            """)
        
        with col3:
            st.markdown("""
            **🔗 Arquivo Relações**
            - Factores por cliente
            - Multiplicadores por año
            - Ajustes de crecimiento
            """)

def executar_forecast(df_resumo, df_logicas, df_relaciones, fecha_base, 
                     usar_media, usar_suavizacao, usar_arima,
                     alpha, arima_params):
    
    resultados = {}
    
    try:
        # Preparar datos
        datos_preparados = preparar_datos(df_resumo, df_logicas, df_relaciones, fecha_base)
        
        # Ejecutar modelos seleccionados
        if usar_media:
            with st.spinner("Ejecutando Media Móvil..."):
                modelo_media = ModeloMediaMovil(datos_preparados)
                resultados['media_movil'] = modelo_media.calcular()
        
        if usar_suavizacao:
            with st.spinner("Ejecutando Suavização Exponencial..."):
                modelo_suav = SuavizacaoExponencial(datos_preparados, alpha)
                resultados['suavizacao_exponencial'] = modelo_suav.calcular()
        
        if usar_arima:
            with st.spinner("Ejecutando ARIMA..."):
                modelo_arima = ModeloARIMA(datos_preparados, arima_params)
                resultados['arima'] = modelo_arima.calcular()
        
        return resultados
    
    except Exception as e:
        st.error(f"Error en executar_forecast: {str(e)}")
        return {}

def mostrar_resultados(resultados):
    st.header("📊 Resultados del Forecast")
    
    # Resumen general
    total_celulas = sum([r.get('celulas_atualizadas', 0) for r in resultados.values()])
    st.metric("�� Total de Células Actualizadas", total_celulas)
    
    # Tabs para cada modelo
    tab_names = []
    for modelo in resultados.keys():
        if modelo == 'media_movil':
            tab_names.append("📈 Media Móvil")
        elif modelo == 'suavizacao_exponencial':
            tab_names.append("📊 Suavización Exp.")
        elif modelo == 'arima':
            tab_names.append("🔬 ARIMA")
    
    tabs = st.tabs(tab_names)
    
    for i, (modelo, resultado) in enumerate(resultados.items()):
        with tabs[i]:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Gráfico
                try:
                    fig = crear_grafico(resultado, modelo)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"No se pudo generar el gráfico: {str(e)}")
            
            with col2:
                # Métricas
                st.subheader("📊 Métricas")
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>Células Actualizadas</h4>
                    <h2>{resultado.get('celulas_atualizadas', 0)}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                if 'parametros' in resultado:
                    st.subheader("⚙️ Parámetros")
                    for param, valor in resultado['parametros'].items():
                        st.write(f"**{param}:** {valor}")
                
                # Botón de descarga
                if 'dataframe' in resultado:
                    csv = resultado['dataframe'].to_csv(index=False)
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv,
                        file_name=f"forecast_{modelo}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

def crear_grafico(resultado, modelo):
    """Crear gráfico de resultados"""
    
    if 'dataframe' not in resultado:
        raise ValueError("No hay dataframe en el resultado")
    
    df = resultado['dataframe']
    
    # Crear gráfico simple por ahora
    fig = go.Figure()
    
    # Agregar datos si existen columnas apropiadas
    if 'forecast' in df.columns:
        valores_forecast = df['forecast'].dropna()
        
        fig.add_trace(go.Scatter(
            y=valores_forecast.head(50),  # Primeros 50 valores
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#FF6B6B', width=2)
        ))
    
    fig.update_layout(
        title=f"Resultados - {modelo.replace('_', ' ').title()}",
        xaxis_title="Índice",
        yaxis_title="Valor",
        hovermode='x unified',
        height=400
    )
    
    return fig

if __name__ == "__main__":
    main()