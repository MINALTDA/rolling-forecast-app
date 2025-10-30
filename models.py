import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Importaciones opcionales para modelos avanzados
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False

def preparar_datos(df_resumo, df_logicas, df_relaciones, fecha_base):
    """Prepara y limpia los datos para los modelos"""
    
    # Convertir fecha_base a datetime si es necesario
    if isinstance(fecha_base, str):
        fecha_base = datetime.strptime(fecha_base, '%Y-%m-%d')
    elif hasattr(fecha_base, 'date'):
        fecha_base = fecha_base.date()
    
    return {
        'resumo': df_resumo.copy(),
        'logicas': df_logicas.copy(),
        'relaciones': df_relaciones.copy(),
        'fecha_base': fecha_base
    }

class LogicaLancamento:
    """Lógica complementaria para determinar datos históricos basados en mes de lanzamiento"""
    
    def __init__(self, datos):
        self.datos = datos
    
    def es_logica_lancamento(self, classe):
        """Verifica si una clase usa lógica de lanzamiento"""
        try:
            logicas_classe = self.datos['logicas'][
                self.datos['logicas'].iloc[:, 2] == classe  # Asumiendo columna 2 es clase
            ]
            
            if not logicas_classe.empty:
                logica_text = str(logicas_classe.iloc[0, 6]).lower()  # Asumiendo columna 6 es lógica
                return "depende do mês de lançamento" in logica_text or "depende do mes de lancamento" in logica_text
            
            return False
        except Exception:
            return False
    
    def obtener_serie_temporal(self, row):
        """Determina qué datos históricos usar basado en lógica de lanzamiento"""
        
        try:
            # Buscar columnas de mes de lançamento y percentual
            # Asumiendo que están en las últimas columnas como en tu macro
            df_resumo = self.datos['resumo']
            
            # Buscar mes de lançamento (columna AY = 51 en tu macro)
            mes_lancamento = None
            percentual_crescimento = None
            produto_referencia = None
            
            # Intentar encontrar las columnas correctas
            for col_idx, col_name in enumerate(df_resumo.columns):
                if col_idx >= len(row):
                    break
                    
                # Buscar fecha de lanzamiento
                if pd.notna(row.iloc[col_idx]) and isinstance(row.iloc[col_idx], (datetime, pd.Timestamp)):
                    mes_lancamento = row.iloc[col_idx]
                
                # Buscar percentual (número entre 0 y 10 típicamente)
                if pd.notna(row.iloc[col_idx]) and isinstance(row.iloc[col_idx], (int, float)):
                    if 0 < row.iloc[col_idx] <= 10:
                        percentual_crescimento = row.iloc[col_idx]
            
            # Buscar produto de referencia (DEPARA SEGUINTE)
            if len(row) > 7:  # Columna H = índice 7
                produto_referencia = row.iloc[7]
            
            if not all([mes_lancamento, percentual_crescimento, produto_referencia]):
                return np.array([])
            
            # Buscar datos del producto de referencia
            cliente = row.iloc[0] if len(row) > 0 else None
            
            if cliente is None:
                return np.array([])
            
            # Buscar fila del producto de referencia
            produto_ref_mask = (
                (df_resumo.iloc[:, 0] == cliente) &  # Cliente
                (df_resumo.iloc[:, 1] == produto_referencia)  # Produto
            )
            
            produto_ref_rows = df_resumo[produto_ref_mask]
            
            if produto_ref_rows.empty:
                return np.array([])
            
            # Extraer serie temporal del producto de referencia
            produto_ref_row = produto_ref_rows.iloc[0]
            serie_referencia = self.extraer_serie_desde_lancamento(produto_ref_row, mes_lancamento)
            
            # Aplicar factor de crecimiento
            return serie_referencia * percentual_crescimento
            
        except Exception as e:
            return np.array([])
    
    def extraer_serie_desde_lancamento(self, row_referencia, mes_lancamento):
        """Extrae datos históricos desde el mes de lanzamiento hacia atrás"""
        
        try:
            valores = []
            
            # Buscar columnas con fechas
            for col_idx, col_val in enumerate(row_referencia):
                if col_idx < 2:  # Saltar cliente y producto
                    continue
                
                # Intentar encontrar columnas de fechas y valores
                if pd.notna(col_val) and isinstance(col_val, (int, float)) and col_val > 0:
                    valores.append(col_val)
            
            # Tomar últimos 12 valores como máximo
            if len(valores) > 12:
                valores = valores[-12:]
            
            return np.array(valores)
            
        except Exception:
            return np.array([])

class ModeloBase:
    def __init__(self, datos):
        self.datos = datos
        self.logica_lancamento = LogicaLancamento(datos)
    
    def obtener_datos_historicos(self, row):
        """Determina qué datos históricos usar según la lógica"""
        
        try:
            # Obtener clase (asumiendo columna 5 = índice 5)
            classe = row.iloc[5] if len(row) > 5 else None
            
            if classe and self.logica_lancamento.es_logica_lancamento(classe):
                # Usar lógica de lanzamiento
                return self.logica_lancamento.obtener_serie_temporal(row)
            else:
                # Usar datos históricos normales
                return self.extraer_serie_temporal_normal(row)
                
        except Exception:
            return self.extraer_serie_temporal_normal(row)
    
    def extraer_serie_temporal_normal(self, row):
        """Extrae serie temporal normal de la fila"""
        
        try:
            valores = []
            
            # Buscar valores numéricos en la fila (saltando primeras columnas de metadata)
            for col_idx in range(2, len(row)):  # Empezar desde columna 2
                val = row.iloc[col_idx]
                if pd.notna(val) and isinstance(val, (int, float)) and val > 0:
                    valores.append(val)
            
            # Tomar últimos 12 valores como máximo
            if len(valores) > 12:
                valores = valores[-12:]
            
            return np.array(valores)
            
        except Exception:
            return np.array([])
    
    def obtener_factor(self, cliente, año):
        """Obtiene factor desde tabla de relaciones"""
        
        try:
            relaciones = self.datos['relaciones']
            
            # Buscar cliente en primera columna
            cliente_mask = relaciones.iloc[:, 0] == cliente
            cliente_rows = relaciones[cliente_mask]
            
            if cliente_rows.empty:
                return 1.0
            
            # Buscar columna del año
            if año == 2026 and len(cliente_rows.columns) > 1:
                factor = cliente_rows.iloc[0, 1]  # Columna B
            elif año == 2027 and len(cliente_rows.columns) > 2:
                factor = cliente_rows.iloc[0, 2]  # Columna C
            else:
                return 1.0
            
            return float(factor) if pd.notna(factor) else 1.0
            
        except Exception:
            return 1.0
    
    def calcular_precisao(self, df_resultado):
        """Calcula precisión básica"""
        try:
            if 'forecast' in df_resultado.columns:
                forecasts = df_resultado['forecast'].dropna()
                return len(forecasts) / len(df_resultado) if len(df_resultado) > 0 else 0
            return 0
        except Exception:
            return 0

class ModeloMediaMovil(ModeloBase):
    """Modelo de Media Móvil (lógica actual)"""
    
    def calcular(self):
        contador = 0
        df_resultado = self.datos['resumo'].copy()
        df_resultado['forecast'] = np.nan
        
        try:
            for index, row in df_resultado.iterrows():
                # Obtener datos históricos
                serie_historica = self.obtener_datos_historicos(row)
                
                if len(serie_historica) >= 3:
                    # Calcular media móvil de últimos 3 períodos
                    media_movil = np.mean(serie_historica[-3:])
                    
                    # Obtener factor
                    cliente = row.iloc[0] if len(row) > 0 else None
                    año_objetivo = 2026  # Por defecto, podrías calcularlo dinámicamente
                    
                    if cliente:
                        factor = self.obtener_factor(cliente, año_objetivo)
                        forecast_value = media_movil * factor
                        
                        df_resultado.loc[index, 'forecast'] = forecast_value
                        contador += 1
            
            return {
                'dataframe': df_resultado,
                'celulas_atualizadas': contador,
                'modelo': 'Media Móvil',
                'precisao': self.calcular_precisao(df_resultado)
            }
            
        except Exception as e:
            return {
                'dataframe': df_resultado,
                'celulas_atualizadas': 0,
                'modelo': 'Media Móvil',
                'error': str(e),
                'precisao': 0
            }

class SuavizacaoExponencial(ModeloBase):
    """Suavización Exponencial Simple"""
    
    def __init__(self, datos, alpha=0.3):
        super().__init__(datos)
        self.alpha = alpha
    
    def calcular(self):
        if not STATSMODELS_AVAILABLE:
            return {
                'dataframe': self.datos['resumo'].copy(),
                'celulas_atualizadas': 0,
                'modelo': 'Suavización Exponencial',
                'error': 'statsmodels no está disponible',
                'precisao': 0
            }
        
        contador = 0
        df_resultado = self.datos['resumo'].copy()
        df_resultado['forecast'] = np.nan
        
        try:
            for index, row in df_resultado.iterrows():
                serie_historica = self.obtener_datos_historicos(row)
                
                if len(serie_historica) >= 6:  # Mínimo para suavización
                    try:
                        # Aplicar suavización exponencial simple
                        modelo = ExponentialSmoothing(
                            serie_historica, 
                            trend=None, 
                            seasonal=None
                        )
                        fitted_model = modelo.fit(smoothing_level=self.alpha)
                        
                        # Forecast próximo período
                        forecast = fitted_model.forecast(steps=1)[0]
                        
                        if forecast > 0:
                            df_resultado.loc[index, 'forecast'] = forecast
                            contador += 1
                    
                    except Exception:
                        continue
            
            return {
                'dataframe': df_resultado,
                'celulas_atualizadas': contador,
                'modelo': 'Suavización Exponencial',
                'parametros': {'alpha': self.alpha},
                'precisao': self.calcular_precisao(df_resultado)
            }
            
        except Exception as e:
            return {
                'dataframe': df_resultado,
                'celulas_atualizadas': 0,
                'modelo': 'Suavización Exponencial',
                'error': str(e),
                'precisao': 0
            }

class ModeloARIMA(ModeloBase):
    """Modelo ARIMA"""
    
    def __init__(self, datos, params=(1,1,1)):
        super().__init__(datos)
        self.p, self.d, self.q = params
    
    def calcular(self):
        if not ARIMA_AVAILABLE:
            return {
                'dataframe': self.datos['resumo'].copy(),
                'celulas_atualizadas': 0,
                'modelo': 'ARIMA',
                'error': 'statsmodels.tsa.arima no está disponible',
                'precisao': 0
            }
        
        contador = 0
        df_resultado = self.datos['resumo'].copy()
        df_resultado['forecast'] = np.nan
        
        try:
            for index, row in df_resultado.iterrows():
                serie_historica = self.obtener_datos_historicos(row)
                
                if len(serie_historica) >= 12:  # ARIMA necesita más datos
                    try:
                        # Ajustar modelo ARIMA
                        modelo = ARIMA(serie_historica, order=(self.p, self.d, self.q))
                        fitted_model = modelo.fit()
                        
                        # Forecast próximo período
                        forecast = fitted_model.forecast(steps=1)[0]
                        
                        if forecast > 0:
                            df_resultado.loc[index, 'forecast'] = forecast
                            contador += 1
                    
                    except Exception:
                        continue
            
            return {
                'dataframe': df_resultado,
                'celulas_atualizadas': contador,
                'modelo': 'ARIMA',
                'parametros': {'p': self.p, 'd': self.d, 'q': self.q},
                'precisao': self.calcular_precisao(df_resultado)
            }
            
        except Exception as e:
            return {
                'dataframe': df_resultado,
                'celulas_atualizadas': 0,
                'modelo': 'ARIMA',
                'error': str(e),
                'precisao': 0
            }