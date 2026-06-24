import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf



from utils.data_fetcher import DataFetcher
from utils.risk_metrics import RiskMetrics
from utils.portfolio_optimizer import PortfolioOptimizer
from utils.backtest_engine import BacktestEngine
from strategies.momentum_strategies import (
    RSIMomentumStrategy,
    MACDMomentumStrategy,
    StochasticMomentumStrategy,
    MovingAverageCrossover,
    BollingerBandsStrategy,
    CombinedStrategy
)

from strategies.strategy_registry import strategy_registry

# Configuración de la página
st.set_page_config(
    page_title="Backtest a portafolios",
    page_icon="📈",
    layout="wide"
)

@st.cache_resource
def get_optimizer(train_returns, risk_free_rate):
    """Cachear el optimizador para evitar recrearlo"""
    return PortfolioOptimizer(train_returns, risk_free_rate)

# Título principal
st.title("📈 Analizador de Estrategias de Trading con Optimización de Portafolio")

# Sidebar para configuración
st.sidebar.header("⚙️ Configuración General")

# 1. Selección de Tickers
st.sidebar.subheader("📊 Selección de Activos")
tickers_input = st.sidebar.text_input(
    "Ingresa los tickers separados por comas",
    "AAPL,MSFT,GOOGL,AMZN,TSLA"
)
tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

if len(tickers) < 1 or len(tickers) > 10:
    st.sidebar.error("Debes seleccionar entre 1 y 10 tickers")
    st.stop()

# 2. Configuración de Periodos
st.sidebar.subheader("📅 Periodos de Análisis")

col1, col2 = st.sidebar.columns(2)
with col1:
    train_start = st.date_input(
        "Inicio entrenamiento",
        datetime.now() - timedelta(days=365*3)
    )
with col2:
    train_end = st.date_input(
        "Fin entrenamiento",
        datetime.now() - timedelta(days=365)
    )

test_start = st.sidebar.date_input(
    "Inicio prueba",
    datetime.now() - timedelta(days=365)
)
test_end = st.sidebar.date_input(
    "Fin prueba",
    datetime.now()
)

# Validar periodos
if train_start >= train_end:
    st.sidebar.error("El inicio del entrenamiento debe ser anterior al fin")
    st.stop()

if test_start >= test_end:
    st.sidebar.error("El inicio de la prueba debe ser anterior al fin")
    st.stop()

# Calcular duración en meses
train_months = (train_end.year - train_start.year) * 12 + (train_end.month - train_start.month)
test_months = (test_end.year - test_start.year) * 12 + (test_end.month - test_start.month)

# Validar periodos mínimos y máximos
if train_months < 3:
    st.sidebar.error("El periodo de entrenamiento mínimo es de 3 meses")
    st.stop()
if train_months > 60:  # 5 años
    st.sidebar.error("El periodo de entrenamiento máximo es de 5 años")
    st.stop()
if test_months < 3:
    st.sidebar.error(f"El periodo de prueba mínimo es de 3 meses")
    st.stop()
if test_months > 24:  # 2 años
    st.sidebar.error(f"El periodo de prueba máximo es de 2 años")
    st.stop()

# 3. Tasa libre de riesgo
st.sidebar.subheader("💰 Tasa Libre de Riesgo")
risk_free_rate = st.sidebar.number_input(
    "Tasa libre de riesgo anual (%)",
    min_value=0.0,
    max_value=20.0,
    value=0.0,
    step=0.1
) / 100

# Obtener datos
@st.cache_data
def load_data(tickers, train_start, train_end, test_start, test_end):
    train_data = DataFetcher.fetch_data(tickers, train_start, train_end)
    test_data = DataFetcher.fetch_data(tickers, test_start, test_end)
    return train_data, test_data

try:
    with st.spinner('Cargando datos...'):
        train_data, test_data = load_data(tickers, train_start, train_end, test_start, test_end)
        train_returns = DataFetcher.calculate_returns(train_data)
        test_returns = DataFetcher.calculate_returns(test_data)
    st.success('Datos cargados exitosamente!')
except Exception as e:
    st.error(f"Error al cargar datos: {str(e)}")
    st.stop()

# Tabs principales
tab1, tab2, tab3 = st.tabs(["📊 Optimización de Portafolio", "🎯 Estrategias de Trading", "📈 Backtesting"])

with tab1:
    st.header("Optimización de Portafolio - Frontera Eficiente")
    
    # Inicializar variables en session_state si no existen
    if 'optimization_done' not in st.session_state:
        st.session_state['optimization_done'] = False
        st.session_state['portfolios'] = {}
        st.session_state['frontiers'] = {}
        st.session_state['selected_weights'] = None
        st.session_state['optimizer_params'] = {}
    
    # Verificar si necesitamos recalcular
    current_params = {
        'tickers': tuple(tickers),
        'train_start': train_start,
        'train_end': train_end,
        'risk_free_rate': risk_free_rate
    }
    
    need_recalculation = (
        not st.session_state['optimization_done'] or
        st.session_state.get('optimizer_params', {}) != current_params
    )
    
    # Botón para calcular/actualizar frontera
    col_calc, col_status = st.columns([1, 3])
    with col_calc:
        calculate_button = st.button(
            "🔄 Calcular Frontera Eficiente",
            type="primary",
            disabled=not need_recalculation and st.session_state['optimization_done']
        )
    
    with col_status:
        if st.session_state['optimization_done'] and not need_recalculation:
            st.success("✅ Frontera eficiente calculada y almacenada")
        elif need_recalculation:
            st.warning("⚠️ Cambios detectados - Recalcula la frontera")
    
    if calculate_button or (st.session_state['optimization_done'] and not need_recalculation):
        if calculate_button or need_recalculation:
            with st.spinner('Calculando fronteras eficientes...'):
                try:
                    # Inicializar optimizador
                    optimizer = PortfolioOptimizer(train_returns, risk_free_rate)
                    
                    # Generar frontera con desviación estándar
                    frontier_std = optimizer.generate_efficient_frontier(50, risk_metric='std')
                    
                    # Generar frontera con VaR
                    frontier_var = optimizer.generate_efficient_frontier(50, risk_metric='var')
                    
                    # Generar frontera con CVaR
                    frontier_cvar = optimizer.generate_efficient_frontier(50, risk_metric='cvar')
                    
                    # Calcular portafolios óptimos para cada métrica
                    portfolios = {
                        "Máximo Retorno": optimizer.maximize_return(),
                        "Mínimo Riesgo (Std)": optimizer.minimize_risk(),
                        "Máximo Sharpe Ratio": optimizer.maximize_sharpe_ratio(),
                        "Mínimo VaR": optimizer.minimize_var(),
                        "Mínimo CVaR": optimizer.minimize_cvar(),
                        "Máximo Sharpe-VaR": optimizer.maximize_var_sharpe_ratio(),
                        "Máximo STARR": optimizer.maximize_staar_ratio()
                    }
                    
                    # Almacenar en session_state
                    st.session_state['optimization_done'] = True
                    st.session_state['portfolios'] = portfolios
                    st.session_state['frontiers'] = {
                        'Desviación Estándar': frontier_std,
                        'VaR (Value at Risk)': frontier_var,
                        'CVaR (Conditional VaR)': frontier_cvar
                    }
                    st.session_state['optimizer_params'] = current_params
                    st.session_state['optimizer'] = optimizer
                    
                except Exception as e:
                    st.error(f"Error en optimización: {str(e)}")
                    st.stop()
        
        # Si ya tenemos resultados, mostrarlos
        if st.session_state['optimization_done']:
            portfolios = st.session_state['portfolios']
            frontiers = st.session_state['frontiers']
            optimizer = st.session_state['optimizer']
            
            # Selector de tipo de frontera
            st.subheader("📊 Visualización de Frontera Eficiente")
            
            frontier_type = st.radio(
                "Selecciona la métrica de riesgo para la frontera:",
                list(frontiers.keys()),
                horizontal=True
            )
            
            frontier_df = frontiers[frontier_type]
            
            # Crear dos columnas principales
            col_left, col_right = st.columns([3, 2])
            
            with col_left:
                # Gráfico de frontera eficiente
                fig = go.Figure()
                
                # Frontera eficiente
                fig.add_trace(go.Scatter(
                    x=frontier_df['risk'],
                    y=frontier_df['return'],
                    mode='lines',
                    name=f'Frontera Eficiente ({frontier_type})',
                    line=dict(color='blue', width=2),
                    hovertemplate='Riesgo: %{x:.2%}<br>Retorno: %{y:.2%}<extra></extra>'
                ))
                
                # Portafolios óptimos relacionados con esta frontera
                relevant_portfolios = {
                    "Máximo Retorno": portfolios["Máximo Retorno"],
                    "Mínimo Riesgo (Std)": portfolios["Mínimo Riesgo (Std)"],
                    "Máximo Sharpe Ratio": portfolios["Máximo Sharpe Ratio"]
                }
                
                if 'VaR' in frontier_type:
                    relevant_portfolios = {
                        "Mínimo VaR": portfolios["Mínimo VaR"],
                        "Máximo Sharpe-VaR": portfolios["Máximo Sharpe-VaR"]
                    }
                if 'CVaR' in frontier_type:
                    relevant_portfolios = {
                        "Mínimo CVaR": portfolios["Mínimo CVaR"],
                        "Máximo STARR": portfolios["Máximo STARR"]
                    }
                
                colors = ['red', 'green', 'orange', 'purple', 'brown']
                for i, (name, port) in enumerate(relevant_portfolios.items()):
                    fig.add_trace(go.Scatter(
                        x=[port['risk']],
                        y=[port['return']],
                        mode='markers+text',
                        name=name,
                        text=[name],
                        textposition="top center",
                        marker=dict(size=12, symbol='star', color=colors[i % len(colors)]),
                        hovertemplate=f'{name}<br>Riesgo: %{{x:.2%}}<br>Retorno: %{{y:.2%}}<extra></extra>'
                    ))
                
                fig.update_layout(
                    title=f"Frontera Eficiente - {frontier_type}",
                    xaxis_title="Riesgo",
                    yaxis_title="Retorno Esperado",
                    showlegend=True,
                    height=600,
                    hovermode='closest'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Botón para descargar gráfico
                st.download_button(
                    label="📥 Descargar este gráfico",
                    data=fig.to_html(),
                    file_name=f"frontera_{frontier_type.lower().replace(' ', '_')}.html",
                    mime="text/html"
                )
            
            with col_right:
                # Selección de portafolio
                st.subheader("🎯 Selección de Portafolio")
                
                portfolio_type = st.selectbox(
                    "Elige el portafolio para estrategias:",
                    list(portfolios.keys()),
                    help="Este portafolio se usará en las estrategias de trading"
                )
                
                selected_portfolio = portfolios[portfolio_type]
                st.session_state['selected_weights'] = selected_portfolio['weights']
                
                # Estadísticas del portafolio seleccionado
                st.subheader("📈 Estadísticas del Portafolio")
                
                stats_col1, stats_col2 = st.columns(2)
                
                with stats_col1:
                    st.metric(
                        "Retorno Anualizado",
                        f"{selected_portfolio['return']:.2%}",
                        help="Retorno esperado anualizado del portafolio"
                    )
                    st.metric(
                        "Riesgo",
                        f"{selected_portfolio['risk']:.2%}",
                        help=f"Medido como {frontier_type}"
                    )
                
                with stats_col2:
                    sharpe = (selected_portfolio['return'] - risk_free_rate) / selected_portfolio['risk'] if selected_portfolio['risk'] > 0 else 0
                    st.metric(
                        "Sharpe Ratio",
                        f"{sharpe:.2f}",
                        help="Retorno ajustado por riesgo"
                    )
                    st.metric(
                        "Tasa Libre de Riesgo",
                        f"{risk_free_rate:.2%}"
                    )
                
                # Métricas adicionales específicas
                st.subheader("📊 Métricas Adicionales")
                
                additional_metrics = {}
                for key, value in selected_portfolio.items():
                    if key not in ['weights', 'return', 'risk', 'sharpe_ratio']:
                        if isinstance(value, (int, float)):
                            additional_metrics[key.replace('_', ' ').title()] = value
                
                if additional_metrics:
                    for metric_name, metric_value in additional_metrics.items():
                        st.metric(metric_name, f"{metric_value:.4f}")
            
            # Composición del portafolio (debajo de todo)
            st.markdown("---")
            st.subheader("🏗️ Composición del Portafolio Seleccionado")
            
            # Crear tres columnas para la composición
            comp_col1, comp_col2 = st.columns([1, 1])
            
            with comp_col1:
                # Gráfico de pastel
                weights_data = selected_portfolio['weights']
                # Filtrar solo pesos significativos
                significant_weights = [(ticker, weight) for ticker, weight in zip(tickers, weights_data) if weight > 0.001]
                
                if significant_weights:
                    labels, values = zip(*significant_weights)
                    
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=labels,
                        values=values,
                        hole=0.4,
                        textinfo='label+percent',
                        textposition='outside',
                        marker=dict(colors=colors[:len(labels)])
                    )])
                    
                    fig_pie.update_layout(
                        title="Distribución de Pesos",
                        showlegend=True,
                        height=400
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.info("No hay pesos significativos para mostrar")
            
            with comp_col2:
                # Tabla de pesos
                weights_df = pd.DataFrame({
                    'Activo': tickers,
                    'Peso': weights_data,
                    'Peso %': [f"{w:.2%}" for w in weights_data]
                })
                
                # Ordenar por peso descendente
                weights_df = weights_df.sort_values('Peso', ascending=False)
                
                # Mostrar tabla
                st.dataframe(
                            weights_df.style.format({'Peso': '{:.2%}'}),
                            use_container_width=True
                )
                
                # Estadísticas de concentración
                st.write("**Estadísticas de Concentración:**")
                col_a, col_b = st.columns(2)
                
                with col_a:
                    n_effective = 1 / (weights_df['Peso'] ** 2).sum() if (weights_df['Peso'] ** 2).sum() > 0 else 0
                    st.metric("Nº Efectivo de Activos", f"{n_effective:.1f}")
                
                with col_b:
                    top3_conc = weights_df.nlargest(3, 'Peso')['Peso'].sum()
                    st.metric("Concentración Top 3", f"{top3_conc:.2%}")


with tab2:
    st.header("🎯 Configuración de Estrategias de Trading")
    
    st.subheader("Selecciona y Configura las Estrategias")
    
    # Lista para almacenar estrategias seleccionadas
    if 'selected_strategies' not in st.session_state:
        st.session_state['selected_strategies'] = []
    
    # Obtener estrategias del registro
    all_strategies = strategy_registry.get_all_strategies()
    categories = strategy_registry.get_strategies_by_category()
    
    # Mostrar categorías disponibles
    with st.expander("📋 Ver todas las estrategias disponibles por categoría"):
        for category, strategies in categories.items():
            st.write(f"**{category}** ({len(strategies)} estrategias)")
            for s in strategies:
                info = all_strategies[s]
                st.write(f"  • {s}: {info['description']}")
    
    # Seleccionar estrategias
    base_strategies = {
        k: v for k, v in all_strategies.items() 
        if v.get('category') != 'Combinadas'
    }

    selected_strategies_names = st.multiselect(
        "Selecciona las estrategias a utilizar:",
        list(base_strategies.keys()),  # Solo estrategias base
        default=["RSI", "MACD"] if "RSI" in base_strategies and "MACD" in base_strategies else list(base_strategies.keys())[:2],
    )
    
    # Configurar parámetros de cada estrategia
    configured_strategies = []
    
    if selected_strategies_names:
        st.subheader("Parámetros de Estrategias")
        
        for strategy_name in selected_strategies_names:
            strategy_info = all_strategies[strategy_name]
            
            with st.expander(f"⚙️ {strategy_name} - {strategy_info['description']}"):
                params = {}
                strategy_params = strategy_info['params']
                
                if strategy_params:
                    # Crear columnas para los parámetros (máximo 4 por fila)
                    param_items = list(strategy_params.items())
                    n_params = len(param_items)
                    
                    for i in range(0, n_params, 4):
                        cols = st.columns(min(4, n_params - i))
                        for j, col in enumerate(cols):
                            if i + j < n_params:
                                param_name, param_info = param_items[i + j]
                                with col:
                                    if param_info["type"] == "int":
                                        value = st.number_input(
                                            param_info["description"],
                                            min_value=param_info["min"],
                                            max_value=param_info["max"],
                                            value=param_info["default"],
                                            key=f"tab2_{strategy_name}_{param_name}"
                                        )
                                    elif param_info["type"] == "float":
                                        value = st.number_input(
                                            param_info["description"],
                                            min_value=param_info["min"],
                                            max_value=param_info["max"],
                                            value=param_info["default"],
                                            step=param_info.get("step", 0.1),
                                            key=f"tab2_{strategy_name}_{param_name}"
                                        )
                                    params[param_name] = value
                
                # Crear instancia de la estrategia
                strategy_instance = strategy_registry.get_strategy(strategy_name, **params)
                configured_strategies.append(strategy_instance)
                st.success(f"✅ {strategy_instance.name}")
    
    # Opción de estrategia combinada
    if len(configured_strategies) > 1:
           # Filtrar solo estrategias NO combinadas para crear nueva combinada
        base_only = [s for s in configured_strategies if not hasattr(s, 'combination_type')]
        
        if len(base_only) > 1:
            st.subheader("🔗 Estrategia Combinada")
            st.write("Combina las estrategias seleccionadas con lógica booleana:")
            
            combination_type = st.selectbox(
                "Tipo de combinación:",
                ["AND", "OR", "MAJORITY"],
                help="AND: Todas deben coincidir | OR: Al menos una | MAJORITY: Mayoría de votos"
            )
            
            if st.button("➕ Agregar Estrategia Combinada"):
                combined_strategy = CombinedStrategy(configured_strategies, combination_type)
                
                strategy_registry.register_strategy(
                    name=combined_strategy.name,
                    strategy_class=type(combined_strategy),
                    category="Combinadas",
                    description=f"Estrategia combinada ({combination_type}) - {len(configured_strategies)} estrategias base",
                    params={}  # Sin parámetros adicionales
                )
                
                # También guardar la instancia para poder usarla
                strategy_registry._strategies[combined_strategy.name]['instance'] = combined_strategy            
                    
                configured_strategies.append(combined_strategy)

                st.session_state['configured_strategies'] = configured_strategies
                st.success(f"✅ Estrategia combinada agregada: **{combined_strategy.name}**")
                st.info("La estrategia combinada ya está disponible en el Backtest (Tab 3)")
    
        
    # Guardar estrategias en session_state
    st.session_state['configured_strategies'] = configured_strategies
    
    # Mostrar resumen
    if configured_strategies:
        st.subheader("📋 Resumen de Estrategias Configuradas")
        for i, strategy in enumerate(configured_strategies, 1):
            if hasattr(strategy, 'combination_type'):
                st.write(f"{i}. 🔗 **{strategy.name}** (Combinada {strategy.combination_type})")
            else:
                st.write(f"{i}. ⚙️ **{strategy.name}**")
    else:
        st.info("Selecciona al menos una estrategia para configurar.")


with tab3:
    st.header("📈 Backtesting de Estrategias")
    
    # Verificar que tengamos los datos necesarios
    if 'optimization_done' not in st.session_state or not st.session_state['optimization_done']:
        st.warning("⚠️ Primero calcula la frontera eficiente en la pestaña de Optimización")
    else:
        # Obtener pesos del portafolio
        selected_weights = st.session_state.get('selected_weights', None)
        
        if selected_weights is None:
            st.warning("Selecciona un portafolio en la pestaña de Optimización")
        else:
            # ============================================
            # CONFIGURACIÓN
            # ============================================
            st.subheader("⚙️ Configuración del Backtest")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                initial_capital = st.number_input(
                    "Capital Inicial ($)",
                    min_value=1000,
                    value=10000,
                    step=1000,
                    key="backtest_capital"
                )
            
            with col2:
                commission_fixed = st.number_input(
                    "Comisión Fija ($)",
                    min_value=0.0,
                    value=0.0,
                    step=0.5,
                    key="commission_fixed"
                )
            
            with col3:
                commission_rate = st.number_input(
                    "Comisión Variable (%)",
                    min_value=0.0,
                    max_value=5.0,
                    value=0.1,
                    step=0.05,
                    key="commission_rate"
                ) / 100
            
            try:
                dist_df = pd.DataFrame({
                    'Activo': tickers,
                    'Peso': selected_weights,
                    'Capital': [initial_capital * w for w in selected_weights]
                })
                st.dataframe(
                    dist_df.style.format({
                        'Peso': lambda x: f'{x:.2%}',
                        'Capital': lambda x: f'${x:,.2f}'
                    }),
                    use_container_width=True
                )
            except ValueError:
                st.warning("⚠️ Algunos tickers no tienen datos para el período seleccionado.")
            
            st.markdown("---")
            
            # ============================================
            # SELECCIÓN DE ESTRATEGIA
            # ============================================
            st.subheader("🎯 Estrategia a Evaluar")
            
            strategy_mode = st.radio(
                "Modo:",
                ["📊 Estrategia Estándar", "🎨 Portafolio Personalizado"],
                horizontal=True,
                help="Estándar: misma estrategia para todos los activos. Personalizado: diferente estrategia por activo"
            )
            
            if strategy_mode == "📊 Estrategia Estándar":
                # Elegir una estrategia del registro
                all_strategies = strategy_registry.get_all_strategies()

                
                selected_strategy_name = st.selectbox(
                    "Elige la estrategia a evaluar:",
                    list(all_strategies.keys()),
                    help="Esta estrategia se aplicará a todos los activos"
                )
                
                # Mostrar info de la estrategia
                strategy_info = all_strategies[selected_strategy_name]
                st.caption(f"📝 {strategy_info['description']}")
                

                # Configurar parámetros
                st.write("**Parámetros:**")
                params = {}
                strategy_params = strategy_info['params']
                
                if strategy_params:
                    cols = st.columns(min(len(strategy_params), 4))
                    for i, (param_name, param_info) in enumerate(strategy_params.items()):
                        with cols[i % 4]:
                            if param_info["type"] == "int":
                                params[param_name] = st.number_input(
                                    param_info["description"],
                                    min_value=param_info["min"],
                                    max_value=param_info["max"],
                                    value=param_info["default"],
                                    key=f"eval_{param_name}"
                                )
                            elif param_info["type"] == "float":
                                params[param_name] = st.number_input(
                                    param_info["description"],
                                    min_value=param_info["min"],
                                    max_value=param_info["max"],
                                    value=param_info["default"],
                                    step=param_info.get("step", 0.1),
                                    key=f"eval_{param_name}"
                                )
                
                
                                # Crear la estrategia
                if 'instance' in strategy_info:
                    eval_strategy = strategy_info['instance']
                else:
                    eval_strategy = strategy_registry.get_strategy(selected_strategy_name, **params)

                st.success(f"✅ Estrategia: **{eval_strategy.name}**")
                asset_strategies = {ticker: eval_strategy for ticker in tickers}


                
                # Aplicar a todos los activos
                asset_strategies = {ticker: eval_strategy for ticker in tickers}
                
            else:
                # Modo personalizado
                st.write("**Configura estrategias por activo:**")
                
                all_strategies = strategy_registry.get_all_strategies()


                asset_strategies = {}
                
                for ticker in tickers:
                    with st.expander(f"📊 {ticker} - ${initial_capital * selected_weights[tickers.index(ticker)]:,.2f}"):
                        strat_name = st.selectbox(
                            f"Estrategia para {ticker}",
                            list(all_strategies.keys()),
                            key=f"custom_{ticker}"
                        )
                        
                        strat_info = all_strategies[strat_name]
                        params = {}
                        strat_params = strat_info['params']
                        
                        if strat_params:
                            cols = st.columns(min(len(strat_params), 3))
                            for i, (param_name, param_info) in enumerate(strat_params.items()):
                                with cols[i % 3]:
                                    if param_info["type"] == "int":
                                        params[param_name] = st.number_input(
                                            param_info["description"],
                                            min_value=param_info["min"],
                                            max_value=param_info["max"],
                                            value=param_info["default"],
                                            key=f"custom_{ticker}_{param_name}"
                                        )
                                    elif param_info["type"] == "float":
                                        params[param_name] = st.number_input(
                                            param_info["description"],
                                            min_value=param_info["min"],
                                            max_value=param_info["max"],
                                            value=param_info["default"],
                                            step=param_info.get("step", 0.1),
                                            key=f"custom_{ticker}_{param_name}"
                                        )
                        
                        # AQUI EL CAMBIO:
                        if 'instance' in strat_info:
                            asset_strategies[ticker] = strat_info['instance']
                        else:
                            asset_strategies[ticker] = strategy_registry.get_strategy(strat_name, **params)


            st.markdown("---")
            

                                # Inicializar engine
            backtest_engine = BacktestEngine(
                initial_capital=initial_capital,
                commission_fixed=commission_fixed,
                commission_rate=commission_rate
                    )
            # ============================================
            # EJECUTAR BACKTEST
            # ============================================
            if st.button("🚀 Ejecutar Backtest", type="primary", use_container_width=True):
                with st.spinner('Ejecutando backtest...'):
                    

                    # ============================================
                    # 1. CALCULAR RENDIMIENTOS INDIVIDUALES
                    # ============================================
                    individual_results = {}

                    for ticker in tickers:
                        weight = selected_weights[tickers.index(ticker)]
                        
                        # Si el peso es 0%, saltar este activo
                        if weight < 0.0001:
                            zero_series = pd.Series(0.0, index=test_returns.index)
                            individual_results[ticker] = {
                                'strategy_returns': zero_series.copy(),
                                'asset_returns': zero_series.copy(),
                                'strategy_name': 'Sin asignación (0%)'
                            }
                            continue
                        
                        asset_capital = initial_capital * weight
                        
                        # Obtener datos del activo
                        if isinstance(test_data.columns, pd.MultiIndex):
                            asset_data = test_data.xs(ticker, level=1, axis=1)
                        else:
                            asset_data = test_data[[ticker]]
                        
                        # Obtener precios de cierre
                        if isinstance(asset_data.columns, pd.MultiIndex):
                            close_prices = asset_data['Close'].iloc[:, 0]
                        elif 'Close' in asset_data.columns:
                            close_prices = asset_data['Close']
                        else:
                            close_prices = asset_data.iloc[:, 0]
                        
                        # Rendimientos del activo
                        asset_returns = close_prices.pct_change().fillna(0)
                        
                        # --- Estrategia evaluada ---
                        strategy = asset_strategies[ticker]
                        signals = strategy.generate_signals(asset_data)
                        signals = signals.reindex(asset_returns.index, fill_value=0)
                        
                        # Rendimientos de la estrategia
                        strategy_returns = asset_returns * signals
                        
                        # Costos
                        signal_changes = signals.diff().abs()
                        signal_changes.iloc[0] = abs(signals.iloc[0])
                        fixed_costs = signal_changes * commission_fixed / asset_capital
                        variable_costs = signal_changes * commission_rate * abs(strategy_returns)
                        strategy_returns = strategy_returns - fixed_costs - variable_costs
                        
                        individual_results[ticker] = {
                            'strategy_returns': strategy_returns * weight,
                            'asset_returns': asset_returns * weight,
                            'strategy_name': strategy.name
                        }

                    # ============================================
                    # 2. PORTAFOLIO = SUMA DE INDIVIDUALES
                    # ============================================
                    portfolio_strategy_returns = pd.Series(0.0, index=test_returns.index)
                    portfolio_benchmark_returns = pd.Series(0.0, index=test_returns.index)

                    for ticker in tickers:
                        portfolio_strategy_returns += individual_results[ticker]['strategy_returns']
                        portfolio_benchmark_returns += individual_results[ticker]['asset_returns']

                    # Métricas del portafolio
                    portfolio_strategy_metrics = backtest_engine.calculate_metrics(
                        portfolio_strategy_returns, "Estrategia Principal"
                    )
                    portfolio_benchmark_metrics = backtest_engine.calculate_metrics(
                        portfolio_benchmark_returns, "Buy & Hold"
                    )

                    # Ajustar retornos acumulados al capital
                    portfolio_strategy_metrics['cumulative_returns'] = (
                        (1 + portfolio_strategy_returns).cumprod() * initial_capital
                    )
                    portfolio_benchmark_metrics['cumulative_returns'] = (
                        (1 + portfolio_benchmark_returns).cumprod() * initial_capital
                    )

                    # ============================================
                    # 3. BENCHMARKS FIJOS (MACD y Bollinger)
                    # ============================================
                    benchmark_strategies = {
                        'MACD': MACDMomentumStrategy(fast=12, slow=26, signal=9),
                        'Bollinger Bands': BollingerBandsStrategy(period=20, std_dev=2.0)
                    }

                    benchmark_returns_dict = {
                        'Buy & Hold': portfolio_benchmark_returns
                    }

                    for bench_name, bench_strategy in benchmark_strategies.items():
                        bench_portfolio_returns = pd.Series(0.0, index=test_returns.index)
                        
                        for ticker in tickers:
                            weight = selected_weights[tickers.index(ticker)]
                            
                            if weight < 0.0001:
                                continue  # Saltar activos sin peso
                            
                            asset_capital = initial_capital * weight
                            
                            if isinstance(test_data.columns, pd.MultiIndex):
                                asset_data = test_data.xs(ticker, level=1, axis=1)
                            else:
                                asset_data = test_data[[ticker]]
                            
                            if isinstance(asset_data.columns, pd.MultiIndex):
                                close_prices = asset_data['Close'].iloc[:, 0]
                            elif 'Close' in asset_data.columns:
                                close_prices = asset_data['Close']
                            else:
                                close_prices = asset_data.iloc[:, 0]
                            
                            asset_returns = close_prices.pct_change().fillna(0)
                            signals = bench_strategy.generate_signals(asset_data)
                            signals = signals.reindex(asset_returns.index, fill_value=0)
                            
                            bench_asset_returns = asset_returns * signals
                            
                            # Costos
                            signal_changes = signals.diff().abs()
                            signal_changes.iloc[0] = abs(signals.iloc[0])
                            fixed_costs = signal_changes * commission_fixed / asset_capital
                            variable_costs = signal_changes * commission_rate * abs(bench_asset_returns)
                            bench_asset_returns = bench_asset_returns - fixed_costs - variable_costs
                            
                            bench_portfolio_returns += bench_asset_returns * weight
                        
                        benchmark_returns_dict[bench_name] = bench_portfolio_returns

                    # ============================================
                    # 4. GUARDAR RESULTADOS
                    # ============================================
                    st.session_state['backtest_results'] = {
                        'individual_results': individual_results,
                        'portfolio_strategy_returns': portfolio_strategy_returns,
                        'portfolio_strategy_metrics': portfolio_strategy_metrics,
                        'portfolio_benchmark_metrics': portfolio_benchmark_metrics,
                        'benchmark_returns': benchmark_returns_dict,
                        'asset_strategies': asset_strategies,
                        'strategy_mode': strategy_mode
                    }

                    st.success('✅ Backtest completado!')
                    st.rerun()
            
            # ============================================
            # MOSTRAR RESULTADOS
            # ============================================
            if 'backtest_results' in st.session_state and st.session_state['backtest_results']:
                results = st.session_state['backtest_results']
                
                st.markdown("---")
                st.header("📊 Resultados del Backtest")
                
                # ============================================
                # GRÁFICO PRINCIPAL: TODAS LAS TRAYECTORIAS
                # ============================================
                st.subheader("📈 Rendimiento del Portafolio")
                
                fig_main = go.Figure()
                
                # Estrategia principal (destacada)
                cum_strategy = results['portfolio_strategy_metrics']['cumulative_returns']
                strategy_label = "🎯 Estrategia Principal"
                
                fig_main.add_trace(go.Scatter(
                    x=cum_strategy.index,
                    y=cum_strategy,
                    mode='lines',
                    name=strategy_label,
                    line=dict(color='#ff1493', width=4)
                ))
                
                # Benchmarks
                benchmark_colors = {
                    'Buy & Hold': '#7f7f7f',
                    'MACD': '#1f77b4',
                    'Bollinger Bands': '#2ca02c'
                }
                
                for bench_name, bench_returns in results['benchmark_returns'].items():
                    cum_bench = (1 + bench_returns).cumprod() * initial_capital
                    
                    fig_main.add_trace(go.Scatter(
                        x=cum_bench.index,
                        y=cum_bench,
                        mode='lines',
                        name=f"📊 {bench_name}",
                        line=dict(
                            color=benchmark_colors.get(bench_name, 'gray'),
                            width=2,
                            dash='dash' if bench_name == 'Buy & Hold' else 'solid'
                        ),
                        opacity=0.8
                    ))
                
                fig_main.add_hline(y=initial_capital, line_dash="dot", line_color="gray", opacity=0.5)
                
                fig_main.update_layout(
                    title="Comparación: Estrategia Principal vs Benchmarks",
                    xaxis_title="Fecha",
                    yaxis_title="Capital ($)",
                    height=500,
                    hovermode='x unified',
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                
                st.plotly_chart(fig_main, use_container_width=True)
                
                # ============================================
                # MÉTRICAS COMPARATIVAS
                # ============================================
                st.subheader("📊 Métricas Comparativas")
                
                all_metrics = []
                
                # Estrategia principal
                metrics = results['portfolio_strategy_metrics']
                all_metrics.append({
                    'Estrategia': '🎯 Estrategia Principal',
                    'Retorno Total': metrics['total_return'],
                    'Retorno Anual': metrics['annual_return'],
                    'Volatilidad': metrics['annual_volatility'],
                    'Sharpe Ratio': metrics['sharpe_ratio'],
                    'Max Drawdown': metrics['max_drawdown'],
                    'Win Rate': metrics['win_rate'],
                    'Profit Factor': metrics['profit_factor']
                })
                
                # Benchmarks
                for bench_name, bench_returns in results['benchmark_returns'].items():
                    bench_metrics = backtest_engine.calculate_metrics(bench_returns, bench_name)
                    all_metrics.append({
                        'Estrategia': f'📊 {bench_name}',
                        'Retorno Total': bench_metrics['total_return'],
                        'Retorno Anual': bench_metrics['annual_return'],
                        'Volatilidad': bench_metrics['annual_volatility'],
                        'Sharpe Ratio': bench_metrics['sharpe_ratio'],
                        'Max Drawdown': bench_metrics['max_drawdown'],
                        'Win Rate': bench_metrics['win_rate'],
                        'Profit Factor': bench_metrics['profit_factor']
                    })
                
                metrics_df = pd.DataFrame(all_metrics)
                
                # Formatear
                display_df = metrics_df.copy()
                pct_cols = ['Retorno Total', 'Retorno Anual', 'Volatilidad', 'Max Drawdown', 'Win Rate']
                for col in pct_cols:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
                display_df['Sharpe Ratio'] = display_df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}")
                display_df['Profit Factor'] = display_df['Profit Factor'].apply(
                    lambda x: f"{x:.2f}" if x != float('inf') else "∞"
                )
                
                def highlight_main(row):
                    try:
                        if '🎯' in str(row['Estrategia']):
                            return ['background-color: #ffe6f0; font-weight: bold; color: #ff1493'] * len(row)
                    except:
                        pass
                    return [''] * len(row)

                # Asegurar que no haya valores problemáticos
                display_df = display_df.fillna('N/A')
                display_df = display_df.replace([float('inf'), float('-inf')], '∞')

                st.dataframe(
                    display_df.style.apply(highlight_main, axis=1),
                    use_container_width=True
                )
                
                # ============================================
                # RENDIMIENTO POR ACTIVO
                # ============================================
                st.subheader("💼 Rendimiento por Activo")
                
                # Selector de activo
                selected_asset = st.selectbox("Selecciona un activo:", tickers)
                
                if selected_asset in results['individual_results']:
                    asset_data = results['individual_results'][selected_asset]
                    
                    col_a1, col_a2 = st.columns(2)
                    
                    with col_a1:
                        st.write(f"**Estrategia:** {asset_data['strategy_name']}")
                        
                        # Métricas individuales
                        ind_metrics = backtest_engine.calculate_metrics(
                            asset_data['strategy_returns'] / selected_weights[tickers.index(selected_asset)],
                            selected_asset
                        )
                        
                        st.metric("Retorno Total", f"{ind_metrics['total_return']:.2%}")
                        st.metric("Sharpe Ratio", f"{ind_metrics['sharpe_ratio']:.2f}")
                        st.metric("Max Drawdown", f"{ind_metrics['max_drawdown']:.2%}")
                    
                    with col_a2:
                        # Gráfico del activo
                        cum_asset_strategy = (1 + asset_data['strategy_returns'] / selected_weights[tickers.index(selected_asset)]).cumprod()
                        cum_asset_bh = (1 + asset_data['asset_returns'] / selected_weights[tickers.index(selected_asset)]).cumprod()
                        
                        fig_asset = go.Figure()
                        fig_asset.add_trace(go.Scatter(
                            x=cum_asset_strategy.index,
                            y=cum_asset_strategy,
                            mode='lines',
                            name='Estrategia',
                            line=dict(color='#ff1493', width=2)
                        ))
                        fig_asset.add_trace(go.Scatter(
                            x=cum_asset_bh.index,
                            y=cum_asset_bh,
                            mode='lines',
                            name='Buy & Hold',
                            line=dict(color='gray', width=2, dash='dash')
                        ))
                        fig_asset.update_layout(
                            title=f"Rendimiento - {selected_asset}",
                            height=300,
                            showlegend=True
                        )
                        st.plotly_chart(fig_asset, use_container_width=True)
                
                # ============================================
                # RESUMEN FINAL
                # ============================================
                st.subheader("📋 Resumen Final")
                
                final_capital = initial_capital * (1 + results['portfolio_strategy_metrics']['total_return'])
                bh_capital = initial_capital * (1 + results['portfolio_benchmark_metrics']['total_return'])
                
                col_f1, col_f2, col_f3 = st.columns(3)
                
                with col_f1:
                    st.metric(
                        "Capital Final (Estrategia)",
                        f"${final_capital:,.2f}",
                        f"{results['portfolio_strategy_metrics']['total_return']:.2%}"
                    )
                
                with col_f2:
                    st.metric(
                        "Capital Final (Buy & Hold)",
                        f"${bh_capital:,.2f}",
                        f"{results['portfolio_benchmark_metrics']['total_return']:.2%}"
                    )
                
                with col_f3:
                    diferencia = final_capital - bh_capital
                    st.metric(
                        "Diferencia",
                        f"${diferencia:,.2f}",
                        f"{diferencia/bh_capital:.2%}" if bh_capital > 0 else "N/A"
                    )

# Footer
st.markdown("---")
st.markdown("### 📋 Resumen de Configuración")

try:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Activos:** {', '.join(tickers)}")
    with col2:
        st.write(f"**Periodo Entrenamiento:** {train_start} a {train_end}")
    with col3:
        st.write(f"**Periodo Prueba:** {test_start} a {test_end}")
except Exception as e:
    st.warning("⚠️ No se pudo mostrar el resumen completo. Algunos datos pueden estar incompletos.")