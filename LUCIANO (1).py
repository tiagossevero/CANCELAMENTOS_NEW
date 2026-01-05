"""
================================================================================
PROJETO LUCIANO - Dashboard de Machine Learning para Cancelamento de IE
================================================================================
Sistema de identificação de empresas ATIVAS com perfil similar às canceladas
Desenvolvido para a Receita Estadual de Santa Catarina

Versão: 1.1
Data: Janeiro 2026

CORREÇÕES v1.1:
- Filtro de empresas ATIVAS em todas as queries e páginas
- Rankings e alertas agora mostram APENAS empresas ativas (flag_atualmente_cancelada = 0)
- Função carregar_empresas_ativas() corrigida com LEFT JOIN
- Empresas já canceladas são excluídas das análises de risco
================================================================================
"""

import streamlit as st
import hashlib

# =============================================================================
# CONFIGURAÇÃO DE AUTENTICAÇÃO
# =============================================================================

SENHA = "luciano2025"  # Altere conforme necessário

def check_password():
    """Verifica a senha de acesso ao sistema."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.markdown("""
            <div style='text-align: center; padding: 50px;'>
                <h1>🔐 PROJETO LUCIANO</h1>
                <h3>Sistema de ML para Cancelamento de IE</h3>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            senha_input = st.text_input("Digite a senha:", type="password", key="pwd_input")
            if st.button("Entrar", use_container_width=True):
                if senha_input == SENHA:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("❌ Senha incorreta")
        st.stop()

check_password()

# =============================================================================
# IMPORTS
# =============================================================================

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import warnings
import ssl

# Sklearn para ML
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve
)

# Configuração SSL
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================

st.set_page_config(
    page_title="LUCIANO - ML para Cancelamento de IE",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# ESTILOS CSS
# =============================================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1565c0;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 1rem;
    }

    div[data-testid="stPlotlyChart"] {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        background-color: #ffffff;
    }
    
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 2px solid #2c3e50;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    div[data-testid="stMetric"] > label {
        font-weight: 600;
        color: #2c3e50;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f77b4;
    }
    
    .alert-critico {
        background-color: #ffebee;
        border-left: 5px solid #c62828;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .alert-alto {
        background-color: #fff3e0;
        border-left: 5px solid #ef6c00;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .alert-medio {
        background-color: #fffde7;
        border-left: 5px solid #fbc02d;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .alert-positivo {
        background-color: #e8f5e9;
        border-left: 5px solid #2e7d32;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #1976d2;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .priority-muito-alto {
        background-color: #b71c1c;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .priority-alto {
        background-color: #e65100;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }

    .info-card {
        background-color: #e3f2fd;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-box {
        background-color: #f5f5f5;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        margin: 5px;
    }
    
    .fiscal-header {
        background: linear-gradient(135deg, #1565c0 0%, #0d47a1 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    
    .priority-medio {
        background-color: #f9a825;
        color: black;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONFIGURAÇÃO DE CONEXÃO
# =============================================================================

IMPALA_HOST = 'bdaworkernode02.sef.sc.gov.br'
IMPALA_PORT = 21050
DATABASE = 'teste'

IMPALA_USER = st.secrets.get("impala_credentials", {}).get("user", "tsevero")
IMPALA_PASSWORD = st.secrets.get("impala_credentials", {}).get("password", "")

# =============================================================================
# FUNÇÕES DE CONEXÃO E CARREGAMENTO
# =============================================================================

@st.cache_resource
def get_impala_engine():
    """Cria engine de conexão Impala."""
    try:
        engine = create_engine(
            f'impala://{IMPALA_HOST}:{IMPALA_PORT}/{DATABASE}',
            connect_args={
                'user': IMPALA_USER,
                'password': IMPALA_PASSWORD,
                'auth_mechanism': 'LDAP',
                'use_ssl': True
            }
        )
        return engine
    except Exception as e:
        st.sidebar.error(f"Erro na conexão: {str(e)[:100]}")
        return None


def formatar_cnpj(cnpj):
    """Formata CNPJ para exibição."""
    if pd.isna(cnpj) or cnpj is None:
        return None
    
    cnpj_str = str(cnpj).zfill(14)
    return f"{cnpj_str[:2]}.{cnpj_str[2:5]}.{cnpj_str[5:8]}/{cnpj_str[8:12]}-{cnpj_str[12:14]}"


def limpar_cnpj(cnpj):
    """Limpa e padroniza CNPJ."""
    if pd.isna(cnpj) or cnpj is None:
        return None
    
    if isinstance(cnpj, (int, float)):
        cnpj_str = format(int(cnpj), 'd')
    else:
        cnpj_str = str(cnpj)
    
    if '.' in cnpj_str:
        cnpj_str = cnpj_str.split('.')[0]
    
    cnpj_limpo = ''.join(filter(str.isdigit, cnpj_str))
    
    if not cnpj_limpo:
        return None
    
    if len(cnpj_limpo) > 14:
        cnpj_limpo = cnpj_limpo[:14]
    
    return cnpj_limpo.zfill(14)


@st.cache_data(ttl=3600)
def carregar_resumo_executivo(_engine):
    """Carrega dados resumidos para o dashboard principal - RÁPIDO."""
    dados = {}
    
    if _engine is None:
        return {}
    
    try:
        with _engine.connect() as conn:
            st.sidebar.success("✅ Conexão Impala OK!")
    except Exception as e:
        st.sidebar.error(f"Falha na conexão: {str(e)[:100]}")
        return {}
    
    # 1. Resumo geral
    try:
        query_resumo = f"SELECT * FROM {DATABASE}.luciano_resumo LIMIT 1"
        dados['resumo'] = pd.read_sql(query_resumo, _engine)
    except Exception as e:
        st.sidebar.warning(f"⚠️ luciano_resumo: {str(e)[:50]}")
        dados['resumo'] = pd.DataFrame()
    
    # 2. Top 100 empresas - APENAS ATIVAS (flag_atualmente_cancelada = 0)
    try:
        query_top100 = f"""
            SELECT * FROM {DATABASE}.luciano_top100
            WHERE flag_atualmente_cancelada = 0
            ORDER BY ranking_fiscalizacao
            LIMIT 100
        """
        dados['top100'] = pd.read_sql(query_top100, _engine)
    except Exception as e:
        st.sidebar.warning(f"⚠️ luciano_top100: {str(e)[:50]}")
        dados['top100'] = pd.DataFrame()
    
    # 3. Análise temporal (agregado)
    try:
        query_temporal = f"SELECT * FROM {DATABASE}.luciano_temporal ORDER BY ano_cancelamento DESC, mes_cancelamento DESC"
        dados['temporal'] = pd.read_sql(query_temporal, _engine)
    except Exception as e:
        st.sidebar.warning(f"⚠️ luciano_temporal: {str(e)[:50]}")
        dados['temporal'] = pd.DataFrame()
    
    # 4. Análise por fiscal
    try:
        query_fiscal = f"SELECT * FROM {DATABASE}.luciano_fiscal ORDER BY qtde_protocolos_fiscal DESC"
        dados['fiscal'] = pd.read_sql(query_fiscal, _engine)
    except Exception as e:
        st.sidebar.warning(f"⚠️ luciano_fiscal: {str(e)[:50]}")
        dados['fiscal'] = pd.DataFrame()
    
    # 5. Scores (apenas agregações) - APENAS EMPRESAS ATIVAS
    try:
        query_scores_agg = f"""
            SELECT
                classificacao_risco_final,
                COUNT(*) as qtde,
                SUM(saldo_credor_atual) as saldo_total,
                AVG(score_total) as score_medio,
                AVG(qtde_indicios) as indicios_medio
            FROM {DATABASE}.luciano_scores
            WHERE flag_atualmente_cancelada = 0
            GROUP BY classificacao_risco_final
        """
        dados['scores_agg'] = pd.read_sql(query_scores_agg, _engine)
    except Exception as e:
        st.sidebar.warning(f"⚠️ luciano_scores_agg: {str(e)[:50]}")
        dados['scores_agg'] = pd.DataFrame()
    
    return dados


@st.cache_data(ttl=3600)
def carregar_metricas_completas(_engine):
    """Carrega métricas completas para análises detalhadas."""
    try:
        query = f"""
            SELECT 
                cnpj, nome_contribuinte, razao_social, cd_cnae, descricao_cnae,
                municipio, uf, gerencia_regional, grupo_economico,
                tipo_contribuinte, regime_apuracao,
                total_protocolos, anos_distintos_cancelamento,
                qtde_usuarios_distintos, qtde_individuais, qtde_massivos,
                qtde_automaticos, qtde_manuais, qtde_ainda_cancelada,
                qtde_reativada, qtde_casos_reativacao,
                media_dias_desde_cancelamento, min_dias_desde_cancelamento,
                max_dias_desde_cancelamento, dias_entre_primeiro_ultimo,
                taxa_reativacao_perc, taxa_permanencia_cancelamento_perc,
                dias_medios_reativacao, classificacao_frequencia,
                classificacao_persistencia, efetividade_cancelamento,
                padrao_cancelamento_predominante, lista_fiscais_envolvidos,
                flag_empresa_reincidente, flag_atualmente_cancelada,
                data_primeiro_cancelamento, data_ultimo_cancelamento,
                media_anos_atividade
            FROM {DATABASE}.luciano_metricas
        """
        df = pd.read_sql(query, _engine)
        
        if 'cnpj' in df.columns:
            df['cnpj'] = df['cnpj'].apply(limpar_cnpj)
            df['cnpj_formatado'] = df['cnpj'].apply(formatar_cnpj)
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar métricas: {str(e)[:100]}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def carregar_scores_completos(_engine):
    """Carrega scores completos para ML e análises."""
    try:
        query = f"SELECT * FROM {DATABASE}.luciano_scores"
        df = pd.read_sql(query, _engine)
        
        if 'cnpj' in df.columns:
            df['cnpj'] = df['cnpj'].apply(limpar_cnpj)
            df['cnpj_formatado'] = df['cnpj'].apply(formatar_cnpj)
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar scores: {str(e)[:100]}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def carregar_indicios(_engine):
    """Carrega dados de indícios."""
    try:
        query = f"SELECT * FROM {DATABASE}.luciano_indicios"
        df = pd.read_sql(query, _engine)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar indícios: {str(e)[:100]}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def carregar_creditos(_engine):
    """Carrega dados de créditos."""
    try:
        query = f"SELECT * FROM {DATABASE}.luciano_credito"
        df = pd.read_sql(query, _engine)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar créditos: {str(e)[:100]}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def carregar_empresas_ativas(_engine):
    """
    Carrega empresas ATIVAS para aplicar o modelo de ML.

    CORREÇÃO: Usa LEFT JOIN para incluir:
    1. Empresas que NUNCA foram analisadas (não estão em luciano_scores)
    2. Empresas que JÁ FORAM analisadas mas estão ATIVAS (flag_atualmente_cancelada = 0)

    Isso corrige o problema de excluir empresas reativadas da análise.
    """
    try:
        query = f"""
            SELECT
                ods.nu_cnpj as cnpj,
                ods.nu_ie as ie,
                ods.nm_razao_social as razao_social,
                ods.cd_sit_cadastral as cod_situacao,
                ods.nm_sit_cadastral as situacao_cadastral,
                LPAD(CAST(ods.cd_cnae AS STRING), 7, '0') as cd_cnae,
                ods.de_cnae as descricao_cnae,
                ods.cd_munic as cod_municipio,
                ods.nm_munic as municipio,
                ods.nm_gerfe as gerencia_regional,
                ods.nm_tipo_contribuinte as tipo_contribuinte,
                ods.nm_reg_apuracao as regime_apuracao,
                ods.nu_cnpj_grupo as grupo_economico,
                CASE WHEN sc.cnpj IS NULL THEN 0 ELSE 1 END as ja_analisada,
                sc.score_total as score_existente
            FROM usr_sat_ods.vw_ods_contrib ods
            LEFT JOIN {DATABASE}.luciano_scores sc
                ON ods.nu_cnpj = sc.cnpj
            WHERE ods.cd_sit_cadastral = 1
            AND (
                sc.cnpj IS NULL
                OR sc.flag_atualmente_cancelada = 0
            )
            LIMIT 100000
        """
        df = pd.read_sql(query, _engine)

        if 'cnpj' in df.columns:
            df['cnpj'] = df['cnpj'].apply(limpar_cnpj)

        return df
    except Exception as e:
        st.error(f"Erro ao carregar empresas ativas: {str(e)[:100]}")
        return pd.DataFrame()


def carregar_detalhes_empresa(_engine, cnpj):
    """Carrega detalhes completos de uma empresa específica."""
    cnpj_limpo = limpar_cnpj(cnpj)
    
    detalhes = {}
    
    # Métricas
    try:
        query = f"SELECT * FROM {DATABASE}.luciano_metricas WHERE cnpj = '{cnpj_limpo}'"
        detalhes['metricas'] = pd.read_sql(query, _engine)
    except:
        detalhes['metricas'] = pd.DataFrame()
    
    # Créditos
    try:
        query = f"SELECT * FROM {DATABASE}.luciano_credito WHERE cnpj = '{cnpj_limpo}'"
        detalhes['creditos'] = pd.read_sql(query, _engine)
    except:
        detalhes['creditos'] = pd.DataFrame()
    
    # Indícios
    try:
        query = f"SELECT * FROM {DATABASE}.luciano_indicios WHERE cnpj = '{cnpj_limpo}'"
        detalhes['indicios'] = pd.read_sql(query, _engine)
    except:
        detalhes['indicios'] = pd.DataFrame()
    
    # Scores
    try:
        query = f"SELECT * FROM {DATABASE}.luciano_scores WHERE cnpj = '{cnpj_limpo}'"
        detalhes['scores'] = pd.read_sql(query, _engine)
    except:
        detalhes['scores'] = pd.DataFrame()
    
    # Base (histórico de protocolos)
    try:
        query = f"SELECT * FROM {DATABASE}.luciano_base WHERE cnpj = '{cnpj_limpo}' ORDER BY data_inicio_protocolo DESC"
        detalhes['base'] = pd.read_sql(query, _engine)
    except:
        detalhes['base'] = pd.DataFrame()
    
    return detalhes

def carregar_detalhes_fiscal(_engine, matricula_fiscal):
    """Carrega todos os detalhes de cancelamentos de um fiscal específico."""
    detalhes = {}
    
    # 1. Dados base dos cancelamentos do fiscal
    try:
        query_base = f"""
            SELECT 
                b.*,
                s.score_total,
                s.score_comportamento,
                s.score_credito,
                s.score_indicios,
                s.classificacao_risco_final,
                s.nivel_alerta,
                s.saldo_credor_atual,
                i.qtde_indicios,
                i.qtde_indicios_graves,
                i.soma_scores_indicios,
                i.classificacao_risco_indicios
            FROM {DATABASE}.luciano_base b
            LEFT JOIN {DATABASE}.luciano_scores s ON b.cnpj = s.cnpj
            LEFT JOIN {DATABASE}.luciano_indicios i ON b.cnpj = i.cnpj
            WHERE b.cod_usuario_inicio = '{matricula_fiscal}'
            AND b.flag_cancelamento_automatico = 0
        """
        detalhes['base'] = pd.read_sql(query_base, _engine)
    except Exception as e:
        detalhes['base'] = pd.DataFrame()
        print(f"Erro base: {e}")
    
    # 2. Resumo do fiscal
    try:
        query_fiscal = f"""
            SELECT * FROM {DATABASE}.luciano_fiscal 
            WHERE matricula_fiscal = '{matricula_fiscal}'
        """
        detalhes['resumo'] = pd.read_sql(query_fiscal, _engine)
    except:
        detalhes['resumo'] = pd.DataFrame()
    
    return detalhes

@st.cache_data(ttl=3600)
def carregar_contadores_resumo(_engine):
    """Carrega dados resumidos de contadores."""
    try:
        query = f"SELECT * FROM {DATABASE}.luciano_contabilista_top50 ORDER BY ranking_contador"
        return pd.read_sql(query, _engine)
    except Exception as e:
        st.sidebar.warning(f"⚠️ luciano_contabilista_top50: {str(e)[:50]}")
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def carregar_contadores_completo(_engine):
    """Carrega dados completos dos contadores com scores e taxas"""
    query = """
    SELECT 
        cpf_cnpj_contador,
        nome_contador,
        crc_contador,
        municipio_contador,
        uf_contador,
        -- Métricas de volume
        total_empresas_carteira,
        qtde_empresas_com_cancelamento,
        qtde_empresas_ainda_canceladas,
        qtde_empresas_reativadas,
        -- Taxas (NOVAS)
        taxa_cancelamento_carteira_perc,
        taxa_efetividade_perc,
        -- Risco das empresas
        qtde_empresas_risco_critico,
        qtde_empresas_risco_alto,
        qtde_empresas_risco_medio,
        -- Financeiro
        saldo_credor_total,
        -- Scores
        score_volume,
        score_concentracao,
        score_risco_empresas,
        score_financeiro,
        score_total_contador,
        -- Classificação
        ranking_contador,
        classificacao_risco_contador,
        nivel_alerta_contador
    FROM teste.luciano_contabilista_scores
    ORDER BY ranking_contador
    """
    try:
        df = pd.read_sql(query, _engine)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar contadores: {e}")
        return pd.DataFrame()


def carregar_detalhes_contador(_engine, cpf_cnpj_contador):
    """Carrega detalhes de um contador específico."""
    detalhes = {}
    
    # Limpar CPF/CNPJ
    cpf_cnpj_limpo = ''.join(filter(str.isdigit, str(cpf_cnpj_contador)))
    
    # 1. Resumo do contador
    try:
        query_resumo = f"""
            SELECT * FROM {DATABASE}.luciano_contabilista_scores 
            WHERE REGEXP_REPLACE(cpf_cnpj_contador, '[^0-9]', '') = '{cpf_cnpj_limpo}'
        """
        detalhes['resumo'] = pd.read_sql(query_resumo, _engine)
    except:
        detalhes['resumo'] = pd.DataFrame()
    
    # 2. Empresas do contador
    try:
        query_empresas = f"""
            SELECT * FROM {DATABASE}.luciano_contabilista_base 
            WHERE REGEXP_REPLACE(cpf_cnpj_contador, '[^0-9]', '') = '{cpf_cnpj_limpo}'
            ORDER BY score_total DESC NULLS LAST
        """
        detalhes['empresas'] = pd.read_sql(query_empresas, _engine)
    except:
        detalhes['empresas'] = pd.DataFrame()
    
    return detalhes
    
# =============================================================================
# FUNÇÕES DE ANÁLISE E ML
# =============================================================================

def preparar_features_ml(df_scores):
    """Prepara features para o modelo de ML."""
    
    features_numericas = [
        'score_comportamento', 'score_credito', 'score_indicios', 'score_total',
        'total_protocolos', 'taxa_permanencia_cancelamento_perc', 'taxa_reativacao_perc',
        'saldo_credor_atual', 'vl_credito_60m', 'vl_credito_presumido_60m',
        'qtde_indicios', 'soma_scores_indicios', 'qtde_indicios_graves',
        'perc_valores_iguais_12m', 'variacao_saldo_perc_60m'
    ]
    
    # Filtrar apenas colunas existentes
    features_existentes = [f for f in features_numericas if f in df_scores.columns]
    
    df_ml = df_scores[features_existentes].copy()
    
    # Preencher NaN com 0
    df_ml = df_ml.fillna(0)
    
    return df_ml, features_existentes


def treinar_modelo_cancelamento(df_scores):
    """
    Treina modelo de ML para prever empresas candidatas ao cancelamento.
    Usa empresas já canceladas como target positivo.
    """
    
    if df_scores.empty:
        return None, None, None, None
    
    # Preparar features
    df_ml, features = preparar_features_ml(df_scores)
    
    if len(features) == 0:
        return None, None, None, None
    
    # Target: flag_atualmente_cancelada
    if 'flag_atualmente_cancelada' not in df_scores.columns:
        return None, None, None, None
    
    y = df_scores['flag_atualmente_cancelada'].fillna(0).astype(int)
    X = df_ml
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Normalizar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Treinar Random Forest
    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    modelo.fit(X_train_scaled, y_train)
    
    # Avaliação
    y_pred = modelo.predict(X_test_scaled)
    y_proba = modelo.predict_proba(X_test_scaled)[:, 1]
    
    metricas = {
        'accuracy': (y_pred == y_test).mean(),
        'roc_auc': roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'feature_importance': dict(zip(features, modelo.feature_importances_))
    }
    
    return modelo, scaler, features, metricas


def aplicar_modelo_empresas_ativas(modelo, scaler, features, df_ativas, df_indicios=None):
    """
    Aplica o modelo treinado em empresas ativas para identificar candidatas.
    """
    
    if modelo is None or df_ativas.empty:
        return pd.DataFrame()
    
    # Criar features para empresas ativas
    df_pred = df_ativas.copy()
    
    # Inicializar features com valores padrão
    for f in features:
        if f not in df_pred.columns:
            df_pred[f] = 0
    
    # Enriquecer com indícios se disponível
    if df_indicios is not None and not df_indicios.empty:
        # Merge por CNPJ
        df_pred = df_pred.merge(
            df_indicios[['cnpj', 'qtde_indicios', 'soma_scores_indicios', 'qtde_indicios_graves']],
            on='cnpj',
            how='left',
            suffixes=('', '_ind')
        )
        for col in ['qtde_indicios', 'soma_scores_indicios', 'qtde_indicios_graves']:
            if col + '_ind' in df_pred.columns:
                df_pred[col] = df_pred[col + '_ind'].fillna(0)
    
    # Preparar X
    X = df_pred[features].fillna(0)
    
    # Normalizar
    X_scaled = scaler.transform(X)
    
    # Predizer
    df_pred['prob_cancelamento'] = modelo.predict_proba(X_scaled)[:, 1]
    df_pred['risco_cancelamento'] = modelo.predict(X_scaled)
    
    # Classificar por probabilidade
    df_pred['classificacao_ml'] = pd.cut(
        df_pred['prob_cancelamento'],
        bins=[0, 0.3, 0.5, 0.7, 0.85, 1.0],
        labels=['BAIXO', 'MÉDIO', 'ALTO', 'MUITO ALTO', 'CRÍTICO']
    )
    
    # Ordenar por probabilidade
    df_pred = df_pred.sort_values('prob_cancelamento', ascending=False)
    
    return df_pred


# =============================================================================
# FUNÇÕES DE VISUALIZAÇÃO
# =============================================================================

def criar_gauge_score(valor, titulo, max_valor=100):
    """Cria um gráfico de gauge para scores."""
    
    if valor <= 30:
        cor = '#2e7d32'  # Verde
    elif valor <= 50:
        cor = '#fbc02d'  # Amarelo
    elif valor <= 70:
        cor = '#ef6c00'  # Laranja
    else:
        cor = '#c62828'  # Vermelho
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=valor,
        title={'text': titulo, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, max_valor], 'tickwidth': 1},
            'bar': {'color': cor},
            'steps': [
                {'range': [0, 30], 'color': '#e8f5e9'},
                {'range': [30, 50], 'color': '#fffde7'},
                {'range': [50, 70], 'color': '#fff3e0'},
                {'range': [70, 100], 'color': '#ffebee'}
            ],
            'threshold': {
                'line': {'color': 'red', 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(t=50, b=0, l=30, r=30))
    
    return fig


def criar_grafico_importancia_features(feature_importance, top_n=15):
    """Cria gráfico de importância das features."""
    
    df_imp = pd.DataFrame([
        {'Feature': k, 'Importância': v}
        for k, v in feature_importance.items()
    ]).nlargest(top_n, 'Importância')
    
    fig = px.bar(
        df_imp,
        x='Importância',
        y='Feature',
        orientation='h',
        title=f'Top {top_n} Features Mais Importantes',
        color='Importância',
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
    
    return fig


# =============================================================================
# FILTROS
# =============================================================================

def criar_filtros_sidebar(dados):
    """Cria filtros na sidebar."""
    filtros = {}
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Filtros")
    
    with st.sidebar.expander("Filtros de Risco", expanded=True):
        filtros['classificacoes'] = st.multiselect(
            "Classificação de Risco",
            ['CRÍTICO', 'ALTO', 'MÉDIO', 'BAIXO'],
            default=['CRÍTICO', 'ALTO', 'MÉDIO', 'BAIXO']
        )
        
        filtros['score_minimo'] = st.slider(
            "Score Mínimo",
            min_value=0,
            max_value=100,
            value=0
        )
    
    with st.sidebar.expander("Filtros Geográficos", expanded=False):
        # GERFE
        df_metricas = dados.get('metricas', pd.DataFrame())
        if not df_metricas.empty and 'gerencia_regional' in df_metricas.columns:
            gerfes = ['TODAS'] + sorted(df_metricas['gerencia_regional'].dropna().unique().tolist())
            filtros['gerfe'] = st.selectbox("GERFE", gerfes)
        else:
            filtros['gerfe'] = 'TODAS'
        
        # Município
        if not df_metricas.empty and 'municipio' in df_metricas.columns:
            municipios = ['TODOS'] + sorted(df_metricas['municipio'].dropna().unique().tolist())
            filtros['municipio'] = st.selectbox("Município", municipios)
        else:
            filtros['municipio'] = 'TODOS'
    
    with st.sidebar.expander("Filtros de Comportamento", expanded=False):
        filtros['apenas_reincidentes'] = st.checkbox("Apenas reincidentes", value=False)
        filtros['apenas_canceladas'] = st.checkbox("Apenas canceladas", value=False)
        filtros['min_protocolos'] = st.slider("Mín. protocolos", 1, 20, 1)
    
    with st.sidebar.expander("Visualização", expanded=False):
        filtros['tema'] = st.selectbox(
            "Tema dos Gráficos",
            ["plotly", "plotly_white", "plotly_dark"],
            index=1
        )
        filtros['mostrar_valores'] = st.checkbox("Mostrar valores", value=True)
    
    return filtros


def aplicar_filtros(df, filtros):
    """Aplica filtros no DataFrame."""
    if df.empty:
        return df
    
    df_filtrado = df.copy()
    
    # Classificação de risco
    if 'classificacao_risco_final' in df_filtrado.columns:
        if filtros.get('classificacoes') and len(filtros['classificacoes']) < 4:
            df_filtrado = df_filtrado[df_filtrado['classificacao_risco_final'].isin(filtros['classificacoes'])]
    
    # Score mínimo
    if 'score_total' in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado['score_total'] >= filtros.get('score_minimo', 0)]
    
    # GERFE
    if filtros.get('gerfe') and filtros['gerfe'] != 'TODAS':
        col_gerfe = 'gerencia_regional' if 'gerencia_regional' in df_filtrado.columns else 'nm_gerfe'
        if col_gerfe in df_filtrado.columns:
            df_filtrado = df_filtrado[df_filtrado[col_gerfe] == filtros['gerfe']]
    
    # Município
    if filtros.get('municipio') and filtros['municipio'] != 'TODOS':
        if 'municipio' in df_filtrado.columns:
            df_filtrado = df_filtrado[df_filtrado['municipio'] == filtros['municipio']]
    
    # Reincidentes
    if filtros.get('apenas_reincidentes') and 'flag_empresa_reincidente' in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado['flag_empresa_reincidente'] == 1]
    
    # Canceladas
    if filtros.get('apenas_canceladas') and 'flag_atualmente_cancelada' in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado['flag_atualmente_cancelada'] == 1]
    
    # Mín protocolos
    if 'total_protocolos' in df_filtrado.columns:
        df_filtrado = df_filtrado[df_filtrado['total_protocolos'] >= filtros.get('min_protocolos', 1)]
    
    return df_filtrado


# =============================================================================
# PÁGINAS DO DASHBOARD
# =============================================================================

def pagina_dashboard_executivo(dados, filtros):
    """Dashboard executivo principal."""
    st.markdown("<h1 class='main-header'>🎯 PROJETO LUCIANO - Dashboard Executivo</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <b>Objetivo:</b> Sistema de Machine Learning para identificar empresas ativas com perfil 
    similar às que tiveram IE cancelada, priorizando ações de fiscalização proativa.
    </div>
    """, unsafe_allow_html=True)
    
    # Dados do resumo
    df_resumo = dados.get('resumo', pd.DataFrame())
    
    if df_resumo.empty:
        st.warning("⚠️ Dados de resumo não disponíveis. Execute as queries SQL primeiro.")
        return
    
    resumo = df_resumo.iloc[0] if not df_resumo.empty else {}
    
    # KPIs Principais
    st.subheader("📊 Indicadores do Modelo")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total = resumo.get('total_empresas', 0)
        st.metric("Empresas Analisadas", f"{int(total):,}")
    
    with col2:
        protocolos = resumo.get('total_protocolos', 0)
        st.metric("Total de Protocolos", f"{int(protocolos):,}")
    
    with col3:
        criticos = resumo.get('empresas_risco_critico', 0)
        st.metric("🔴 Risco CRÍTICO", f"{int(criticos):,}")
    
    with col4:
        altos = resumo.get('empresas_risco_alto', 0)
        st.metric("🟠 Risco ALTO", f"{int(altos):,}")
    
    with col5:
        score_medio = resumo.get('score_medio_total', 0)
        st.metric("Score Médio", f"{float(score_medio):.1f}")
    
    st.divider()
    
    # Segunda linha de KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        canceladas = resumo.get('empresas_ainda_canceladas', 0)
        perc_canc = resumo.get('perc_ainda_canceladas', 0)
        st.metric("Ainda Canceladas", f"{int(canceladas):,}", delta=f"{float(perc_canc):.1f}%")
    
    with col2:
        reincidentes = resumo.get('empresas_reincidentes', 0)
        st.metric("Reincidentes", f"{int(reincidentes):,}")
    
    with col3:
        saldo_total = resumo.get('saldo_credor_total', 0)
        st.metric("Saldo Credor Total", f"R$ {float(saldo_total)/1e6:.1f}M")
    
    with col4:
        com_indicios = resumo.get('empresas_com_indicios', 0)
        st.metric("Com Indícios NEAF", f"{int(com_indicios):,}")
    
    with col5:
        indicios_graves = resumo.get('empresas_indicios_graves', 0)
        st.metric("Indícios Graves", f"{int(indicios_graves):,}")
    
    st.divider()
    
    # Gráficos
    col1, col2 = st.columns(2)

    # Terceira linha - KPIs de ALERTAS (NOVO)
    st.divider()
    st.subheader("🚨 Distribuição de Alertas")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        acao_imediata = resumo.get('alertas_acao_imediata', 0)
        st.metric("🔴 AÇÃO IMEDIATA", f"{int(acao_imediata):,}")
    
    with col2:
        muito_urgente = resumo.get('alertas_muito_urgente', 0)
        st.metric("🟠 MUITO URGENTE", f"{int(muito_urgente):,}")
    
    with col3:
        urgente = resumo.get('alertas_urgente', 0)
        st.metric("🟡 URGENTE", f"{int(urgente):,}")
    
    with col4:
        prioridade_alta = resumo.get('alertas_prioridade_alta', 0)
        st.metric("⚪ PRIORIDADE ALTA", f"{int(prioridade_alta):,}")
    
    with col5:
        monitorar = resumo.get('alertas_monitorar', 0)
        st.metric("🟢 MONITORAR", f"{int(monitorar):,}")
    
    # Saldos por alerta
    col1, col2 = st.columns(2)
    with col1:
        saldo_imediata = resumo.get('saldo_acao_imediata', 0)
        st.metric("💰 Saldo Ação Imediata", f"R$ {float(saldo_imediata)/1e6:.2f}M")
    with col2:
        saldo_urgentes = resumo.get('saldo_alertas_urgentes', 0)
        st.metric("💰 Saldo Alertas Urgentes", f"R$ {float(saldo_urgentes)/1e6:.2f}M")
    
    # Distribuição por Risco
    df_scores_agg = dados.get('scores_agg', pd.DataFrame())
    
    with col1:
        if not df_scores_agg.empty:
            fig = px.pie(
                df_scores_agg,
                values='qtde',
                names='classificacao_risco_final',
                title='Distribuição por Nível de Risco',
                color='classificacao_risco_final',
                color_discrete_map={
                    'CRÍTICO': '#c62828',
                    'ALTO': '#ef6c00',
                    'MÉDIO': '#fbc02d',
                    'BAIXO': '#388e3c'
                },
                hole=0.4,
                template=filtros['tema']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if not df_scores_agg.empty:
            fig = px.bar(
                df_scores_agg,
                x='classificacao_risco_final',
                y='saldo_total',
                title='Saldo Credor por Nível de Risco',
                color='classificacao_risco_final',
                color_discrete_map={
                    'CRÍTICO': '#c62828',
                    'ALTO': '#ef6c00',
                    'MÉDIO': '#fbc02d',
                    'BAIXO': '#388e3c'
                },
                template=filtros['tema']
            )
            fig.update_yaxes(title_text="Saldo (R$)")
            st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Top 10 Empresas
    st.subheader("🎯 Top 10 Empresas Prioritárias")
    
    df_top100 = dados.get('top100', pd.DataFrame())
    
    if not df_top100.empty:
        df_top10 = df_top100.head(10)
        
        colunas_display = [
            'ranking_fiscalizacao', 'cnpj', 'nome_contribuinte',
            'classificacao_risco_final', 'score_total',
            'total_protocolos', 'saldo_credor_atual', 'qtde_indicios'
        ]
        
        colunas_existentes = [c for c in colunas_display if c in df_top10.columns]
        
        st.dataframe(
            df_top10[colunas_existentes].style.format({
                'score_total': '{:.1f}',
                'saldo_credor_atual': 'R$ {:,.2f}'
            }),
            use_container_width=True,
            height=400
        )
    else:
        st.info("Dados de ranking não disponíveis.")


def pagina_analise_temporal(dados, filtros):
    """Análise temporal dos cancelamentos."""
    st.markdown("<h1 class='main-header'>📈 Análise Temporal de Cancelamentos</h1>", unsafe_allow_html=True)
    
    df_temporal = dados.get('temporal', pd.DataFrame())
    
    if df_temporal.empty:
        st.warning("⚠️ Dados temporais não disponíveis.")
        return
    
    st.markdown("""
    <div class='info-box'>
    <b>Objetivo:</b> Analisar a evolução dos cancelamentos ao longo do tempo,
    identificando padrões sazonais e tendências.
    </div>
    """, unsafe_allow_html=True)
    
    # KPIs temporais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_periodos = len(df_temporal)
        st.metric("Períodos Analisados", f"{total_periodos}")
    
    with col2:
        media_protocolos = df_temporal['qtde_protocolos'].mean()
        st.metric("Média Protocolos/Mês", f"{media_protocolos:.1f}")
    
    with col3:
        media_empresas = df_temporal['qtde_empresas_distintas'].mean()
        st.metric("Média Empresas/Mês", f"{media_empresas:.1f}")
    
    with col4:
        taxa_media = df_temporal['taxa_permanencia_perc'].mean()
        st.metric("Taxa Permanência Média", f"{taxa_media:.1f}%")
    
    st.divider()
    
    # Gráfico de evolução temporal
    if 'periodo_cancelamento' in df_temporal.columns:
        df_temporal_sorted = df_temporal.sort_values('periodo_cancelamento')
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('Protocolos por Período', 'Taxa de Permanência (%)'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Bar(
                x=df_temporal_sorted['periodo_cancelamento'],
                y=df_temporal_sorted['qtde_protocolos'],
                name='Protocolos',
                marker_color='#1976d2'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_temporal_sorted['periodo_cancelamento'],
                y=df_temporal_sorted['taxa_permanencia_perc'],
                name='Taxa Permanência',
                mode='lines+markers',
                line=dict(color='#c62828', width=2)
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            height=600,
            template=filtros['tema'],
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Análise por ano
    st.subheader("📊 Análise por Ano")
    
    if 'ano_cancelamento' in df_temporal.columns:
        df_ano = df_temporal.groupby('ano_cancelamento').agg({
            'qtde_protocolos': 'sum',
            'qtde_empresas_distintas': 'sum',
            'qtde_automaticos': 'sum',
            'qtde_manuais': 'sum',
            'taxa_permanencia_perc': 'mean'
        }).reset_index()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                df_ano,
                x='ano_cancelamento',
                y=['qtde_automaticos', 'qtde_manuais'],
                title='Cancelamentos por Tipo (Automático vs Manual)',
                barmode='stack',
                template=filtros['tema']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.line(
                df_ano,
                x='ano_cancelamento',
                y='taxa_permanencia_perc',
                title='Evolução da Taxa de Permanência',
                markers=True,
                template=filtros['tema']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Tabela de dados
    st.subheader("📋 Dados por Período")
    
    st.dataframe(
        df_temporal.style.format({
            'taxa_permanencia_perc': '{:.1f}%',
            'taxa_reativacao_perc': '{:.1f}%'
        }),
        use_container_width=True,
        height=400
    )


def pagina_analise_fiscal(dados, filtros, engine):
    """Análise por fiscal com drill-down individual."""
    st.markdown("<h1 class='main-header'>👤 Análise por Fiscal</h1>", unsafe_allow_html=True)
    
    df_fiscal = dados.get('fiscal', pd.DataFrame())
    
    if df_fiscal.empty:
        st.warning("⚠️ Dados de fiscais não disponíveis.")
        return
    
    # Tabs: Visão Geral vs Drill-Down Individual
    tab1, tab2 = st.tabs(["📊 Visão Geral", "🔍 Análise Individual de Fiscal"])
    
    # =========================================================================
    # TAB 1: VISÃO GERAL (código original)
    # =========================================================================
    with tab1:
        st.markdown("""
        <div class='info-box'>
        <b>Objetivo:</b> Analisar a performance dos fiscais nos processos de cancelamento,
        identificando padrões de efetividade.
        </div>
        """, unsafe_allow_html=True)
        
        # KPIs
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_fiscais = len(df_fiscal)
            st.metric("Total de Fiscais", f"{total_fiscais}")
        
        with col2:
            total_protocolos = df_fiscal['qtde_protocolos_fiscal'].sum()
            st.metric("Total Protocolos", f"{int(total_protocolos):,}")
        
        with col3:
            media_efetividade = df_fiscal['taxa_efetividade_perc'].mean()
            st.metric("Efetividade Média", f"{media_efetividade:.1f}%")
        
        with col4:
            media_dias = df_fiscal['media_dias_processamento'].mean()
            st.metric("Tempo Médio (dias)", f"{media_dias:.1f}")
        
        st.divider()
        
        # Top 10 Fiscais
        st.subheader("🏆 Top 10 Fiscais por Volume")
        
        df_top_fiscal = df_fiscal.nlargest(10, 'qtde_protocolos_fiscal')
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                df_top_fiscal,
                x='qtde_protocolos_fiscal',
                y='nome_fiscal',
                orientation='h',
                title='Volume de Protocolos',
                template=filtros['tema'],
                color='taxa_efetividade_perc',
                color_continuous_scale='RdYlGn'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                df_fiscal,
                x='qtde_protocolos_fiscal',
                y='taxa_efetividade_perc',
                size='qtde_empresas_distintas',
                color='media_dias_processamento',
                hover_data=['nome_fiscal'],
                title='Volume vs Efetividade',
                template=filtros['tema'],
                color_continuous_scale='RdYlBu_r'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Tabela clicável
        st.subheader("📋 Lista de Fiscais")
        st.info("💡 Para análise detalhada, vá para a aba 'Análise Individual de Fiscal'")
        
        colunas_display = [
            'matricula_fiscal', 'nome_fiscal', 'qtde_protocolos_fiscal',
            'qtde_empresas_distintas', 'qtde_cancelamentos_efetivos',
            'taxa_efetividade_perc', 'media_dias_processamento'
        ]
        
        colunas_existentes = [c for c in colunas_display if c in df_fiscal.columns]
        
        st.dataframe(
            df_fiscal[colunas_existentes].style.format({
                'taxa_efetividade_perc': '{:.1f}%',
                'media_dias_processamento': '{:.1f}'
            }),
            use_container_width=True,
            height=500
        )
    
    # =========================================================================
    # TAB 2: DRILL-DOWN INDIVIDUAL
    # =========================================================================
    with tab2:
        st.markdown("""
        <div class='info-box'>
        <b>Objetivo:</b> Análise detalhada dos cancelamentos realizados por um fiscal específico,
        incluindo distribuição de risco, motivos, empresas prioritárias e indicadores.
        </div>
        """, unsafe_allow_html=True)
        
        # Seleção do fiscal
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if 'nome_fiscal' in df_fiscal.columns and 'matricula_fiscal' in df_fiscal.columns:
                opcoes_fiscal = df_fiscal[['matricula_fiscal', 'nome_fiscal', 'qtde_protocolos_fiscal']].drop_duplicates()
                opcoes_fiscal = opcoes_fiscal.sort_values('qtde_protocolos_fiscal', ascending=False)
                
                fiscal_selecionado = st.selectbox(
                    "Selecione o Fiscal:",
                    opcoes_fiscal['matricula_fiscal'].tolist(),
                    format_func=lambda x: f"{opcoes_fiscal[opcoes_fiscal['matricula_fiscal']==x]['nome_fiscal'].iloc[0]} ({int(opcoes_fiscal[opcoes_fiscal['matricula_fiscal']==x]['qtde_protocolos_fiscal'].iloc[0])} protocolos)"
                )
            else:
                st.error("Dados de fiscais incompletos.")
                return
        
        if st.button("🔍 Carregar Análise do Fiscal", type="primary"):
            with st.spinner(f"Carregando dados do fiscal {fiscal_selecionado}..."):
                detalhes_fiscal = carregar_detalhes_fiscal(engine, fiscal_selecionado)
                st.session_state['detalhes_fiscal'] = detalhes_fiscal
                st.session_state['fiscal_selecionado'] = fiscal_selecionado
        
        # Exibir análise se dados carregados
        if 'detalhes_fiscal' in st.session_state:
            exibir_drill_down_fiscal(
                st.session_state['detalhes_fiscal'],
                st.session_state['fiscal_selecionado'],
                df_fiscal,
                filtros
            )


def exibir_drill_down_fiscal(detalhes, matricula_fiscal, df_fiscal, filtros):
    """Exibe a análise detalhada de um fiscal específico."""
    
    df_base = detalhes.get('base', pd.DataFrame())
    df_resumo = detalhes.get('resumo', pd.DataFrame())
    
    if df_base.empty:
        st.warning("⚠️ Nenhum dado encontrado para este fiscal.")
        return
    
    # Info do fiscal
    info_fiscal = df_fiscal[df_fiscal['matricula_fiscal'] == matricula_fiscal]
    nome_fiscal = info_fiscal['nome_fiscal'].iloc[0] if not info_fiscal.empty else matricula_fiscal
    
    st.markdown(f"## 📊 Relatório: {nome_fiscal}")
    st.caption(f"Matrícula: {matricula_fiscal}")
    
    st.divider()
    
    # =========================================================================
    # 1. RESUMO EXECUTIVO
    # =========================================================================
    st.subheader("1️⃣ Resumo Executivo")
    
    total_protocolos = len(df_base)
    total_empresas = df_base['cnpj'].nunique()
    ainda_canceladas = df_base['flag_ainda_cancelada'].sum() if 'flag_ainda_cancelada' in df_base.columns else 0
    reativadas = df_base['flag_reativada'].sum() if 'flag_reativada' in df_base.columns else 0
    saldo_total = df_base['saldo_credor_atual'].sum() if 'saldo_credor_atual' in df_base.columns else 0
    total_indicios = df_base['qtde_indicios'].sum() if 'qtde_indicios' in df_base.columns else 0
    indicios_graves = df_base['qtde_indicios_graves'].sum() if 'qtde_indicios_graves' in df_base.columns else 0
    score_medio = df_base['score_total'].mean() if 'score_total' in df_base.columns else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Protocolos", f"{total_protocolos:,}")
        st.metric("Empresas Distintas", f"{total_empresas:,}")
    
    with col2:
        st.metric("Ainda Canceladas", f"{int(ainda_canceladas):,}", 
                  delta=f"{100*ainda_canceladas/total_protocolos:.1f}%" if total_protocolos > 0 else "0%")
        st.metric("Reativadas", f"{int(reativadas):,}")
    
    with col3:
        st.metric("Saldo Credor Total", f"R$ {saldo_total/1e6:.2f}M")
        st.metric("Score Médio", f"{score_medio:.1f}")
    
    with col4:
        st.metric("Total Indícios", f"{int(total_indicios):,}")
        st.metric("Indícios Graves", f"{int(indicios_graves):,}")
    
    st.divider()
    
    # =========================================================================
    # 2. DISTRIBUIÇÃO POR CLASSIFICAÇÃO DE RISCO
    # =========================================================================
    st.subheader("2️⃣ Distribuição por Classificação de Risco")
    
    if 'classificacao_risco_final' in df_base.columns:
        df_risco = df_base.groupby('classificacao_risco_final').agg({
            'cnpj': 'count',
            'saldo_credor_atual': 'sum',
            'qtde_indicios_graves': 'sum'
        }).reset_index()
        df_risco.columns = ['Risco', 'Quantidade', 'Saldo Total', 'Indícios Graves']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(
                df_risco,
                values='Quantidade',
                names='Risco',
                title='Empresas por Nível de Risco',
                color='Risco',
                color_discrete_map={
                    'CRÍTICO': '#c62828',
                    'ALTO': '#ef6c00',
                    'MÉDIO': '#fbc02d',
                    'BAIXO': '#388e3c'
                },
                hole=0.4,
                template=filtros['tema']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                df_risco,
                x='Risco',
                y='Saldo Total',
                title='Saldo Credor por Risco',
                color='Risco',
                color_discrete_map={
                    'CRÍTICO': '#c62828',
                    'ALTO': '#ef6c00',
                    'MÉDIO': '#fbc02d',
                    'BAIXO': '#388e3c'
                },
                template=filtros['tema']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Alertas
        criticos = df_risco[df_risco['Risco'] == 'CRÍTICO']
        altos = df_risco[df_risco['Risco'] == 'ALTO']
        
        if not criticos.empty and criticos['Quantidade'].iloc[0] > 0:
            st.markdown(f"""
            <div class='alert-critico'>
            ⚠️ <b>CRÍTICO:</b> {int(criticos['Quantidade'].iloc[0])} empresas com R$ {criticos['Saldo Total'].iloc[0]/1e6:.2f}M em saldo credor
            </div>
            """, unsafe_allow_html=True)
        
        if not altos.empty and altos['Quantidade'].iloc[0] > 0:
            st.markdown(f"""
            <div class='alert-alto'>
            ⚠️ <b>ALTO:</b> {int(altos['Quantidade'].iloc[0])} empresas com R$ {altos['Saldo Total'].iloc[0]/1e6:.2f}M em saldo credor
            </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    
    # =========================================================================
    # 3. DISTRIBUIÇÃO POR MOTIVO DE CANCELAMENTO
    # =========================================================================
    st.subheader("3️⃣ Distribuição por Motivo de Cancelamento")
    
    if 'cod_motivo' in df_base.columns:
        df_motivo = df_base.groupby('cod_motivo').agg({
            'cnpj': 'count',
            'saldo_credor_atual': 'sum',
            'flag_ainda_cancelada': 'sum'
        }).reset_index()
        df_motivo.columns = ['Código Motivo', 'Quantidade', 'Saldo Total', 'Ainda Canceladas']
        df_motivo = df_motivo.sort_values('Quantidade', ascending=False)
        
        fig = px.bar(
            df_motivo.head(10),
            x='Quantidade',
            y='Código Motivo',
            orientation='h',
            title='Top 10 Motivos de Cancelamento',
            color='Saldo Total',
            color_continuous_scale='Reds',
            template=filtros['tema']
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df_motivo, use_container_width=True, height=300)
    
    st.divider()
    
    # =========================================================================
    # 4. DISTRIBUIÇÃO POR MUNICÍPIO
    # =========================================================================
    st.subheader("4️⃣ Top 15 Municípios com Mais Cancelamentos")
    
    if 'municipio' in df_base.columns:
        df_mun = df_base.groupby('municipio').agg({
            'cnpj': 'count',
            'saldo_credor_atual': 'sum',
            'flag_ainda_cancelada': 'sum',
            'qtde_indicios_graves': 'sum'
        }).reset_index()
        df_mun.columns = ['Município', 'Quantidade', 'Saldo Total', 'Ainda Canceladas', 'Indícios Graves']
        df_mun = df_mun.sort_values('Quantidade', ascending=False).head(15)
        
        fig = px.bar(
            df_mun,
            x='Quantidade',
            y='Município',
            orientation='h',
            title='Cancelamentos por Município',
            color='Saldo Total',
            color_continuous_scale='Blues',
            template=filtros['tema']
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # =========================================================================
    # 5. DISTRIBUIÇÃO TEMPORAL
    # =========================================================================
    st.subheader("5️⃣ Distribuição Temporal dos Cancelamentos")
    
    if 'data_inicio_protocolo' in df_base.columns:
        df_base['ano_cancelamento'] = pd.to_datetime(df_base['data_inicio_protocolo']).dt.year
        
        df_temporal = df_base.groupby('ano_cancelamento').agg({
            'cnpj': 'count',
            'saldo_credor_atual': 'sum'
        }).reset_index()
        df_temporal.columns = ['Ano', 'Quantidade', 'Saldo Total']
        
        fig = px.bar(
            df_temporal,
            x='Ano',
            y='Quantidade',
            title='Cancelamentos por Ano',
            color='Saldo Total',
            color_continuous_scale='Oranges',
            template=filtros['tema']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # =========================================================================
    # 6. TOP 20 EMPRESAS PRIORITÁRIAS
    # =========================================================================
    st.subheader("6️⃣ Top 20 Empresas Prioritárias (Maior Score)")
    
    if 'score_total' in df_base.columns:
        df_top20 = df_base.nlargest(20, 'score_total')
        
        colunas_top = [
            'cnpj', 'nome_contribuinte', 'municipio', 
            'classificacao_risco_final', 'score_total',
            'saldo_credor_atual', 'qtde_indicios_graves'
        ]
        colunas_existentes = [c for c in colunas_top if c in df_top20.columns]
        
        # Adicionar ranking
        df_display = df_top20[colunas_existentes].copy()
        df_display.insert(0, '#', range(1, len(df_display) + 1))
        
        st.dataframe(
            df_display.style.format({
                'score_total': '{:.1f}',
                'saldo_credor_atual': 'R$ {:,.2f}'
            }),
            use_container_width=True,
            height=500
        )
        
        # Detalhes das top 5
        with st.expander("📋 Detalhes das Top 5 Empresas", expanded=False):
            for i, (_, row) in enumerate(df_top20.head(5).iterrows(), 1):
                st.markdown(f"""
                ---
                **#{i} - {row.get('nome_contribuinte', 'N/A')}**
                
                - **CNPJ:** {row.get('cnpj', 'N/A')} | **IE:** {row.get('ie', 'N/A')}
                - **Município:** {row.get('municipio', 'N/A')} ({row.get('gerencia_regional', 'N/A')})
                - **CNAE:** {row.get('cd_cnae', 'N/A')}
                - **Data Cancelamento:** {row.get('data_inicio_protocolo', 'N/A')}
                - **Situação:** {'AINDA CANCELADA' if row.get('flag_ainda_cancelada', 0) == 1 else 'REATIVADA'}
                - **Saldo Credor:** R$ {row.get('saldo_credor_atual', 0):,.2f}
                - **Score:** {row.get('score_total', 0):.1f} | **Risco:** {row.get('classificacao_risco_final', 'N/A')}
                - **Indícios:** {int(row.get('qtde_indicios', 0))} total | {int(row.get('qtde_indicios_graves', 0))} graves
                """)
    
    st.divider()
    
    # =========================================================================
    # 7. EMPRESAS COM SALDO ALTO AINDA CANCELADAS
    # =========================================================================
    st.subheader("7️⃣ Empresas com Saldo Alto Ainda Canceladas (> R$ 100.000)")
    
    if 'flag_ainda_cancelada' in df_base.columns and 'saldo_credor_atual' in df_base.columns:
        df_saldo_alto = df_base[
            (df_base['flag_ainda_cancelada'] == 1) & 
            (df_base['saldo_credor_atual'] > 100000)
        ].sort_values('saldo_credor_atual', ascending=False)
        
        if not df_saldo_alto.empty:
            total_saldo_alto = df_saldo_alto['saldo_credor_atual'].sum()
            
            st.markdown(f"""
            <div class='alert-critico'>
            🚨 <b>ALERTA:</b> {len(df_saldo_alto)} empresas canceladas com saldo credor > R$ 100.000!<br>
            Valor total em risco: <b>R$ {total_saldo_alto/1e6:.2f}M</b>
            </div>
            """, unsafe_allow_html=True)
            
            colunas_saldo = ['cnpj', 'nome_contribuinte', 'saldo_credor_atual', 'classificacao_risco_final']
            colunas_existentes = [c for c in colunas_saldo if c in df_saldo_alto.columns]
            
            st.dataframe(
                df_saldo_alto[colunas_existentes].head(15).style.format({
                    'saldo_credor_atual': 'R$ {:,.2f}'
                }),
                use_container_width=True,
                height=400
            )
        else:
            st.success("✅ Nenhuma empresa cancelada com saldo credor > R$ 100.000")
    
    st.divider()
    
    # =========================================================================
    # 8. EXPORTAR DADOS
    # =========================================================================
    st.subheader("8️⃣ Exportar Dados do Fiscal")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Exportar Todos os Cancelamentos (CSV)"):
            csv = df_base.to_csv(index=False, encoding='utf-8-sig', sep=';')
            st.download_button(
                label="Baixar CSV Completo",
                data=csv.encode('utf-8-sig'),
                file_name=f"cancelamentos_fiscal_{matricula_fiscal}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv'
            )
    
    with col2:
        if 'score_total' in df_base.columns:
            df_prioritarias = df_base[df_base['classificacao_risco_final'].isin(['CRÍTICO', 'ALTO'])]
            if st.button("📥 Exportar Prioritárias (CSV)"):
                csv = df_prioritarias.to_csv(index=False, encoding='utf-8-sig', sep=';')
                st.download_button(
                    label="Baixar CSV Prioritárias",
                    data=csv.encode('utf-8-sig'),
                    file_name=f"prioritarias_fiscal_{matricula_fiscal}_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime='text/csv'
                )

def pagina_analise_contador(dados, filtros, engine):
    """Página de análise por contador/contabilista"""
    
    st.title("📋 Análise por Contador/Contabilista")
    
    # Carregar dados
    df_contab = carregar_contadores_completo(engine)
    
    if df_contab.empty:
        st.warning("Não foi possível carregar dados dos contadores.")
        return
    
    # =========================================================================
    # VERIFICAR E CRIAR COLUNAS SE NÃO EXISTIREM (COMPATIBILIDADE)
    # =========================================================================
    colunas_novas = {
        'total_empresas_carteira': 0,
        'taxa_cancelamento_carteira_perc': 0.0,
        'taxa_efetividade_perc': 0.0
    }
    
    colunas_faltando = []
    for col, default in colunas_novas.items():
        if col not in df_contab.columns:
            df_contab[col] = default
            colunas_faltando.append(col)
    
    if colunas_faltando:
        st.warning(f"""
        ⚠️ **Colunas não encontradas na tabela:** {', '.join(colunas_faltando)}
        
        Execute o SQL atualizado para criar as tabelas `luciano_contabilista_carteira` e 
        `luciano_contabilista_scores` com as novas colunas.
        
        Os valores estão sendo exibidos como 0 temporariamente.
        """)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Visão Geral", "🏆 Ranking", "🔍 Análise Individual"])
    
    # =========================================================================
    # TAB 1: VISÃO GERAL
    # =========================================================================
    with tab1:
        st.subheader("Indicadores Gerais")
        
        # KPIs - Linha 1: Volume
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        with col1:
            total_contab = len(df_contab)
            st.metric("Total Contadores", f"{total_contab:,}")
        
        with col2:
            total_carteira = df_contab['total_empresas_carteira'].sum()
            st.metric("📁 Empresas na Carteira", f"{int(total_carteira):,}")
        
        with col3:
            total_canceladas = df_contab['qtde_empresas_ainda_canceladas'].sum()
            st.metric("🔴 Ainda Canceladas", f"{int(total_canceladas):,}")
        
        with col4:
            criticos = len(df_contab[df_contab['classificacao_risco_contador'] == 'CRÍTICO'])
            st.metric("🔴 Contadores Críticos", f"{criticos:,}")
        
        with col5:
            investigar = len(df_contab[df_contab['nivel_alerta_contador'] == 'INVESTIGAR'])
            st.metric("🚨 A Investigar", f"{investigar:,}")
        
        with col6:
            saldo_total = df_contab['saldo_credor_total'].sum()
            st.metric("💰 Saldo Total", f"R$ {saldo_total/1e9:.2f}B")
        
        # KPIs - Linha 2: Taxas
        st.divider()
        st.subheader("📊 Taxas Médias")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            media_taxa_carteira = df_contab['taxa_cancelamento_carteira_perc'].mean()
            st.metric(
                "⚠️ Taxa Média Cancel. Carteira", 
                f"{media_taxa_carteira:.2f}%",
                help="Média do % da carteira que está cancelada. Quanto MAIOR, PIOR."
            )
        
        with col2:
            media_taxa_efet = df_contab['taxa_efetividade_perc'].mean()
            st.metric(
                "📈 Taxa Média Efetividade", 
                f"{media_taxa_efet:.2f}%",
                help="Média do % de cancelamentos que permaneceram. Mede eficácia."
            )
        
        with col3:
            # Contadores com taxa de cancelamento carteira > 20%
            contadores_alto_risco = len(df_contab[df_contab['taxa_cancelamento_carteira_perc'] > 20])
            st.metric(
                "🔴 Contadores >20% Carteira Cancel.", 
                f"{contadores_alto_risco:,}",
                help="Contadores com mais de 20% da carteira cancelada"
            )
        
        with col4:
            # Contadores com taxa de cancelamento carteira > 50%
            contadores_critico = len(df_contab[df_contab['taxa_cancelamento_carteira_perc'] > 50])
            st.metric(
                "🚨 Contadores >50% Carteira Cancel.", 
                f"{contadores_critico:,}",
                help="Contadores com mais de 50% da carteira cancelada - CRÍTICO"
            )
        
        st.divider()
        
        # Gráficos - Linha 1
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribuição por risco
            df_risco = df_contab.groupby('classificacao_risco_contador').size().reset_index(name='qtde')
            
            fig = px.pie(
                df_risco,
                values='qtde',
                names='classificacao_risco_contador',
                title='Distribuição por Nível de Risco',
                color='classificacao_risco_contador',
                color_discrete_map={
                    'CRÍTICO': '#c62828',
                    'ALTO': '#ef6c00',
                    'MÉDIO': '#fbc02d',
                    'BAIXO': '#4caf50',
                    'MUITO BAIXO': '#81c784'
                },
                hole=0.4,
                template=filtros['tema']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Distribuição por alerta
            df_alerta = df_contab.groupby('nivel_alerta_contador').agg({
                'cpf_cnpj_contador': 'count',
                'saldo_credor_total': 'sum'
            }).reset_index()
            df_alerta.columns = ['Nível de Alerta', 'Quantidade', 'Saldo Total']
            
            fig = px.bar(
                df_alerta,
                x='Nível de Alerta',
                y='Quantidade',
                title='Contadores por Nível de Alerta',
                color='Nível de Alerta',
                color_discrete_map={
                    'INVESTIGAR': '#c62828',
                    'ATENÇÃO ESPECIAL': '#ef6c00',
                    'MONITORAR': '#fbc02d',
                    'NORMAL': '#4caf50'
                },
                template=filtros['tema']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Gráficos - Linha 2: Análise das Taxas
        st.subheader("📈 Análise das Taxas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histograma: Taxa de Cancelamento na Carteira
            fig = px.histogram(
                df_contab,
                x='taxa_cancelamento_carteira_perc',
                nbins=20,
                title='Distribuição: Taxa de Cancelamento na Carteira',
                labels={'taxa_cancelamento_carteira_perc': 'Taxa Cancel. Carteira (%)'},
                color_discrete_sequence=['#c62828'],
                template=filtros['tema']
            )
            fig.add_vline(x=20, line_dash="dash", line_color="orange", 
                          annotation_text="Limite 20%")
            fig.add_vline(x=50, line_dash="dash", line_color="red", 
                          annotation_text="Crítico 50%")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Histograma: Taxa de Efetividade
            fig = px.histogram(
                df_contab,
                x='taxa_efetividade_perc',
                nbins=20,
                title='Distribuição: Taxa de Efetividade',
                labels={'taxa_efetividade_perc': 'Taxa Efetividade (%)'},
                color_discrete_sequence=['#1976d2'],
                template=filtros['tema']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Scatter: Taxa Carteira vs Total Empresas
        st.subheader("📈 Análise de Concentração de Risco")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.scatter(
                df_contab,
                x='total_empresas_carteira',
                y='taxa_cancelamento_carteira_perc',
                size='saldo_credor_total',
                color='classificacao_risco_contador',
                hover_data=['nome_contador', 'qtde_empresas_ainda_canceladas'],
                title='Tamanho da Carteira vs Taxa de Cancelamento',
                color_discrete_map={
                    'CRÍTICO': '#c62828',
                    'ALTO': '#ef6c00',
                    'MÉDIO': '#fbc02d',
                    'BAIXO': '#4caf50',
                    'MUITO BAIXO': '#81c784'
                },
                template=filtros['tema']
            )
            fig.update_xaxes(title_text="Total Empresas na Carteira")
            fig.update_yaxes(title_text="Taxa Cancelamento Carteira (%)")
            fig.add_hline(y=20, line_dash="dash", line_color="orange")
            fig.add_hline(y=50, line_dash="dash", line_color="red")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(
                df_contab,
                x='qtde_empresas_ainda_canceladas',
                y='score_total_contador',
                size='saldo_credor_total',
                color='classificacao_risco_contador',
                hover_data=['nome_contador', 'taxa_cancelamento_carteira_perc'],
                title='Empresas Canceladas vs Score de Risco',
                color_discrete_map={
                    'CRÍTICO': '#c62828',
                    'ALTO': '#ef6c00',
                    'MÉDIO': '#fbc02d',
                    'BAIXO': '#4caf50',
                    'MUITO BAIXO': '#81c784'
                },
                template=filtros['tema']
            )
            fig.update_xaxes(title_text="Quantidade de Empresas Canceladas")
            fig.update_yaxes(title_text="Score de Risco do Contador")
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Análise por município do contador
        st.subheader("📍 Distribuição por Município do Contador")
        
        if 'municipio_contador' in df_contab.columns:
            df_mun_cont = df_contab.groupby('municipio_contador').agg({
                'cpf_cnpj_contador': 'count',
                'total_empresas_carteira': 'sum',
                'qtde_empresas_ainda_canceladas': 'sum',
                'saldo_credor_total': 'sum',
                'taxa_cancelamento_carteira_perc': 'mean'
            }).reset_index()
            df_mun_cont.columns = ['Município', 'Contadores', 'Empresas Carteira', 
                                   'Empresas Canceladas', 'Saldo Total', 'Taxa Média Cancel.']
            df_mun_cont = df_mun_cont.sort_values('Contadores', ascending=False).head(15)
            
            fig = px.bar(
                df_mun_cont,
                x='Contadores',
                y='Município',
                orientation='h',
                title='Top 15 Municípios por Quantidade de Contadores',
                color='Taxa Média Cancel.',
                color_continuous_scale='Reds',
                template=filtros['tema']
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Alertas de contadores a investigar
        df_investigar = df_contab[df_contab['nivel_alerta_contador'] == 'INVESTIGAR']
        
        if not df_investigar.empty:
            st.subheader("🚨 Contadores para Investigação")
            
            for _, row in df_investigar.head(5).iterrows():
                st.markdown(f"""
                <div class='alert-critico'>
                <b>{row['nome_contador']}</b> (CRC: {row.get('crc_contador', 'N/A')})<br>
                📁 Carteira: {int(row['total_empresas_carteira'])} empresas | 
                🔴 Canceladas: {int(row['qtde_empresas_ainda_canceladas'])} |
                ⚠️ Taxa Carteira: <b>{row['taxa_cancelamento_carteira_perc']:.1f}%</b> |
                📈 Efetividade: {row['taxa_efetividade_perc']:.1f}%<br>
                💰 Saldo: R$ {row['saldo_credor_total']/1e6:.2f}M |
                🔴 Risco Crítico: {int(row['qtde_empresas_risco_critico'])}<br>
                <small>📍 {row.get('municipio_contador', 'N/A')} - {row.get('uf_contador', 'N/A')}</small>
                </div>
                """, unsafe_allow_html=True)
        
        # Top 10 piores taxas de cancelamento na carteira
        st.divider()
        st.subheader("🚨 Top 10 Contadores com Maior Taxa de Cancelamento na Carteira")
        
        df_piores = df_contab.nlargest(10, 'taxa_cancelamento_carteira_perc')
        
        for i, (_, row) in enumerate(df_piores.iterrows(), 1):
            cor_classe = 'alert-critico' if row['taxa_cancelamento_carteira_perc'] > 50 else 'alert-alto' if row['taxa_cancelamento_carteira_perc'] > 20 else 'alert-medio'
            st.markdown(f"""
            <div class='{cor_classe}'>
            <b>#{i} {row['nome_contador']}</b><br>
            📁 Carteira: {int(row['total_empresas_carteira'])} | 
            🔴 Canceladas: {int(row['qtde_empresas_ainda_canceladas'])} |
            ⚠️ <b>Taxa Carteira: {row['taxa_cancelamento_carteira_perc']:.1f}%</b> |
            📈 Efetividade: {row['taxa_efetividade_perc']:.1f}%
            </div>
            """, unsafe_allow_html=True)
    
    # =========================================================================
    # TAB 2: RANKING
    # =========================================================================
    with tab2:
        st.subheader("🏆 Ranking de Contadores por Risco")
        
        # Filtros
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            filtro_risco = st.multiselect(
                "Filtrar por Risco:",
                ['CRÍTICO', 'ALTO', 'MÉDIO', 'BAIXO', 'MUITO BAIXO'],
                default=['CRÍTICO', 'ALTO', 'MÉDIO']
            )
        
        with col2:
            min_empresas = st.slider("Mín. Empresas Canceladas:", 1, 20, 1)
        
        with col3:
            min_taxa_carteira = st.slider("Mín. Taxa Cancel. Carteira (%):", 0, 50, 0)
        
        with col4:
            ordenar_por = st.selectbox(
                "Ordenar por:",
                [
                    'ranking_contador', 
                    'taxa_cancelamento_carteira_perc',  # NOVO
                    'qtde_empresas_ainda_canceladas', 
                    'taxa_efetividade_perc',            # NOVO
                    'total_empresas_carteira',          # NOVO
                    'saldo_credor_total', 
                    'score_total_contador'
                ],
                format_func=lambda x: {
                    'ranking_contador': 'Ranking Geral',
                    'taxa_cancelamento_carteira_perc': '⚠️ Taxa Cancel. Carteira (PIOR)',
                    'qtde_empresas_ainda_canceladas': 'Empresas Canceladas',
                    'taxa_efetividade_perc': '📈 Taxa Efetividade',
                    'total_empresas_carteira': 'Tamanho da Carteira',
                    'saldo_credor_total': 'Saldo Credor',
                    'score_total_contador': 'Score Total'
                }.get(x, x)
            )
        
        # Aplicar filtros
        df_filtrado = df_contab[
            (df_contab['classificacao_risco_contador'].isin(filtro_risco)) &
            (df_contab['qtde_empresas_ainda_canceladas'] >= min_empresas) &
            (df_contab['taxa_cancelamento_carteira_perc'] >= min_taxa_carteira)
        ]
        
        # Ordenação
        ascending = (ordenar_por == 'ranking_contador')
        df_filtrado = df_filtrado.sort_values(ordenar_por, ascending=ascending)
        
        # KPIs do filtro
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Contadores Filtrados", f"{len(df_filtrado):,}")
        
        with col2:
            total_carteira_filt = df_filtrado['total_empresas_carteira'].sum()
            st.metric("Total Carteira", f"{int(total_carteira_filt):,}")
        
        with col3:
            total_cancel_filt = df_filtrado['qtde_empresas_ainda_canceladas'].sum()
            st.metric("Total Canceladas", f"{int(total_cancel_filt):,}")
        
        with col4:
            media_taxa = df_filtrado['taxa_cancelamento_carteira_perc'].mean()
            st.metric("Média Taxa Carteira", f"{media_taxa:.1f}%")
        
        st.divider()
        
        # Top 20 em gráfico
        df_top20 = df_filtrado.head(20)
        
        if not df_top20.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    df_top20,
                    x='qtde_empresas_ainda_canceladas',
                    y='nome_contador',
                    orientation='h',
                    title='Top 20: Empresas Canceladas',
                    color='classificacao_risco_contador',
                    color_discrete_map={
                        'CRÍTICO': '#c62828',
                        'ALTO': '#ef6c00',
                        'MÉDIO': '#fbc02d',
                        'BAIXO': '#4caf50',
                        'MUITO BAIXO': '#81c784'
                    },
                    template=filtros['tema']
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.bar(
                    df_top20,
                    x='taxa_cancelamento_carteira_perc',
                    y='nome_contador',
                    orientation='h',
                    title='Top 20: Taxa Cancelamento Carteira (%)',
                    color='classificacao_risco_contador',
                    color_discrete_map={
                        'CRÍTICO': '#c62828',
                        'ALTO': '#ef6c00',
                        'MÉDIO': '#fbc02d',
                        'BAIXO': '#4caf50',
                        'MUITO BAIXO': '#81c784'
                    },
                    template=filtros['tema']
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
                fig.add_vline(x=20, line_dash="dash", line_color="orange")
                fig.add_vline(x=50, line_dash="dash", line_color="red")
                st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Tabela completa
        st.subheader("📋 Lista Completa")
        
        colunas_display = [
            'ranking_contador', 
            'nome_contador', 
            'crc_contador',
            'municipio_contador', 
            'uf_contador',
            'total_empresas_carteira',           # NOVO
            'qtde_empresas_ainda_canceladas',
            'taxa_cancelamento_carteira_perc',   # NOVO - quanto maior, PIOR
            'taxa_efetividade_perc',             # NOVO - performance
            'qtde_empresas_risco_critico', 
            'qtde_empresas_risco_alto',
            'saldo_credor_total', 
            'score_total_contador',
            'classificacao_risco_contador', 
            'nivel_alerta_contador'
        ]
        
        colunas_existentes = [c for c in colunas_display if c in df_filtrado.columns]
        
        # Renomear colunas para exibição
        df_display = df_filtrado[colunas_existentes].copy()
        df_display = df_display.rename(columns={
            'ranking_contador': 'Rank',
            'nome_contador': 'Nome',
            'crc_contador': 'CRC',
            'municipio_contador': 'Município',
            'uf_contador': 'UF',
            'total_empresas_carteira': '📁 Carteira',
            'qtde_empresas_ainda_canceladas': '🔴 Canceladas',
            'taxa_cancelamento_carteira_perc': '⚠️ Taxa Carteira %',
            'taxa_efetividade_perc': '📈 Efetividade %',
            'qtde_empresas_risco_critico': 'Crítico',
            'qtde_empresas_risco_alto': 'Alto',
            'saldo_credor_total': '💰 Saldo',
            'score_total_contador': 'Score',
            'classificacao_risco_contador': 'Classif.',
            'nivel_alerta_contador': 'Alerta'
        })
        
        st.dataframe(
            df_display.style.format({
                '⚠️ Taxa Carteira %': '{:.1f}%',
                '📈 Efetividade %': '{:.1f}%',
                '💰 Saldo': 'R$ {:,.2f}',
                'Score': '{:.1f}'
            }).background_gradient(
                subset=['⚠️ Taxa Carteira %'],
                cmap='Reds'
            ),
            use_container_width=True,
            height=600
        )
        
        st.divider()
        
        # Resumo por faixas de taxa de cancelamento
        st.subheader("📊 Resumo por Faixas de Taxa de Cancelamento na Carteira")
        
        df_contab_temp = df_filtrado.copy()
        df_contab_temp['faixa_taxa'] = pd.cut(
            df_contab_temp['taxa_cancelamento_carteira_perc'],
            bins=[-1, 5, 10, 20, 50, 100],
            labels=['0-5%', '5-10%', '10-20%', '20-50%', '>50%']
        )
        
        df_faixas = df_contab_temp.groupby('faixa_taxa').agg({
            'cpf_cnpj_contador': 'count',
            'total_empresas_carteira': 'sum',
            'qtde_empresas_ainda_canceladas': 'sum',
            'saldo_credor_total': 'sum'
        }).reset_index()
        df_faixas.columns = ['Faixa Taxa', 'Contadores', 'Empresas Carteira', 'Empresas Canceladas', 'Saldo Total']
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                df_faixas,
                x='Faixa Taxa',
                y='Contadores',
                title='Contadores por Faixa de Taxa de Cancelamento',
                color='Faixa Taxa',
                color_discrete_map={
                    '0-5%': '#4caf50',
                    '5-10%': '#8bc34a',
                    '10-20%': '#ffc107',
                    '20-50%': '#ff9800',
                    '>50%': '#f44336'
                },
                template=filtros['tema']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                df_faixas,
                x='Faixa Taxa',
                y='Saldo Total',
                title='Saldo Credor por Faixa de Taxa',
                color='Faixa Taxa',
                color_discrete_map={
                    '0-5%': '#4caf50',
                    '5-10%': '#8bc34a',
                    '10-20%': '#ffc107',
                    '20-50%': '#ff9800',
                    '>50%': '#f44336'
                },
                template=filtros['tema']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df_faixas, use_container_width=True)
        
        st.divider()
        
        # Exportar
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📥 Exportar Ranking Completo (CSV)"):
                csv = df_filtrado[colunas_existentes].to_csv(index=False, encoding='utf-8-sig', sep=';')
                st.download_button(
                    label="Baixar CSV",
                    data=csv.encode('utf-8-sig'),
                    file_name=f"ranking_contadores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime='text/csv'
                )
        
        with col2:
            # Exportar apenas os críticos
            df_criticos = df_filtrado[df_filtrado['taxa_cancelamento_carteira_perc'] > 20]
            if st.button(f"📥 Exportar Críticos >20% ({len(df_criticos)} contadores)"):
                csv = df_criticos[colunas_existentes].to_csv(index=False, encoding='utf-8-sig', sep=';')
                st.download_button(
                    label="Baixar CSV Críticos",
                    data=csv.encode('utf-8-sig'),
                    file_name=f"contadores_criticos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime='text/csv'
                )
    
    # =========================================================================
    # TAB 3: ANÁLISE INDIVIDUAL
    # =========================================================================
    with tab3:
        st.subheader("🔍 Análise Individual de Contador")
        
        # Seleção do contador
        colunas_opcoes = ['cpf_cnpj_contador', 'nome_contador', 'qtde_empresas_ainda_canceladas', 'ranking_contador']
        colunas_existentes = [c for c in colunas_opcoes if c in df_contab.columns]
        opcoes_contab = df_contab[colunas_existentes].copy()
        
        # Ordenar por ranking se existir, senão por empresas canceladas
        if 'ranking_contador' in opcoes_contab.columns:
            opcoes_contab = opcoes_contab.sort_values('ranking_contador')
        else:
            opcoes_contab = opcoes_contab.sort_values('qtde_empresas_ainda_canceladas', ascending=False)
        
        def format_contador(x):
            row = opcoes_contab[opcoes_contab['cpf_cnpj_contador'] == x]
            if row.empty:
                return x
            nome = row['nome_contador'].iloc[0]
            qtde = int(row['qtde_empresas_ainda_canceladas'].iloc[0])
            if 'ranking_contador' in row.columns:
                rank = int(row['ranking_contador'].iloc[0])
                return f"#{rank} - {nome} ({qtde} empresas)"
            else:
                return f"{nome} ({qtde} empresas)"
        
        contab_selecionado = st.selectbox(
            "Selecione o Contador:",
            opcoes_contab['cpf_cnpj_contador'].tolist(),
            format_func=format_contador
        )
        
        if st.button("🔍 Carregar Análise do Contador", type="primary"):
            with st.spinner("Carregando dados..."):
                detalhes = carregar_detalhes_contador(engine, contab_selecionado)
                st.session_state['detalhes_contador'] = detalhes
                st.session_state['contador_selecionado'] = contab_selecionado
        
        # Exibir análise
        if 'detalhes_contador' in st.session_state:
            exibir_drill_down_contador(
                st.session_state['detalhes_contador'],
                df_contab,
                filtros
            )


def exibir_drill_down_contador(detalhes, df_contab_geral, filtros):
    """Exibe análise detalhada de um contador."""
    
    df_resumo = detalhes.get('resumo', pd.DataFrame())
    df_empresas = detalhes.get('empresas', pd.DataFrame())
    
    if df_resumo.empty:
        st.warning("⚠️ Dados não encontrados para este contador.")
        return
    
    info = df_resumo.iloc[0]
    
    # Header
    st.markdown(f"## 📋 {info.get('nome_contador', 'N/A')}")
    st.caption(f"CPF/CNPJ: {info.get('cpf_cnpj_contador', 'N/A')} | CRC: {info.get('crc_contador', 'N/A')}")
    st.caption(f"📍 {info.get('municipio_contador', 'N/A')} - {info.get('uf_contador', 'N/A')} | 📧 {info.get('email_contador', 'N/A')} | 📞 {info.get('telefone_contador', 'N/A')}")
    
    # Classificação e alerta
    classif = info.get('classificacao_risco_contador', 'N/A')
    alerta = info.get('nivel_alerta_contador', 'N/A')
    
    col1, col2 = st.columns(2)
    with col1:
        if classif == 'CRÍTICO':
            st.markdown(f"<div class='alert-critico'><b>🔴 Classificação: {classif}</b></div>", unsafe_allow_html=True)
        elif classif == 'ALTO':
            st.markdown(f"<div class='alert-alto'><b>🟠 Classificação: {classif}</b></div>", unsafe_allow_html=True)
        elif classif == 'MÉDIO':
            st.markdown(f"<div class='alert-medio'><b>🟡 Classificação: {classif}</b></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='alert-positivo'><b>🟢 Classificação: {classif}</b></div>", unsafe_allow_html=True)
    
    with col2:
        if alerta == 'INVESTIGAR':
            st.markdown(f"<div class='alert-critico'><b>🚨 Alerta: {alerta}</b></div>", unsafe_allow_html=True)
        elif alerta == 'ATENÇÃO ESPECIAL':
            st.markdown(f"<div class='alert-alto'><b>⚠️ Alerta: {alerta}</b></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='info-box'><b>ℹ️ Alerta: {alerta}</b></div>", unsafe_allow_html=True)
    
    st.divider()
    
    # KPIs - TAXAS SEPARADAS
    st.subheader("📊 Indicadores do Contador")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("📁 Carteira Total", f"{int(info.get('total_empresas_carteira', 0)):,}")
        st.metric("✅ Ativas", f"{int(info.get('empresas_ativas_carteira', 0)):,}")
    
    with col2:
        st.metric("📋 Com Cancelamento", f"{int(info.get('qtde_empresas_com_cancelamento', 0)):,}")
        st.metric("🔴 Ainda Canceladas", f"{int(info.get('qtde_empresas_ainda_canceladas', 0)):,}")
    
    with col3:
        # Taxa de Cancelamento na Carteira (quanto maior, PIOR)
        taxa_carteira = info.get('taxa_cancelamento_carteira_perc', 0)
        delta_color = "inverse"  # vermelho se aumentar
        st.metric(
            "⚠️ Taxa Cancel. Carteira", 
            f"{taxa_carteira:.1f}%",
            help="% da carteira total que está cancelada. Quanto MAIOR, PIOR."
        )
        st.metric("🔄 Reativadas", f"{int(info.get('qtde_empresas_reativadas', 0)):,}")
    
    with col4:
        # Taxa de Efetividade (Performance dos cancelamentos)
        taxa_efet = info.get('taxa_efetividade_perc', 0)
        st.metric(
            "📈 Taxa Efetividade", 
            f"{taxa_efet:.1f}%",
            help="% das empresas canceladas que CONTINUAM canceladas. Mede eficácia."
        )
        st.metric("🟠 Risco Alto", f"{int(info.get('qtde_empresas_risco_alto', 0)):,}")
    
    with col5:
        st.metric("🔴 Risco Crítico", f"{int(info.get('qtde_empresas_risco_critico', 0)):,}")
        st.metric("Saldo Credor", f"R$ {info.get('saldo_credor_total', 0)/1e6:.2f}M")
    
    with col6:
        st.metric("Total Indícios", f"{int(info.get('total_indicios', 0)):,}")
        st.metric("Indícios Graves", f"{int(info.get('total_indicios_graves', 0)):,}")
    
    st.divider()
    
    # Scores do contador
    st.subheader("🎯 Composição do Score de Risco")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fig = criar_gauge_score(float(info.get('score_volume', 0)), "Volume", 30)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = criar_gauge_score(float(info.get('score_concentracao', 0)), "Concentração", 25)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = criar_gauge_score(float(info.get('score_risco_empresas', 0)), "Risco Empresas", 25)
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        fig = criar_gauge_score(float(info.get('score_financeiro', 0)), "Financeiro", 20)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Lista de empresas
    if not df_empresas.empty:
        st.subheader(f"🏢 Empresas do Contador ({len(df_empresas)} registros)")
        
        # Resumo por status
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            canceladas = df_empresas[df_empresas['flag_ainda_cancelada'] == 1]['cnpj'].nunique()
            st.metric("Empresas Canceladas", canceladas)
        
        with col2:
            reativadas = df_empresas[df_empresas['flag_reativada'] == 1]['cnpj'].nunique()
            st.metric("Empresas Reativadas", reativadas)
        
        with col3:
            if 'classificacao_risco_final' in df_empresas.columns:
                criticos = df_empresas[df_empresas['classificacao_risco_final'] == 'CRÍTICO']['cnpj'].nunique()
                st.metric("Risco Crítico", criticos)
        
        with col4:
            municipios = df_empresas['municipio'].nunique()
            st.metric("Municípios", municipios)
        
        # Gráficos lado a lado
        col1, col2 = st.columns(2)
        
        with col1:
            # Por município
            if 'municipio' in df_empresas.columns:
                df_mun = df_empresas.groupby('municipio')['cnpj'].nunique().reset_index()
                df_mun.columns = ['Município', 'Empresas']
                df_mun = df_mun.sort_values('Empresas', ascending=False).head(10)
                
                fig = px.bar(
                    df_mun,
                    x='Empresas',
                    y='Município',
                    orientation='h',
                    title='Top 10 Municípios (Empresas)',
                    template=filtros['tema']
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Por classificação de risco
            if 'classificacao_risco_final' in df_empresas.columns:
                df_risco = df_empresas.groupby('classificacao_risco_final')['cnpj'].nunique().reset_index()
                df_risco.columns = ['Risco', 'Empresas']
                
                fig = px.pie(
                    df_risco,
                    values='Empresas',
                    names='Risco',
                    title='Distribuição por Risco',
                    color='Risco',
                    color_discrete_map={
                        'CRÍTICO': '#c62828',
                        'ALTO': '#ef6c00',
                        'MÉDIO': '#fbc02d',
                        'BAIXO': '#388e3c'
                    },
                    hole=0.4,
                    template=filtros['tema']
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Tabela de empresas
        st.subheader("📋 Lista de Empresas")
        
        colunas_emp = [
            'cnpj', 'nome_contribuinte', 'municipio', 'cd_cnae',
            'flag_ainda_cancelada', 'classificacao_risco_final', 
            'score_total', 'saldo_credor_atual', 'qtde_indicios_graves'
        ]
        
        colunas_existentes = [c for c in colunas_emp if c in df_empresas.columns]
        
        # Remover duplicatas por CNPJ
        df_empresas_unique = df_empresas.drop_duplicates(subset=['cnpj'], keep='first')
        
        st.dataframe(
            df_empresas_unique[colunas_existentes].style.format({
                'score_total': '{:.1f}',
                'saldo_credor_atual': 'R$ {:,.2f}'
            }),
            use_container_width=True,
            height=500
        )
        
        # Exportar
        if st.button("📥 Exportar Empresas do Contador (CSV)"):
            csv = df_empresas_unique.to_csv(index=False, encoding='utf-8-sig', sep=';')
            st.download_button(
                label="Baixar CSV",
                data=csv.encode('utf-8-sig'),
                file_name=f"empresas_contador_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv'
            )
    else:
        st.info("Nenhuma empresa encontrada para este contador.")
        
def pagina_ranking_empresas(dados, filtros):
    """Ranking de empresas prioritárias - APENAS EMPRESAS ATIVAS."""
    st.markdown("<h1 class='main-header'>🏆 Ranking de Empresas Prioritárias</h1>", unsafe_allow_html=True)

    df_top100 = dados.get('top100', pd.DataFrame())

    if df_top100.empty:
        st.warning("⚠️ Dados de ranking não disponíveis.")
        return

    # CORREÇÃO: Filtrar apenas empresas ATIVAS (não canceladas atualmente)
    if 'flag_atualmente_cancelada' in df_top100.columns:
        total_antes = len(df_top100)
        df_top100 = df_top100[df_top100['flag_atualmente_cancelada'] == 0].copy()
        total_depois = len(df_top100)
        if total_antes != total_depois:
            st.info(f"📊 Exibindo {total_depois} empresas ATIVAS (excluídas {total_antes - total_depois} já canceladas)")

    if df_top100.empty:
        st.warning("⚠️ Nenhuma empresa ATIVA encontrada no ranking.")
        return

    st.markdown("""
    <div class='info-box'>
    <b>Objetivo:</b> Ranking das empresas <b>ATIVAS</b> com maior risco baseado em scores compostos
    de comportamento, créditos e indícios NEAF.
    </div>
    """, unsafe_allow_html=True)
    
    # Aplicar filtros
    df_filtrado = aplicar_filtros(df_top100, filtros)
    
    # KPIs do ranking
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Empresas no Ranking", f"{len(df_filtrado)}")
    
    with col2:
        saldo_total = df_filtrado['saldo_credor_atual'].sum() if 'saldo_credor_atual' in df_filtrado.columns else 0
        st.metric("Saldo Total", f"R$ {saldo_total/1e6:.1f}M")
    
    with col3:
        score_medio = df_filtrado['score_total'].mean() if 'score_total' in df_filtrado.columns else 0
        st.metric("Score Médio", f"{score_medio:.1f}")
    
    with col4:
        indicios_total = df_filtrado['qtde_indicios'].sum() if 'qtde_indicios' in df_filtrado.columns else 0
        st.metric("Total Indícios", f"{int(indicios_total):,}")
    
    st.divider()
    
    # Configuração de exibição
    col1, col2 = st.columns([3, 1])
    
    with col1:
        criterio_ordem = st.selectbox(
            "Ordenar por:",
            ['ranking_fiscalizacao', 'score_total', 'saldo_credor_atual', 'qtde_indicios', 'total_protocolos'],
            index=0
        )
    
    with col2:
        top_n = st.slider("Exibir Top N", 10, 100, 50, 10)
    
    # Ordenar
    if criterio_ordem in df_filtrado.columns:
        ascending = criterio_ordem == 'ranking_fiscalizacao'
        df_ordenado = df_filtrado.sort_values(criterio_ordem, ascending=ascending).head(top_n)
    else:
        df_ordenado = df_filtrado.head(top_n)
    
    # Gráfico
    if 'score_total' in df_ordenado.columns:
        fig = px.bar(
            df_ordenado.head(20),
            x='score_total',
            y='nome_contribuinte' if 'nome_contribuinte' in df_ordenado.columns else 'cnpj',
            orientation='h',
            title=f'Top 20 Empresas por Score',
            color='classificacao_risco_final' if 'classificacao_risco_final' in df_ordenado.columns else None,
            color_discrete_map={
                'CRÍTICO': '#c62828',
                'ALTO': '#ef6c00',
                'MÉDIO': '#fbc02d',
                'BAIXO': '#388e3c'
            },
            template=filtros['tema']
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Tabela
    st.subheader("📋 Lista Completa")
    
    colunas_display = [
        'ranking_fiscalizacao', 'cnpj', 'nome_contribuinte', 'municipio',
        'classificacao_risco_final', 'nivel_alerta', 'score_total', 
        'score_comportamento', 'score_credito', 'score_indicios', 
        'total_protocolos', 'saldo_credor_atual', 'qtde_indicios'
    ]
    
    colunas_existentes = [c for c in colunas_display if c in df_ordenado.columns]
    
    st.dataframe(
        df_ordenado[colunas_existentes].style.format({
            'score_total': '{:.1f}',
            'score_comportamento': '{:.0f}',
            'score_credito': '{:.0f}',
            'score_indicios': '{:.0f}',
            'saldo_credor_atual': 'R$ {:,.2f}'
        }),
        use_container_width=True,
        height=600
    )
    
    # Exportar
    if st.button("📥 Exportar Ranking (CSV)"):
        csv = df_ordenado[colunas_existentes].to_csv(index=False, encoding='utf-8-sig', sep=';')
        st.download_button(
            label="Baixar CSV",
            data=csv.encode('utf-8-sig'),
            file_name=f"ranking_luciano_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )


def pagina_analise_setorial(dados, filtros, engine):
    """Análise por setor (CNAE/GERFE)."""
    st.markdown("<h1 class='main-header'>🏭 Análise Setorial</h1>", unsafe_allow_html=True)
    
    # Carregar dados completos se necessário
    df_metricas = dados.get('metricas', pd.DataFrame())
    
    if df_metricas.empty:
        with st.spinner("Carregando métricas completas..."):
            df_metricas = carregar_metricas_completas(engine)
            dados['metricas'] = df_metricas
    
    if df_metricas.empty:
        st.warning("⚠️ Dados de métricas não disponíveis.")
        return
    
    st.markdown("""
    <div class='info-box'>
    <b>Objetivo:</b> Análise segmentada por CNAE e GERFE para identificar 
    setores com maior concentração de riscos.
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs para diferentes análises
    tab1, tab2 = st.tabs(["📊 Por GERFE", "🏭 Por CNAE"])
    
    with tab1:
        st.subheader("Análise por Gerência Regional")
        
        if 'gerencia_regional' in df_metricas.columns:
            df_gerfe = df_metricas.groupby('gerencia_regional').agg({
                'cnpj': 'count',
                'total_protocolos': 'sum',
                'flag_ainda_cancelada' if 'flag_ainda_cancelada' in df_metricas.columns else 'qtde_ainda_cancelada': 'sum',
                'flag_empresa_reincidente': 'sum' if 'flag_empresa_reincidente' in df_metricas.columns else lambda x: 0
            }).reset_index()
            
            df_gerfe.columns = ['GERFE', 'Empresas', 'Protocolos', 'Canceladas', 'Reincidentes']
            df_gerfe = df_gerfe.sort_values('Empresas', ascending=False)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    df_gerfe.head(15),
                    x='Empresas',
                    y='GERFE',
                    orientation='h',
                    title='Empresas por GERFE',
                    template=filtros['tema'],
                    color='Protocolos',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.pie(
                    df_gerfe.head(10),
                    values='Empresas',
                    names='GERFE',
                    title='Distribuição por GERFE (Top 10)',
                    template=filtros['tema']
                )
                st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(df_gerfe, use_container_width=True, height=400)
    
    with tab2:
        st.subheader("Análise por CNAE")
        
        if 'cd_cnae' in df_metricas.columns:
            df_cnae = df_metricas.groupby(['cd_cnae', 'descricao_cnae']).agg({
                'cnpj': 'count',
                'total_protocolos': 'sum'
            }).reset_index()
            
            df_cnae.columns = ['CNAE', 'Descrição', 'Empresas', 'Protocolos']
            df_cnae = df_cnae.sort_values('Empresas', ascending=False)
            
            # Top 20 CNAEs
            fig = px.bar(
                df_cnae.head(20),
                x='Empresas',
                y='Descrição',
                orientation='h',
                title='Top 20 CNAEs com Mais Cancelamentos',
                template=filtros['tema'],
                color='Protocolos',
                color_continuous_scale='Reds'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=700)
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(df_cnae, use_container_width=True, height=400)


def pagina_drill_down_empresa(dados, filtros, engine):
    """Drill-down detalhado de uma empresa."""
    st.markdown("<h1 class='main-header'>🔍 Análise Detalhada de Empresa</h1>", unsafe_allow_html=True)
    
    # Seleção da empresa
    col1, col2 = st.columns([3, 1])
    
    with col1:
        opcao_busca = st.radio("Buscar por:", ['CNPJ', 'Nome'], horizontal=True)
    
    df_top100 = dados.get('top100', pd.DataFrame())
    
    if opcao_busca == 'CNPJ':
        cnpj_input = st.text_input("Digite o CNPJ (apenas números):", max_chars=14)
        cnpj_selecionado = limpar_cnpj(cnpj_input) if cnpj_input else None
    else:
        if not df_top100.empty and 'nome_contribuinte' in df_top100.columns:
            empresas = df_top100[['cnpj', 'nome_contribuinte']].drop_duplicates()
            empresa_selecionada = st.selectbox(
                "Selecione a empresa:",
                empresas['cnpj'].tolist(),
                format_func=lambda x: f"{empresas[empresas['cnpj']==x]['nome_contribuinte'].iloc[0]}" if not empresas[empresas['cnpj']==x].empty else x
            )
            cnpj_selecionado = empresa_selecionada
        else:
            cnpj_selecionado = None
            st.warning("Lista de empresas não disponível.")
    
    if not cnpj_selecionado:
        st.info("Selecione uma empresa para análise detalhada.")
        return
    
    # Carregar detalhes
    with st.spinner(f"Carregando detalhes do CNPJ {cnpj_selecionado}..."):
        detalhes = carregar_detalhes_empresa(engine, cnpj_selecionado)
    
    # Verificar se encontrou dados
    tem_dados = any(not df.empty for df in detalhes.values())
    
    if not tem_dados:
        st.warning(f"Nenhum dado encontrado para o CNPJ {cnpj_selecionado}")
        return
    
    # Exibir informações
    df_metricas = detalhes.get('metricas', pd.DataFrame())
    df_scores = detalhes.get('scores', pd.DataFrame())
    df_indicios = detalhes.get('indicios', pd.DataFrame())
    df_creditos = detalhes.get('creditos', pd.DataFrame())
    df_base = detalhes.get('base', pd.DataFrame())
    
    # Header da empresa
    if not df_metricas.empty:
        info = df_metricas.iloc[0]
        
        st.markdown(f"### {info.get('razao_social', info.get('nome_contribuinte', 'N/A'))}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.caption(f"**CNPJ:** {formatar_cnpj(cnpj_selecionado)}")
            st.caption(f"**CNAE:** {info.get('cd_cnae', 'N/A')} - {info.get('descricao_cnae', 'N/A')}")
        
        with col2:
            st.caption(f"**Município:** {info.get('municipio', 'N/A')}")
            st.caption(f"**GERFE:** {info.get('gerencia_regional', 'N/A')}")
        
        with col3:
            st.caption(f"**Regime:** {info.get('regime_apuracao', 'N/A')}")
            st.caption(f"**Grupo:** {info.get('grupo_economico', 'N/A')}")
    
    st.divider()
    
    # Tabs de detalhes
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Scores", "📋 Indícios", "💰 Créditos", "📜 Histórico"])
    
    with tab1:
        if not df_scores.empty:
            score_info = df_scores.iloc[0]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                fig = criar_gauge_score(
                    float(score_info.get('score_total', 0)),
                    "Score Total"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = criar_gauge_score(
                    float(score_info.get('score_comportamento', 0)),
                    "Comportamento"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col3:
                fig = criar_gauge_score(
                    float(score_info.get('score_credito', 0)),
                    "Crédito"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col4:
                fig = criar_gauge_score(
                    float(score_info.get('score_indicios', 0)),
                    "Indícios"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Classificação
            classif = score_info.get('classificacao_risco_final', 'N/A')
            
            if classif == 'CRÍTICO':
                st.markdown("<div class='alert-critico'><b>⚠️ RISCO CRÍTICO</b></div>", unsafe_allow_html=True)
            elif classif == 'ALTO':
                st.markdown("<div class='alert-alto'><b>⚠️ RISCO ALTO</b></div>", unsafe_allow_html=True)
            elif classif == 'MÉDIO':
                st.markdown("<div class='alert-medio'><b>⚠️ RISCO MÉDIO</b></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='alert-positivo'><b>✅ RISCO BAIXO</b></div>", unsafe_allow_html=True)
        else:
            st.info("Scores não disponíveis para esta empresa.")
    
    with tab2:
        if not df_indicios.empty:
            ind_info = df_indicios.iloc[0]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total de Indícios", int(ind_info.get('qtde_indicios', 0)))
            
            with col2:
                st.metric("Indícios Graves", int(ind_info.get('qtde_indicios_graves', 0)))
            
            with col3:
                st.metric("Score Indícios", f"{float(ind_info.get('soma_scores_indicios', 0)):.0f}")
            
            # Classificação
            classif_ind = ind_info.get('classificacao_risco_indicios', 'N/A')
            st.info(f"**Classificação:** {classif_ind}")
            
            # Lista de indícios
            if 'lista_indicios_detalhada' in ind_info and ind_info['lista_indicios_detalhada']:
                st.subheader("📋 Lista de Indícios")
                indicios_lista = str(ind_info['lista_indicios_detalhada']).split(' | ')
                for ind in indicios_lista:
                    st.write(f"• {ind}")
        else:
            st.success("✅ Nenhum indício registrado para esta empresa.")
    
    with tab3:
        if not df_creditos.empty:
            cred_info = df_creditos.iloc[0]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                saldo_atual = float(cred_info.get('saldo_credor_atual', 0))
                st.metric("Saldo Atual", f"R$ {saldo_atual:,.2f}")
            
            with col2:
                credito_12m = float(cred_info.get('vl_credito_12m', 0))
                st.metric("Crédito 12m", f"R$ {credito_12m:,.2f}")
            
            with col3:
                credito_60m = float(cred_info.get('vl_credito_60m', 0))
                st.metric("Crédito 60m", f"R$ {credito_60m:,.2f}")
            
            # Alertas
            if cred_info.get('flag_saldo_alto_cancelada', 0) == 1:
                st.markdown("<div class='alert-critico'><b>⚠️ Saldo Alto + Cancelada</b></div>", unsafe_allow_html=True)
            
            if cred_info.get('flag_suspeita_valores_repetidos', 0) == 1:
                st.markdown("<div class='alert-alto'><b>⚠️ Valores Repetidos Suspeitos</b></div>", unsafe_allow_html=True)
            
            if cred_info.get('flag_crescimento_anormal_saldo', 0) == 1:
                st.markdown("<div class='alert-alto'><b>⚠️ Crescimento Anormal</b></div>", unsafe_allow_html=True)
        else:
            st.info("Dados de créditos não disponíveis.")
    
    with tab4:
        if not df_base.empty:
            st.subheader("📜 Histórico de Protocolos")
            
            colunas_hist = [
                'protocolo', 'data_inicio_protocolo', 'tipo_evento',
                'cod_motivo', 'parecer', 'nome_fiscal',
                'flag_cancelamento_automatico', 'flag_ainda_cancelada'
            ]
            
            colunas_existentes = [c for c in colunas_hist if c in df_base.columns]
            
            st.dataframe(
                df_base[colunas_existentes],
                use_container_width=True,
                height=400
            )
        else:
            st.info("Histórico não disponível.")


def pagina_machine_learning(dados, filtros, engine):
    """Sistema de Machine Learning para identificação de candidatas."""
    st.markdown("<h1 class='main-header'>🤖 Machine Learning - Identificação de Candidatas</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <b>Objetivo:</b> Treinar modelo de ML usando empresas historicamente canceladas e 
    aplicar em empresas ativas para identificar novas candidatas ao cancelamento.
    </div>
    """, unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Treinar Modelo", "🔮 Aplicar Predições", "📊 Análise do Modelo"])
    
    with tab1:
        st.subheader("Treinamento do Modelo")
        
        st.markdown("""
        O modelo utiliza as seguintes características:
        - **Score de Comportamento**: Baseado em protocolos, reincidência e persistência
        - **Score de Crédito**: Análise de saldos, créditos presumidos e padrões
        - **Score de Indícios**: Quantidade e gravidade de indícios NEAF
        - **Métricas Operacionais**: Total de protocolos, taxas de permanência/reativação
        """)
        
        # Carregar dados de scores
        df_scores = dados.get('scores_full', pd.DataFrame())
        
        if df_scores.empty:
            with st.spinner("Carregando dados para treinamento..."):
                df_scores = carregar_scores_completos(engine)
                dados['scores_full'] = df_scores
        
        if df_scores.empty:
            st.error("Dados de scores não disponíveis para treinamento.")
            return
        
        st.success(f"✅ {len(df_scores):,} empresas disponíveis para treinamento")
        
        # Configurações do modelo
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("Tamanho do conjunto de teste (%)", 10, 40, 20, 5) / 100
        
        with col2:
            n_estimators = st.slider("Número de árvores", 50, 200, 100, 25)
        
        if st.button("🚀 Treinar Modelo", type="primary"):
            with st.spinner("Treinando modelo..."):
                modelo, scaler, features, metricas = treinar_modelo_cancelamento(df_scores)
            
            if modelo is not None:
                # Salvar em session state
                st.session_state['modelo_ml'] = modelo
                st.session_state['scaler_ml'] = scaler
                st.session_state['features_ml'] = features
                st.session_state['metricas_ml'] = metricas
                
                st.success("✅ Modelo treinado com sucesso!")
                
                # Métricas
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Acurácia", f"{metricas['accuracy']*100:.1f}%")
                
                with col2:
                    st.metric("ROC-AUC", f"{metricas['roc_auc']:.3f}")
                
                with col3:
                    st.metric("Features", len(features))
                
                # Importância das features
                st.subheader("📊 Importância das Features")
                fig = criar_grafico_importancia_features(metricas['feature_importance'])
                st.plotly_chart(fig, use_container_width=True, key="ml_feature_importance_train")
                
                # Matriz de confusão
                st.subheader("📉 Matriz de Confusão")
                cm = metricas['confusion_matrix']
                
                fig = px.imshow(
                    cm,
                    labels=dict(x="Predito", y="Real", color="Quantidade"),
                    x=['Não Cancelada', 'Cancelada'],
                    y=['Não Cancelada', 'Cancelada'],
                    color_continuous_scale='Blues',
                    text_auto=True
                )
                st.plotly_chart(fig, use_container_width=True, key="ml_confusion_matrix")
            else:
                st.error("Erro no treinamento do modelo.")
    
    with tab2:
        st.subheader("Aplicar Predições em Empresas Ativas")
        
        # Verificar se modelo está treinado
        if 'modelo_ml' not in st.session_state:
            st.warning("⚠️ Treine o modelo primeiro na aba 'Treinar Modelo'.")
            return
        
        modelo = st.session_state['modelo_ml']
        scaler = st.session_state['scaler_ml']
        features = st.session_state['features_ml']
        
        st.success("✅ Modelo carregado da sessão")
        
        # Carregar empresas ativas
        if st.button("🔍 Carregar Empresas Ativas", type="primary"):
            with st.spinner("Carregando empresas ativas..."):
                df_ativas = carregar_empresas_ativas(engine)
                
                if not df_ativas.empty:
                    # Carregar indícios para enriquecimento
                    df_indicios = dados.get('indicios', pd.DataFrame())
                    if df_indicios.empty:
                        df_indicios = carregar_indicios(engine)
                    
                    # Aplicar modelo
                    df_predicoes = aplicar_modelo_empresas_ativas(
                        modelo, scaler, features, df_ativas, df_indicios
                    )
                    
                    st.session_state['predicoes_ml'] = df_predicoes
                    st.success(f"✅ Predições geradas para {len(df_predicoes):,} empresas")
        
        # Exibir resultados
        if 'predicoes_ml' in st.session_state:
            df_pred = st.session_state['predicoes_ml']
            
            # KPIs
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                criticas = len(df_pred[df_pred['classificacao_ml'] == 'CRÍTICO'])
                st.metric("🔴 CRÍTICO", f"{criticas:,}")
            
            with col2:
                muito_alto = len(df_pred[df_pred['classificacao_ml'] == 'MUITO ALTO'])
                st.metric("🟠 MUITO ALTO", f"{muito_alto:,}")
            
            with col3:
                alto = len(df_pred[df_pred['classificacao_ml'] == 'ALTO'])
                st.metric("🟡 ALTO", f"{alto:,}")
            
            with col4:
                prob_media = df_pred['prob_cancelamento'].mean() * 100
                st.metric("Prob. Média", f"{prob_media:.1f}%")
            
            st.divider()
            
            # Distribuição
            fig = px.histogram(
                df_pred,
                x='prob_cancelamento',
                nbins=50,
                title='Distribuição de Probabilidade de Cancelamento',
                template=filtros['tema'],
                color_discrete_sequence=['#1976d2']
            )
            st.plotly_chart(fig, use_container_width=True, key="ml_prob_distribution")
            
            # Top candidatas
            st.subheader("🎯 Top 50 Candidatas ao Cancelamento")
            
            colunas_pred = [
                'cnpj', 'razao_social', 'municipio', 'gerencia_regional',
                'cd_cnae', 'prob_cancelamento', 'classificacao_ml'
            ]
            
            colunas_existentes = [c for c in colunas_pred if c in df_pred.columns]
            
            df_display = df_pred.head(50)[colunas_existentes].copy()
            df_display.insert(0, 'Rank', range(1, len(df_display) + 1))
            
            st.dataframe(
                df_display.style.format({
                    'prob_cancelamento': '{:.1%}'
                }),
                use_container_width=True,
                height=600
            )
            
            # Exportar
            if st.button("📥 Exportar Candidatas (CSV)"):
                csv = df_pred.head(500)[colunas_existentes].to_csv(
                    index=False, encoding='utf-8-sig', sep=';'
                )
                st.download_button(
                    label="Baixar CSV",
                    data=csv.encode('utf-8-sig'),
                    file_name=f"candidatas_cancelamento_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime='text/csv'
                )
    
    with tab3:
        st.subheader("Análise Detalhada do Modelo")
        
        if 'metricas_ml' not in st.session_state:
            st.warning("⚠️ Treine o modelo primeiro.")
            return
        
        metricas = st.session_state['metricas_ml']
        features = st.session_state['features_ml']
        
        # Resumo do modelo
        st.markdown("### 📋 Resumo do Modelo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Tipo:** Random Forest Classifier  
            **Features utilizadas:** {len(features)}  
            **Acurácia:** {metricas['accuracy']*100:.1f}%  
            **ROC-AUC:** {metricas['roc_auc']:.3f}
            """)
        
        with col2:
            st.markdown("**Features principais:**")
            top_features = sorted(
                metricas['feature_importance'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            for f, imp in top_features:
                st.write(f"• {f}: {imp*100:.1f}%")
        
        # Gráfico de importância completo
        st.subheader("📊 Importância de Todas as Features")
        fig = criar_grafico_importancia_features(metricas['feature_importance'], top_n=len(features))
        st.plotly_chart(fig, use_container_width=True, key="ml_feature_importance_full")


def pagina_alertas_acoes(dados, filtros):
    """Sistema de alertas e ações prioritárias - APENAS EMPRESAS ATIVAS."""
    st.markdown("<h1 class='main-header'>🚨 Alertas e Ações Prioritárias</h1>", unsafe_allow_html=True)

    df_top100 = dados.get('top100', pd.DataFrame())

    if df_top100.empty:
        st.warning("⚠️ Dados não disponíveis.")
        return

    # CORREÇÃO: Filtrar apenas empresas ATIVAS (não canceladas atualmente)
    if 'flag_atualmente_cancelada' in df_top100.columns:
        total_antes = len(df_top100)
        df_top100 = df_top100[df_top100['flag_atualmente_cancelada'] == 0].copy()
        total_depois = len(df_top100)
        if total_antes != total_depois:
            st.info(f"📊 Exibindo alertas de {total_depois} empresas ATIVAS (excluídas {total_antes - total_depois} já canceladas)")

    if df_top100.empty:
        st.warning("⚠️ Nenhuma empresa ATIVA encontrada para alertas.")
        return

    st.markdown("""
    <div class='info-box'>
    <b>Objetivo:</b> Alertas priorizados para empresas <b>ATIVAS</b>, calculados automaticamente
    baseados em percentis de score, indícios graves e saldos credores.
    </div>
    """, unsafe_allow_html=True)

    # Verificar se coluna nivel_alerta existe
    if 'nivel_alerta' not in df_top100.columns:
        st.error("Coluna 'nivel_alerta' não encontrada. Execute o SQL atualizado.")
        return
    
    # KPIs de alertas
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        acao_imediata = len(df_top100[df_top100['nivel_alerta'] == 'AÇÃO IMEDIATA'])
        st.metric("🔴 AÇÃO IMEDIATA", f"{acao_imediata}")
    
    with col2:
        muito_urgente = len(df_top100[df_top100['nivel_alerta'] == 'MUITO URGENTE'])
        st.metric("🟠 MUITO URGENTE", f"{muito_urgente}")
    
    with col3:
        urgente = len(df_top100[df_top100['nivel_alerta'] == 'URGENTE'])
        st.metric("🟡 URGENTE", f"{urgente}")
    
    with col4:
        prioridade_alta = len(df_top100[df_top100['nivel_alerta'] == 'PRIORIDADE ALTA'])
        st.metric("⚪ PRIORIDADE ALTA", f"{prioridade_alta}")
    
    with col5:
        monitorar = len(df_top100[df_top100['nivel_alerta'] == 'MONITORAR'])
        st.metric("🟢 MONITORAR", f"{monitorar}")
    
    st.divider()
    
    # Gráfico de distribuição de alertas
    df_alertas_dist = df_top100.groupby('nivel_alerta').agg({
        'cnpj': 'count',
        'saldo_credor_atual': 'sum'
    }).reset_index()
    df_alertas_dist.columns = ['Nível de Alerta', 'Quantidade', 'Saldo Total']
    
    # Ordenar por prioridade
    ordem_alertas = ['AÇÃO IMEDIATA', 'MUITO URGENTE', 'URGENTE', 'PRIORIDADE ALTA', 'MONITORAR']
    df_alertas_dist['ordem'] = df_alertas_dist['Nível de Alerta'].map(
        {v: i for i, v in enumerate(ordem_alertas)}
    )
    df_alertas_dist = df_alertas_dist.sort_values('ordem')
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            df_alertas_dist,
            x='Nível de Alerta',
            y='Quantidade',
            title='Empresas por Nível de Alerta',
            color='Nível de Alerta',
            color_discrete_map={
                'AÇÃO IMEDIATA': '#b71c1c',
                'MUITO URGENTE': '#e65100',
                'URGENTE': '#f9a825',
                'PRIORIDADE ALTA': '#1976d2',
                'MONITORAR': '#388e3c'
            },
            template=filtros['tema']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            df_alertas_dist,
            x='Nível de Alerta',
            y='Saldo Total',
            title='Saldo Credor por Nível de Alerta',
            color='Nível de Alerta',
            color_discrete_map={
                'AÇÃO IMEDIATA': '#b71c1c',
                'MUITO URGENTE': '#e65100',
                'URGENTE': '#f9a825',
                'PRIORIDADE ALTA': '#1976d2',
                'MONITORAR': '#388e3c'
            },
            template=filtros['tema']
        )
        fig.update_yaxes(title_text="Saldo (R$)")
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Alertas detalhados por nível
    alertas_config = [
        ('AÇÃO IMEDIATA', 'alert-critico', '🔴'),
        ('MUITO URGENTE', 'alert-alto', '🟠'),
        ('URGENTE', 'alert-medio', '🟡')
    ]
    
    for nivel, css_class, emoji in alertas_config:
        df_nivel = df_top100[df_top100['nivel_alerta'] == nivel].head(10)
        
        if not df_nivel.empty:
            with st.expander(f"{emoji} {nivel} ({len(df_top100[df_top100['nivel_alerta'] == nivel])} empresas)", expanded=(nivel == 'AÇÃO IMEDIATA')):
                for _, row in df_nivel.iterrows():
                    st.markdown(f"""
                    <div class='{css_class}'>
                    <b>{row.get('nome_contribuinte', row.get('cnpj', 'N/A'))}</b><br>
                    CNPJ: {row.get('cnpj', 'N/A')} | Score: {row.get('score_total', 0):.1f} | 
                    Indícios Graves: {int(row.get('qtde_indicios_graves', 0))} | 
                    Saldo: R$ {row.get('saldo_credor_atual', 0):,.2f}<br>
                    <small>Município: {row.get('municipio', 'N/A')} | GERFE: {row.get('gerencia_regional', 'N/A')}</small>
                    </div>
                    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Filtro por nível de alerta
    st.subheader("📋 Lista Filtrada de Alertas")
    
    nivel_filtro = st.multiselect(
        "Filtrar por Nível de Alerta:",
        ordem_alertas,
        default=['AÇÃO IMEDIATA', 'MUITO URGENTE', 'URGENTE']
    )
    
    df_filtrado = df_top100[df_top100['nivel_alerta'].isin(nivel_filtro)].copy()
    df_filtrado = df_filtrado.sort_values('prioridade_num')
    
    colunas_alertas = [
        'nivel_alerta', 'prioridade_num', 'cnpj', 'nome_contribuinte',
        'classificacao_risco_final', 'score_total', 'qtde_indicios_graves',
        'saldo_credor_atual', 'municipio', 'gerencia_regional'
    ]
    
    colunas_existentes = [c for c in colunas_alertas if c in df_filtrado.columns]
    
    st.dataframe(
        df_filtrado[colunas_existentes].style.format({
            'score_total': '{:.1f}',
            'saldo_credor_atual': 'R$ {:,.2f}'
        }),
        use_container_width=True,
        height=600
    )
    
    # Exportar
    if st.button("📥 Exportar Alertas (CSV)"):
        csv = df_filtrado[colunas_existentes].to_csv(index=False, encoding='utf-8-sig', sep=';')
        st.download_button(
            label="Baixar CSV",
            data=csv.encode('utf-8-sig'),
            file_name=f"alertas_luciano_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv'
        )


def pagina_analise_motivos(dados, filtros, engine):
    """Análise por motivos de cancelamento."""
    st.markdown("<h1 class='main-header'>📋 Análise por Motivos de Cancelamento</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <b>Objetivo:</b> Analisar a distribuição dos motivos de cancelamento de IE,
    identificando os mais frequentes e seus padrões.
    </div>
    """, unsafe_allow_html=True)
    
    # Query para buscar motivos
    try:
        query_motivos = f"""
            SELECT 
                cod_motivo,
                tipo_evento,
                COUNT(DISTINCT protocolo) as qtde_protocolos,
                COUNT(DISTINCT cnpj) as qtde_empresas,
                SUM(flag_ainda_cancelada) as qtde_ainda_canceladas,
                SUM(flag_reativada) as qtde_reativadas,
                ROUND(SUM(flag_ainda_cancelada) * 100.0 / COUNT(*), 2) as taxa_permanencia,
                ROUND(SUM(flag_reativada) * 100.0 / COUNT(*), 2) as taxa_reativacao,
                MIN(data_inicio_protocolo) as primeira_ocorrencia,
                MAX(data_inicio_protocolo) as ultima_ocorrencia
            FROM {DATABASE}.luciano_base
            WHERE cod_motivo IS NOT NULL
            GROUP BY cod_motivo, tipo_evento
            ORDER BY qtde_protocolos DESC
        """
        df_motivos = pd.read_sql(query_motivos, engine)
    except Exception as e:
        st.error(f"Erro ao carregar motivos: {str(e)[:100]}")
        return
    
    if df_motivos.empty:
        st.warning("⚠️ Dados de motivos não disponíveis.")
        return
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_motivos = df_motivos['cod_motivo'].nunique()
        st.metric("Motivos Distintos", f"{total_motivos}")
    
    with col2:
        total_protocolos = df_motivos['qtde_protocolos'].sum()
        st.metric("Total Protocolos", f"{int(total_protocolos):,}")
    
    with col3:
        motivo_principal = df_motivos.iloc[0]['cod_motivo'] if not df_motivos.empty else 'N/A'
        st.metric("Motivo Mais Frequente", f"Cód. {motivo_principal}")
    
    with col4:
        taxa_media = df_motivos['taxa_permanencia'].mean()
        st.metric("Taxa Permanência Média", f"{taxa_media:.1f}%")
    
    st.divider()
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        # Top 10 motivos por volume
        df_top_motivos = df_motivos.nlargest(10, 'qtde_protocolos')
        
        fig = px.bar(
            df_top_motivos,
            x='qtde_protocolos',
            y='cod_motivo',
            orientation='h',
            title='Top 10 Motivos por Volume de Protocolos',
            color='taxa_permanencia',
            color_continuous_scale='RdYlGn_r',
            template=filtros['tema']
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        fig.update_yaxes(title_text="Código do Motivo")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distribuição por tipo de evento
        df_eventos = df_motivos.groupby('tipo_evento').agg({
            'qtde_protocolos': 'sum',
            'qtde_empresas': 'sum'
        }).reset_index()
        
        fig = px.pie(
            df_eventos,
            values='qtde_protocolos',
            names='tipo_evento',
            title='Distribuição por Tipo de Evento',
            color='tipo_evento',
            color_discrete_map={
                'CANCELAMENTO': '#c62828',
                'REATIVACAO_EXCLUSAO': '#1976d2',
                'REATIVACAO_AUTOMATICA': '#388e3c',
                'OUTRO': '#757575'
            },
            template=filtros['tema']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Análise de efetividade por motivo
    st.subheader("📊 Efetividade por Motivo")
    
    fig = px.scatter(
        df_motivos,
        x='qtde_protocolos',
        y='taxa_permanencia',
        size='qtde_empresas',
        color='tipo_evento',
        hover_data=['cod_motivo'],
        title='Volume vs Taxa de Permanência por Motivo',
        template=filtros['tema']
    )
    fig.update_xaxes(title_text="Quantidade de Protocolos")
    fig.update_yaxes(title_text="Taxa de Permanência (%)")
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Tabela completa
    st.subheader("📋 Lista Completa de Motivos")
    
    st.dataframe(
        df_motivos.style.format({
            'taxa_permanencia': '{:.1f}%',
            'taxa_reativacao': '{:.1f}%'
        }),
        use_container_width=True,
        height=500
    )
    
    # Drill-down por motivo selecionado
    st.divider()
    st.subheader("🔍 Detalhes por Motivo")
    
    motivo_selecionado = st.selectbox(
        "Selecione um motivo para detalhes:",
        df_motivos['cod_motivo'].tolist(),
        format_func=lambda x: f"Motivo {x} ({int(df_motivos[df_motivos['cod_motivo']==x]['qtde_protocolos'].iloc[0])} protocolos)"
    )
    
    if motivo_selecionado:
        try:
            query_detalhe = f"""
                SELECT 
                    cnpj, nome_contribuinte, municipio, gerencia_regional,
                    data_inicio_protocolo, tipo_evento, parecer,
                    flag_ainda_cancelada, flag_reativada, nome_fiscal
                FROM {DATABASE}.luciano_base
                WHERE cod_motivo = {motivo_selecionado}
                ORDER BY data_inicio_protocolo DESC
                LIMIT 100
            """
            df_detalhe = pd.read_sql(query_detalhe, engine)
            
            st.write(f"**Últimos 100 protocolos do Motivo {motivo_selecionado}:**")
            st.dataframe(df_detalhe, use_container_width=True, height=400)
        except Exception as e:
            st.error(f"Erro ao carregar detalhes: {str(e)[:100]}")
            
def pagina_sobre(dados, filtros):
    """Página sobre o sistema."""
    st.markdown("<h1 class='main-header'>ℹ️ Sobre o Projeto LUCIANO</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    ## Sistema de Machine Learning para Cancelamento de IE
    
    ### Versão 1.0
    
    ---
    
    ### 📋 Descrição
    
    O Projeto LUCIANO é uma iniciativa de Machine Learning desenvolvida pela Receita Estadual 
    de Santa Catarina para aprimorar a identificação proativa de empresas candidatas ao 
    cancelamento de Inscrição Estadual.
    
    O sistema analisa 60 meses de dados de administração tributária, processando centenas 
    de milhares de empresas para priorizar atividades de auditoria fiscal.
    
    ---
    
    ### 🎯 Objetivos
    
    1. **Identificar padrões** em empresas que tiveram IE cancelada
    2. **Treinar modelos de ML** usando dados históricos
    3. **Aplicar predições** em empresas ativas
    4. **Priorizar ações fiscais** baseado em scores de risco
    5. **Melhorar a efetividade** das ações de cancelamento
    
    ---
    
    ### 📊 Fontes de Dados
    
    - **Protocolos de Cancelamento**: `usr_sat_cadastro.ruc_protocolo`
    - **Cadastro de Contribuintes**: `usr_sat_ods.vw_ods_contrib`
    - **Indícios NEAF**: `neaf.empresa_indicio`
    - **Declarações DIME**: `usr_sat_ods.ods_decl_dime_raw`
    - **Créditos Presumidos**: `usr_sat_ods.vw_ods_dcip`
    
    ---
    
    ### 🔢 Componentes do Score
    
    | Componente | Peso | Descrição |
    |------------|------|-----------|
    | Comportamento | 25% | Protocolos, reincidência, persistência |
    | Crédito | 35% | Saldos, valores repetidos, crescimento |
    | Indícios | 40% | Quantidade, gravidade, tipos |
    
    ---
    
    ### 📈 Métricas de Classificação
    
    - **CRÍTICO**: Score ≥ 70
    - **ALTO**: Score ≥ 50
    - **MÉDIO**: Score ≥ 30
    - **BAIXO**: Score < 30
    
    ---
    
    ### 👨‍💻 Desenvolvimento
    
    **Projeto:** LUCIANO-ML  
    **Versão:** 1.0  
    **Data:** Dezembro 2025  
    **Desenvolvedor:** Equipe de Análise de Dados - SEF/SC
    
    ---
    
    ### 📞 Suporte
    
    Para dúvidas ou sugestões, entre em contato com a equipe de desenvolvimento.
    """)
    
    # Estatísticas da base
    st.divider()
    st.subheader("📊 Estatísticas da Base Atual")
    
    df_resumo = dados.get('resumo', pd.DataFrame())
    
    if not df_resumo.empty:
        resumo = df_resumo.iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Empresas", f"{int(resumo.get('total_empresas', 0)):,}")
        
        with col2:
            st.metric("Total Protocolos", f"{int(resumo.get('total_protocolos', 0)):,}")
        
        with col3:
            st.metric("Risco Crítico/Alto", f"{int(resumo.get('empresas_risco_critico', 0) + resumo.get('empresas_risco_alto', 0)):,}")
        
        with col4:
            st.metric("Atualização", datetime.now().strftime('%d/%m/%Y'))


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main():
    """Função principal do dashboard."""
    
    # Sidebar - Título
    st.sidebar.title("🎯 PROJETO LUCIANO")
    st.sidebar.caption("ML para Cancelamento de IE")
    
    # Conexão
    engine = get_impala_engine()
    
    if engine is None:
        st.error("Não foi possível conectar ao banco de dados.")
        return
    
    # Carregar dados resumidos
    with st.spinner('Carregando dados resumidos...'):
        dados = carregar_resumo_executivo(engine)
    
    if not dados:
        st.error("Falha no carregamento dos dados.")
        return
    
    # Info na sidebar
    df_resumo = dados.get('resumo', pd.DataFrame())
    if not df_resumo.empty:
        resumo = df_resumo.iloc[0]
        st.sidebar.success(f"✅ {int(resumo.get('total_empresas', 0)):,} empresas")
        st.sidebar.info(f"🔴 {int(resumo.get('empresas_risco_critico', 0)):,} críticas\n🟠 {int(resumo.get('empresas_risco_alto', 0)):,} alto risco")
    
    # Menu de navegação
    st.sidebar.markdown("---")
    st.sidebar.subheader("📋 Navegação")
    
    paginas = [
        "🎯 Dashboard Executivo",
        "📈 Análise Temporal",
        "📋 Análise por Motivos",  # NOVO
        "👤 Análise por Fiscal",
        "📋 Análise por Contador",  # NOVO
        "🏆 Ranking de Empresas",
        "🏭 Análise Setorial",
        "🔍 Drill-Down Empresa",
        "🤖 Machine Learning",
        "🚨 Alertas e Ações",
        "ℹ️ Sobre o Sistema"
    ]
    
    pagina_selecionada = st.sidebar.radio(
        "Selecione:",
        paginas,
        label_visibility="collapsed"
    )
    
    # Filtros
    filtros = criar_filtros_sidebar(dados)
    
    # Informações
    st.sidebar.markdown("---")
    with st.sidebar.expander("ℹ️ Informações"):
        st.caption(f"**Versão:** 1.0")
        st.caption(f"**Atualização:** {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        st.caption(f"**Dev:** Equipe SEF/SC")
    
    # Roteamento de páginas
    try:
        if pagina_selecionada == "🎯 Dashboard Executivo":
            pagina_dashboard_executivo(dados, filtros)
        
        elif pagina_selecionada == "📈 Análise Temporal":
            pagina_analise_temporal(dados, filtros)

        elif pagina_selecionada == "📋 Análise por Motivos":
            pagina_analise_motivos(dados, filtros, engine)
            
        elif pagina_selecionada == "👤 Análise por Fiscal":
            pagina_analise_fiscal(dados, filtros, engine)

        elif pagina_selecionada == "📋 Análise por Contador":
            pagina_analise_contador(dados, filtros, engine)
            
        elif pagina_selecionada == "🏆 Ranking de Empresas":
            pagina_ranking_empresas(dados, filtros)
        
        elif pagina_selecionada == "🏭 Análise Setorial":
            pagina_analise_setorial(dados, filtros, engine)
        
        elif pagina_selecionada == "🔍 Drill-Down Empresa":
            pagina_drill_down_empresa(dados, filtros, engine)
        
        elif pagina_selecionada == "🤖 Machine Learning":
            pagina_machine_learning(dados, filtros, engine)
        
        elif pagina_selecionada == "🚨 Alertas e Ações":
            pagina_alertas_acoes(dados, filtros)
        
        elif pagina_selecionada == "ℹ️ Sobre o Sistema":
            pagina_sobre(dados, filtros)
    
    except Exception as e:
        st.error(f"Erro ao carregar a página: {str(e)}")
        st.exception(e)
    
    # Rodapé
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #666;'>"
        f"Projeto LUCIANO v1.0 | SEF/SC | "
        f"{datetime.now().strftime('%d/%m/%Y %H:%M')}"
        f"</div>",
        unsafe_allow_html=True
    )


# =============================================================================
# EXECUÇÃO
# =============================================================================

if __name__ == "__main__":
    main()