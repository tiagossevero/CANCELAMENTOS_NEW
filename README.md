# LUCIANO - Machine Learning para Cancelamento de IE

> Sistema de Machine Learning desenvolvido pela Secretaria de Estado da Fazenda de Santa Catarina (SEF/SC) para identificar empresas ativas com perfil de risco similar àquelas que tiveram sua Inscrição Estadual (IE) cancelada.

**Versão:** 1.1
**Última atualização:** Janeiro/2026

---

## Sumário

- [Visão Geral](#visão-geral)
- [Funcionalidades](#funcionalidades)
- [Tecnologias Utilizadas](#tecnologias-utilizadas)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Executando o Sistema](#executando-o-sistema)
- [Composição do Score](#composição-do-score)
- [Classificação de Risco](#classificação-de-risco)
- [Machine Learning](#machine-learning)
- [Fontes de Dados](#fontes-de-dados)
- [Páginas do Dashboard](#páginas-do-dashboard)
- [Notebooks de Análise](#notebooks-de-análise)
- [Melhorias Recentes](#melhorias-recentes)
- [Licença](#licença)

---

## Visão Geral

O **LUCIANO** é um sistema analítico que utiliza técnicas de Machine Learning para:

- Identificar empresas ativas com perfil comportamental similar a empresas canceladas
- Priorizar ações de fiscalização com base em scores de risco
- Fornecer insights acionáveis para a administração tributária
- Monitorar padrões temporais e setoriais de cancelamentos

O sistema analisa dados históricos de **60 meses** contemplando:
- ~243.000 empresas únicas
- ~277.000 protocolos de cancelamento
- Múltiplas dimensões de análise (comportamento, crédito, indícios)

---

## Funcionalidades

### Dashboard Interativo
- **Dashboard Executivo:** KPIs, métricas consolidadas e distribuição de alertas
- **Análise Temporal:** Tendências e padrões históricos de cancelamentos
- **Análise por Motivos:** Categorização das causas de cancelamento
- **Análise por Fiscal:** Métricas de desempenho por auditor fiscal
- **Análise por Contador:** Relacionamento empresa-contador e riscos associados
- **Ranking de Empresas:** Top 100 empresas ativas de maior risco
- **Análise Setorial:** Riscos por setor econômico (CNAE)
- **Drill-Down Empresa:** Análise detalhada individual
- **Machine Learning:** Treinamento, avaliação e aplicação de modelos
- **Alertas e Ações:** Sistema de priorização baseado em urgência
- **Sobre o Sistema:** Documentação e metodologia

### Sistema de Filtros
- Classificação de risco (Crítico, Alto, Médio, Baixo)
- Threshold de score ajustável
- Filtros geográficos (GERFE, Município)
- Filtros comportamentais (reincidente, cancelado, quantidade de protocolos)

### Cache Inteligente
- TTL de 3600 segundos para performance otimizada
- Atualização automática de dados

---

## Tecnologias Utilizadas

| Categoria | Tecnologias |
|-----------|-------------|
| **Framework Web** | Streamlit |
| **Linguagem** | Python 3.7+ |
| **Processamento de Dados** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn (RandomForest, GradientBoosting) |
| **Visualização** | Plotly (Express, Graph Objects) |
| **Banco de Dados** | Apache Impala via SQLAlchemy |
| **Big Data** | PySpark/Spark SQL (notebooks) |
| **Autenticação BD** | LDAP com SSL |

---

## Estrutura do Projeto

```
CANCELAMENTOS_NEW/
├── LUCIANO (1).py                    # Aplicação principal Streamlit (157 KB)
├── LUCIANO-ANALISES.ipynb            # Notebook de análises (2.1 MB)
├── LUCIANO-ANALISES-Copy1 (1).ipynb  # Cópia de análises (70 KB)
├── LUCIANO-ML.ipynb                  # Notebook de ML (305 KB)
├── LUCIANO.json                      # Exportação de dados Django (371 KB)
├── ANALISE_MELHORIAS_LOGICA.md       # Documentação de melhorias (14 KB)
└── README.md                         # Este arquivo
```

---

## Pré-requisitos

- Python 3.7 ou superior
- Acesso à rede interna SEF/SC
- Credenciais LDAP válidas
- Conectividade com o servidor Impala

---

## Instalação

1. **Clone o repositório:**
```bash
git clone <repository-url>
cd CANCELAMENTOS_NEW
```

2. **Crie um ambiente virtual (recomendado):**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

3. **Instale as dependências:**
```bash
pip install streamlit pandas numpy plotly sqlalchemy scikit-learn impyla
```

---

## Configuração

### Conexão com Banco de Dados

O sistema conecta-se ao Apache Impala com as seguintes configurações:

| Parâmetro | Valor |
|-----------|-------|
| Host | bdaworkernode02.sef.sc.gov.br |
| Porta | 21050 |
| Database | teste |
| Auth | LDAP |
| SSL | Habilitado |

### Tabelas Utilizadas

| Tabela | Descrição |
|--------|-----------|
| `luciano_resumo` | Métricas executivas consolidadas |
| `luciano_top100` | Top 100 empresas de maior risco |
| `luciano_scores` | Scores de risco de todas empresas |
| `luciano_metricas` | Métricas detalhadas por empresa |
| `luciano_indicios` | Indícios fiscais (NEAF) |
| `luciano_credito` | Informações de crédito/saldo |
| `luciano_base` | Dados de protocolos |
| `luciano_temporal` | Análise temporal |
| `luciano_fiscal` | Análise por auditor fiscal |
| `luciano_contabilista_*` | Análise por contador |

---

## Executando o Sistema

```bash
streamlit run "LUCIANO (1).py"
```

O sistema será iniciado em `http://localhost:8501`

### Fluxo de Acesso

1. Autenticar com senha do sistema
2. Inserir credenciais LDAP para conexão ao banco
3. Navegar pelas páginas do dashboard via menu lateral

---

## Composição do Score

O sistema utiliza um modelo de scoring ponderado:

| Componente | Peso | Descrição |
|------------|------|-----------|
| **Comportamento** | 25% | Protocolos, reincidência, persistência de cancelamento |
| **Crédito** | 35% | Saldos, valores duplicados, padrões de crescimento |
| **Indícios** | 40% | Quantidade, gravidade e tipos de indícios fiscais |

### Fórmula do Score Total

```
Score Total = (Score Comportamento × 0.25) +
              (Score Crédito × 0.35) +
              (Score Indícios × 0.40)
```

---

## Classificação de Risco

| Nível | Score | Ação Recomendada |
|-------|-------|------------------|
| CRÍTICO | ≥ 70 | Ação fiscal imediata |
| ALTO | 50-69 | Monitoramento próximo e priorização |
| MÉDIO | 30-49 | Acompanhamento regular |
| BAIXO | < 30 | Monitoramento básico |

### Sistema de Alertas

| Alerta | Descrição |
|--------|-----------|
| Ação Imediata | Empresas críticas requerendo intervenção urgente |
| Muito Urgente | Alta prioridade para análise |
| Urgente | Prioridade elevada |
| Alta Prioridade | Necessita atenção em breve |
| Monitorar | Acompanhamento contínuo |

---

## Machine Learning

### Algoritmo

```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced'
)
```

### Variável Alvo

- `flag_atualmente_cancelada`:
  - 1 = Empresa atualmente cancelada
  - 0 = Empresa ainda ativa

### Features (15 variáveis)

| Feature | Descrição |
|---------|-----------|
| `score_comportamento` | Score de comportamento |
| `score_credito` | Score de crédito |
| `score_indicios` | Score de indícios |
| `score_total` | Score total de risco |
| `total_protocolos` | Total de protocolos de cancelamento |
| `taxa_permanencia_cancelamento_perc` | Taxa de permanência no cancelamento |
| `taxa_reativacao_perc` | Taxa de reativação |
| `saldo_credor_atual` | Saldo credor atual |
| `vl_credito_60m` | Valor de crédito (60 meses) |
| `vl_credito_presumido_60m` | Crédito presumido (60 meses) |
| `qtde_indicios` | Quantidade de indícios |
| `soma_scores_indicios` | Soma dos scores de indícios |
| `qtde_indicios_graves` | Quantidade de indícios graves |
| `perc_valores_iguais_12m` | Percentual de valores iguais (12 meses) |
| `variacao_saldo_perc_60m` | Variação de saldo percentual (60 meses) |

### Métricas de Avaliação

- Acurácia
- ROC-AUC Score
- Matriz de Confusão
- Importância das Features
- Curvas Precision/Recall

### Workflow de ML

1. Treinar com dados históricos (empresas com desfecho de cancelamento)
2. Normalizar features com StandardScaler
3. Aplicar modelo em empresas ativas
4. Classificar em níveis de risco

---

## Fontes de Dados

### Dados Primários (60 meses)

| Fonte | Tabela | Descrição |
|-------|--------|-----------|
| Protocolos de Cancelamento | `usr_sat_cadastro.ruc_protocolo` | Histórico de protocolos |
| Cadastro de Contribuintes | `usr_sat_ods.vw_ods_contrib` | Dados cadastrais ativos |
| Indícios Fiscais | `neaf.empresa_indicio` | Indicadores graves (simulação, passivo fictício) |
| Declarações | `usr_sat_ods.ods_decl_dime_raw` | Declarações DIME |
| Créditos Presumidos | `usr_sat_ods.vw_ods_dcip` | Informações de crédito |

### Estatísticas do Dataset

- **243.053** empresas únicas analisadas
- **277.218** protocolos no período
- **95,63%** das empresas canceladas permanecem canceladas
- **9,41%** possuem indícios NEAF

---

## Páginas do Dashboard

### 1. Dashboard Executivo
Visão consolidada com KPIs, gauges de score e distribuição de alertas.

### 2. Análise Temporal
Tendências mensais e anuais de cancelamentos.

### 3. Análise por Motivos
Categorização e breakdown por motivo de cancelamento.

### 4. Análise por Fiscal
Performance e métricas por auditor fiscal responsável.

### 5. Análise por Contador
Top 50 contadores por exposição ao risco.

### 6. Ranking de Empresas
Top 100 empresas ativas com maior score de risco.

### 7. Análise Setorial
Distribuição de risco por setor econômico (CNAE).

### 8. Drill-Down Empresa
Análise individual detalhada com:
- Dados de crédito
- Indícios fiscais
- Métricas comportamentais
- Histórico de protocolos

### 9. Machine Learning
Interface para:
- Treinamento de modelo
- Visualização de métricas
- Importância de features
- Aplicação em empresas ativas

### 10. Alertas e Ações
Sistema de priorização com ações recomendadas.

### 11. Sobre o Sistema
Documentação, metodologia e composição dos scores.

---

## Notebooks de Análise

### LUCIANO-ANALISES.ipynb
Notebook principal contendo:
- Análise exploratória completa
- Queries Spark SQL para processamento distribuído
- Computação de métricas do dashboard
- Sumários estatísticos

### LUCIANO-ML.ipynb
Pipeline de Machine Learning:
- Preparação de features
- Treinamento e validação
- Avaliação de performance
- Cross-validation

---

## Melhorias Recentes

### Versão 1.1 (Janeiro/2026)

**Correções de Lógica:**
- Filtro de empresas ativas corrigido em todas as páginas (`flag_atualmente_cancelada = 0`)
- Ranking e Alertas agora excluem empresas já canceladas
- Função `carregar_empresas_ativas()` reescrita com LEFT JOIN correto

**Melhorias de UX:**
- Tooltips adicionados a todos indicadores
- Alertas com código de cores aprimorado
- Hierarquia visual melhorada

**Documentação:**
- Análise detalhada de melhorias em `ANALISE_MELHORIAS_LOGICA.md`

---

## Contribuindo

1. Crie uma branch para sua feature (`git checkout -b feature/nova-funcionalidade`)
2. Commit suas mudanças (`git commit -m 'Adiciona nova funcionalidade'`)
3. Push para a branch (`git push origin feature/nova-funcionalidade`)
4. Abra um Pull Request

---

## Licença

Sistema de uso interno da Secretaria de Estado da Fazenda de Santa Catarina (SEF/SC).

---

## Contato

**Secretaria de Estado da Fazenda de Santa Catarina**
Diretoria de Administração Tributária

---

*Desenvolvido com Streamlit e Scikit-learn*
