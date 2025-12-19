# PROJETO LUCIANO

## Sistema de Machine Learning para Cancelamento de Inscrição Estadual (IE)

Sistema de identificação de empresas ativas com perfil similar às canceladas, desenvolvido para a **Receita Estadual de Santa Catarina (SEF/SC)**.

---

## Sobre o Projeto

O **PROJETO LUCIANO** utiliza técnicas de Machine Learning para:

- Identificar empresas ativas com padrões comportamentais semelhantes às que tiveram sua Inscrição Estadual cancelada
- Priorizar ações de fiscalização proativa baseadas em scores de risco
- Analisar padrões de cancelamento por diversos critérios (temporal, setorial, fiscal, contabilista)
- Gerar alertas e rankings para tomada de decisão

---

## Tecnologias Utilizadas

| Tecnologia | Versão | Finalidade |
|------------|--------|------------|
| Python | 3.x | Linguagem principal |
| Streamlit | - | Framework para dashboard web |
| Pandas | - | Manipulação de dados |
| Scikit-learn | - | Modelos de Machine Learning |
| Plotly | - | Visualizações interativas |
| SQLAlchemy | - | Conexão com banco de dados |
| Apache Impala | - | Banco de dados analítico |

---

## Estrutura do Projeto

```
LUCIANO/
├── LUCIANO (1).py              # Dashboard principal (Streamlit)
├── LUCIANO-ANALISES.ipynb      # Notebook com análises exploratórias
├── LUCIANO-ML.ipynb            # Notebook de Machine Learning
├── LUCIANO.json                # Dados de configuração/amostra
└── README.md                   # Este arquivo
```

---

## Funcionalidades

### 1. Dashboard Executivo
- KPIs principais: total de empresas, protocolos, riscos
- Distribuição por nível de risco (Crítico, Alto, Médio, Baixo)
- Saldo credor total por classificação
- Top 10 empresas prioritárias
- Alertas de ação imediata

### 2. Análise Temporal
- Evolução dos cancelamentos ao longo do tempo
- Taxa de permanência e reativação por período
- Análise por ano e mês
- Comparativo de cancelamentos automáticos vs manuais

### 3. Análise por Motivos
- Distribuição dos motivos de cancelamento
- Taxa de permanência por motivo
- Identificação dos motivos mais frequentes

### 4. Análise por Fiscal
- Performance dos fiscais nos processos de cancelamento
- Volume de protocolos por fiscal
- Taxa de efetividade
- Drill-down individual com detalhes completos

### 5. Análise por Contador/Contabilista
- Ranking de contadores por risco
- Taxa de cancelamento da carteira
- Identificação de contadores com alta concentração de cancelamentos
- Análise de empresas vinculadas

### 6. Ranking de Empresas
- Ordenação por score total, saldo credor, indícios
- Filtros por classificação de risco
- Exportação para CSV

### 7. Análise Setorial
- Distribuição por Gerência Regional (GERFE)
- Análise por CNAE
- Concentração de riscos por setor

### 8. Drill-Down de Empresa
- Detalhes completos de uma empresa específica
- Histórico de protocolos
- Scores detalhados (comportamento, crédito, indícios)
- Situação de créditos

### 9. Machine Learning
- Treinamento de modelo Random Forest
- Features: comportamento, crédito, indícios, métricas operacionais
- Aplicação em empresas ativas para predição
- Identificação de novas candidatas ao cancelamento
- Métricas: Acurácia, ROC-AUC, Matriz de Confusão
- Importância das features

### 10. Alertas e Ações
- Níveis de alerta: Ação Imediata, Muito Urgente, Urgente, Prioridade Alta, Monitorar
- Priorização baseada em percentis de score
- Lista exportável de ações

---

## Sistema de Scores

O sistema utiliza múltiplos scores para classificação de risco:

| Score | Descrição | Peso |
|-------|-----------|------|
| Score de Comportamento | Baseado em protocolos, reincidência e persistência | - |
| Score de Crédito | Análise de saldos, créditos presumidos e padrões | - |
| Score de Indícios | Quantidade e gravidade de indícios NEAF | - |
| Score Total | Composição dos scores acima | - |

### Classificações de Risco
- **CRÍTICO**: Score > 70 ou indícios graves + saldo alto
- **ALTO**: Score entre 50-70
- **MÉDIO**: Score entre 30-50
- **BAIXO**: Score < 30

---

## Instalação

### Pré-requisitos

```bash
# Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows

# Instalar dependências
pip install streamlit pandas numpy plotly sqlalchemy scikit-learn
pip install impyla  # Driver para Impala
```

### Configuração

1. Configure as credenciais do Impala no arquivo `.streamlit/secrets.toml`:

```toml
[impala_credentials]
user = "seu_usuario"
password = "sua_senha"
```

2. Verifique a conectividade com o servidor Impala:
   - Host: `bdaworkernode02.sef.sc.gov.br`
   - Porta: `21050`
   - Database: `teste`

---

## Execução

### Dashboard Web

```bash
streamlit run "LUCIANO (1).py"
```

O dashboard estará disponível em: `http://localhost:8501`

### Notebooks Jupyter

```bash
jupyter notebook LUCIANO-ANALISES.ipynb
jupyter notebook LUCIANO-ML.ipynb
```

---

## Autenticação

O sistema possui proteção por senha na interface web.

**Senha padrão:** `luciano2025`

> Recomenda-se alterar a senha no arquivo principal antes do deploy em produção.

---

## Tabelas do Banco de Dados

O sistema utiliza as seguintes tabelas no Impala:

| Tabela | Descrição |
|--------|-----------|
| `luciano_base` | Dados base dos protocolos de cancelamento |
| `luciano_metricas` | Métricas calculadas por empresa |
| `luciano_scores` | Scores de risco por empresa |
| `luciano_indicios` | Indícios NEAF por empresa |
| `luciano_credito` | Dados de créditos |
| `luciano_resumo` | Resumo executivo |
| `luciano_top100` | Top 100 empresas prioritárias |
| `luciano_temporal` | Agregações temporais |
| `luciano_fiscal` | Métricas por fiscal |
| `luciano_contabilista_scores` | Scores por contador |
| `luciano_contabilista_base` | Empresas por contador |

---

## Modelo de Machine Learning

### Algoritmo
- **Random Forest Classifier**
  - n_estimators: 100
  - max_depth: 10
  - class_weight: balanced

### Features Principais
- `score_comportamento`
- `score_credito`
- `score_indicios`
- `total_protocolos`
- `taxa_permanencia_cancelamento_perc`
- `saldo_credor_atual`
- `qtde_indicios`
- `soma_scores_indicios`
- `qtde_indicios_graves`

### Saídas
- Probabilidade de cancelamento (0-100%)
- Classificação ML: CRÍTICO, MUITO ALTO, ALTO, MÉDIO, BAIXO

---

## Exportação de Dados

O sistema permite exportar dados em formato CSV com separador `;` e encoding `utf-8-sig` (compatível com Excel):

- Ranking de empresas
- Lista de alertas
- Dados de contadores
- Empresas por fiscal
- Candidatas ao cancelamento (ML)

---

## Arquitetura

```
┌─────────────────────────────────────────────────────────┐
│                     Frontend                             │
│                   (Streamlit)                            │
├─────────────────────────────────────────────────────────┤
│  Dashboard │ Análises │ ML │ Alertas │ Exportação       │
├─────────────────────────────────────────────────────────┤
│                     Backend                              │
│              (Python + Scikit-learn)                     │
├─────────────────────────────────────────────────────────┤
│                  Data Layer                              │
│            (SQLAlchemy + Impala)                         │
├─────────────────────────────────────────────────────────┤
│                   Database                               │
│              (Apache Impala - BDA)                       │
└─────────────────────────────────────────────────────────┘
```

---

## Fluxo de Trabalho

1. **Coleta de Dados**: Extração de dados do sistema ODS/SAT
2. **Processamento**: Cálculo de métricas e scores
3. **Análise**: Exploração via notebooks Jupyter
4. **Modelagem**: Treinamento de modelo ML
5. **Visualização**: Dashboard interativo
6. **Ação**: Exportação de listas prioritárias

---

## Desenvolvedores

**Equipe SEF/SC - Secretaria de Estado da Fazenda de Santa Catarina**

- Versão: 1.0
- Data: Dezembro 2025

---

## Licença

Projeto de uso interno da Secretaria de Estado da Fazenda de Santa Catarina.

---

## Changelog

### v1.0 (Dezembro 2025)
- Dashboard executivo completo
- Sistema de scores multi-dimensional
- Modelo de Machine Learning (Random Forest)
- Análises por fiscal, contador, temporal e setorial
- Sistema de alertas e priorização
- Exportação de dados em CSV
