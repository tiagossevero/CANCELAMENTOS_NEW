# ANALISE DETALHADA - MELHORIAS NA LOGICA DO SISTEMA LUCIANO

**Data:** Janeiro/2026
**Objetivo:** Garantir que o sistema identifique corretamente empresas ATIVAS com perfil de risco, sem confundir com empresas ja CANCELADAS.

---

## 1. RESUMO EXECUTIVO

### Problema Identificado
O sistema possui uma **falha logica potencial** na separacao entre empresas ativas e canceladas em diferentes partes do codigo. O colega estava correto em questionar - ha pontos onde a analise pode estar sendo feita em empresas ja canceladas ao inves de apenas nas ativas.

### Impacto
- Rankings e alertas podem estar mostrando empresas JA CANCELADAS como "candidatas"
- O modelo de ML pode estar sendo aplicado incorretamente
- Decisoes fiscais podem estar sendo tomadas com base em dados incorretos

---

## 2. PROBLEMAS IDENTIFICADOS

### PROBLEMA 1: Tabelas `luciano_top100`, `luciano_scores`, `luciano_metricas` nao filtram por situacao cadastral atual

**Localizacao:** Linhas 333-339, 357-372, 414-427

**Codigo Atual:**
```python
# Linha 335 - Carrega top100 sem verificar se estao ATIVAS
query_top100 = f"SELECT * FROM {DATABASE}.luciano_top100 ORDER BY ranking_fiscalizacao LIMIT 100"

# Linha 366 - Agregacao de scores sem filtrar canceladas
query_scores_agg = f"""
    SELECT
        classificacao_risco_final,
        COUNT(*) as qtde,
        ...
    FROM {DATABASE}.luciano_scores
    GROUP BY classificacao_risco_final
"""

# Linha 417 - Carrega TODOS os scores sem verificar situacao
query = f"SELECT * FROM {DATABASE}.luciano_scores"
```

**Problema:** Essas queries retornam TODAS as empresas, incluindo as que JA ESTAO CANCELADAS. Se a tabela `luciano_scores` ou `luciano_top100` contem empresas com `flag_atualmente_cancelada = 1`, elas serao exibidas nos dashboards como se fossem candidatas.

**Correcao Necessaria:**
```python
# Linha 335 - CORRIGIDO
query_top100 = f"""
    SELECT * FROM {DATABASE}.luciano_top100
    WHERE flag_atualmente_cancelada = 0  -- Apenas ATIVAS
    ORDER BY ranking_fiscalizacao
    LIMIT 100
"""

# Linha 366 - CORRIGIDO (para mostrar apenas ativas em analise)
query_scores_agg = f"""
    SELECT
        classificacao_risco_final,
        COUNT(*) as qtde,
        SUM(saldo_credor_atual) as saldo_total,
        AVG(score_total) as score_medio,
        AVG(qtde_indicios) as indicios_medio
    FROM {DATABASE}.luciano_scores
    WHERE flag_atualmente_cancelada = 0  -- Apenas ATIVAS
    GROUP BY classificacao_risco_final
"""

# Linha 417 - Criar funcao separada para ativas
def carregar_scores_empresas_ativas(_engine):
    """Carrega scores apenas de empresas ATIVAS para analise."""
    try:
        query = f"""
            SELECT * FROM {DATABASE}.luciano_scores
            WHERE flag_atualmente_cancelada = 0
        """
        df = pd.read_sql(query, _engine)
        # ... resto do codigo
```

---

### PROBLEMA 2: Funcao `carregar_empresas_ativas()` usa logica de exclusao fragil

**Localizacao:** Linhas 455-492

**Codigo Atual:**
```python
query = f"""
    SELECT ...
    FROM usr_sat_ods.vw_ods_contrib ods
    WHERE ods.cd_sit_cadastral = 1
    AND ods.nu_cnpj NOT IN (
        SELECT DISTINCT cnpj FROM {DATABASE}.luciano_metricas WHERE cnpj IS NOT NULL
    )
    LIMIT 100000
"""
```

**Problemas:**
1. Exclui TODOS os CNPJs que estao em `luciano_metricas`, mesmo que a empresa tenha sido REATIVADA
2. Nao verifica a situacao ATUAL da empresa na propria tabela de metricas
3. O `NOT IN` com subquery pode ter performance ruim em grandes volumes

**Correcao Necessaria:**
```python
def carregar_empresas_ativas(_engine):
    """
    Carrega empresas ATIVAS para aplicar o modelo.
    CORRIGIDO: Usa LEFT JOIN para melhor performance e logica correta.
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
                COALESCE(sc.flag_atualmente_cancelada, 0) as ja_analisada,
                sc.score_total as score_existente
            FROM usr_sat_ods.vw_ods_contrib ods
            LEFT JOIN {DATABASE}.luciano_scores sc
                ON ods.nu_cnpj = sc.cnpj
            WHERE ods.cd_sit_cadastral = 1  -- Situacao cadastral ATIVA
            AND (
                sc.cnpj IS NULL  -- Nunca foi analisada
                OR sc.flag_atualmente_cancelada = 0  -- Foi analisada mas esta ATIVA
            )
            LIMIT 100000
        """
        df = pd.read_sql(query, _engine)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar empresas ativas: {str(e)[:100]}")
        return pd.DataFrame()
```

---

### PROBLEMA 3: Pagina de Ranking e Alertas mostra empresas canceladas

**Localizacao:** Linhas 2610-2724 (Ranking) e 3262-3427 (Alertas)

**Codigo Atual:**
```python
# Linha 2614 - Usa dados sem filtro
df_top100 = dados.get('top100', pd.DataFrame())

# Linha 3266 - Usa dados sem filtro
df_top100 = dados.get('top100', pd.DataFrame())
```

**Problema:** Os dados vem diretamente das tabelas sem filtrar por situacao cadastral atual. Empresas JA CANCELADAS aparecem nos rankings e alertas.

**Correcao Necessaria:**
```python
def pagina_ranking_empresas(dados, filtros):
    """Ranking de empresas prioritárias - APENAS ATIVAS."""
    st.markdown("<h1 class='main-header'>Ranking de Empresas Prioritarias</h1>", unsafe_allow_html=True)

    df_top100 = dados.get('top100', pd.DataFrame())

    if df_top100.empty:
        st.warning("Dados de ranking nao disponiveis.")
        return

    # CORRECAO: Filtrar apenas empresas ATIVAS
    if 'flag_atualmente_cancelada' in df_top100.columns:
        df_top100 = df_top100[df_top100['flag_atualmente_cancelada'] == 0].copy()
        st.info(f"Exibindo apenas empresas ATIVAS ({len(df_top100)} empresas)")
    else:
        st.warning("Campo 'flag_atualmente_cancelada' nao encontrado - dados podem incluir canceladas!")

    # ... resto do codigo
```

---

### PROBLEMA 4: Treinamento do modelo mistura conceitos

**Localizacao:** Linhas 693-749

**Codigo Atual:**
```python
def treinar_modelo_cancelamento(df_scores):
    """
    Treina modelo de ML para prever empresas candidatas ao cancelamento.
    Usa empresas ja canceladas como target positivo.
    """
    # ...
    y = df_scores['flag_atualmente_cancelada'].fillna(0).astype(int)
```

**Observacao:** Este trecho esta PARCIALMENTE correto. O modelo treina com:
- Classe 1 (positiva): Empresas que FORAM canceladas
- Classe 0 (negativa): Empresas que NAO foram canceladas

**Problema Potencial:** Se `df_scores` contem apenas empresas que passaram por algum processo de cancelamento (mesmo que reativadas), o modelo pode estar enviesado. A classe 0 deveria conter empresas que NUNCA tiveram problemas, nao apenas as que foram reativadas.

**Melhoria Sugerida:**
```python
def treinar_modelo_cancelamento(df_scores, df_ativas_sem_historico=None):
    """
    Treina modelo de ML para prever empresas candidatas ao cancelamento.

    MELHORIA: Inclui opcao de adicionar empresas ativas sem historico
    de problemas como exemplos negativos (classe 0) para melhor balanceamento.
    """

    if df_scores.empty:
        return None, None, None, None

    # Se fornecido, adiciona empresas ativas sem historico como classe 0
    if df_ativas_sem_historico is not None and not df_ativas_sem_historico.empty:
        df_ativas_sem_historico['flag_atualmente_cancelada'] = 0
        # Preencher features faltantes com 0 (sem problemas)
        for col in df_scores.columns:
            if col not in df_ativas_sem_historico.columns:
                df_ativas_sem_historico[col] = 0

        df_scores = pd.concat([df_scores, df_ativas_sem_historico], ignore_index=True)
        st.info(f"Modelo treinado com {len(df_scores)} empresas (incluindo ativas sem historico)")

    # ... resto do codigo igual
```

---

### PROBLEMA 5: Falta validacao de situacao cadastral ATUAL

**Localizacao:** Todo o sistema

**Problema:** O sistema usa `flag_atualmente_cancelada` que pode estar desatualizada. Uma empresa pode ter sido reativada apos a geracao dos dados.

**Correcao Necessaria - Criar funcao de validacao:**
```python
def validar_situacao_cadastral_atual(df, engine, coluna_cnpj='cnpj'):
    """
    Valida a situacao cadastral ATUAL das empresas consultando a view ODS.
    Retorna DataFrame com coluna 'situacao_atual_ods' atualizada.
    """
    if df.empty or coluna_cnpj not in df.columns:
        return df

    cnpjs = df[coluna_cnpj].dropna().unique().tolist()

    if not cnpjs:
        return df

    # Consulta situacao atual
    cnpjs_str = "','".join([str(c) for c in cnpjs])
    query = f"""
        SELECT
            nu_cnpj as cnpj,
            cd_sit_cadastral as cod_situacao_atual,
            nm_sit_cadastral as situacao_atual
        FROM usr_sat_ods.vw_ods_contrib
        WHERE nu_cnpj IN ('{cnpjs_str}')
    """

    try:
        df_situacao = pd.read_sql(query, engine)
        df = df.merge(df_situacao, on='cnpj', how='left')

        # Marcar empresas que estao ATIVAS atualmente
        df['esta_ativa_agora'] = df['cod_situacao_atual'] == 1

        return df
    except Exception as e:
        st.warning(f"Nao foi possivel validar situacao atual: {str(e)[:50]}")
        return df
```

---

## 3. FLUXO CORRIGIDO DO SISTEMA

### Fluxo Atual (COM PROBLEMAS):
```
1. Carrega luciano_scores (TODOS)
2. Treina modelo (usa flag_atualmente_cancelada)
3. Carrega empresas ativas (exclui por luciano_metricas)
4. Aplica modelo em ativas
5. Exibe rankings (SEM FILTRO)  <-- PROBLEMA
6. Exibe alertas (SEM FILTRO)   <-- PROBLEMA
```

### Fluxo Corrigido:
```
1. Carrega luciano_scores COM FILTRO de situacao
2. Para treino: usa TODAS (canceladas + ativas historicas)
3. Para predicao: APENAS flag_atualmente_cancelada = 0
4. Valida situacao atual via ODS antes de exibir
5. Exibe rankings APENAS empresas ATIVAS
6. Exibe alertas APENAS empresas ATIVAS
7. Permite opcao de ver historico (canceladas) separadamente
```

---

## 4. IMPLEMENTACAO SUGERIDA

### PASSO 1: Adicionar coluna de controle nas tabelas Impala
```sql
-- Garantir que todas as tabelas tenham a flag
ALTER TABLE teste.luciano_top100 ADD COLUMNS (flag_atualmente_cancelada INT);
ALTER TABLE teste.luciano_metricas ADD COLUMNS (flag_atualmente_cancelada INT);

-- Atualizar com base na situacao atual do ODS
UPDATE teste.luciano_scores s
SET flag_atualmente_cancelada =
    CASE WHEN EXISTS (
        SELECT 1 FROM usr_sat_ods.vw_ods_contrib o
        WHERE o.nu_cnpj = s.cnpj AND o.cd_sit_cadastral = 1
    ) THEN 0 ELSE 1 END;
```

### PASSO 2: Criar views separadas para ativas e canceladas
```sql
-- View apenas empresas ATIVAS
CREATE VIEW teste.vw_luciano_scores_ativas AS
SELECT * FROM teste.luciano_scores
WHERE flag_atualmente_cancelada = 0;

-- View apenas empresas CANCELADAS (para analise historica)
CREATE VIEW teste.vw_luciano_scores_canceladas AS
SELECT * FROM teste.luciano_scores
WHERE flag_atualmente_cancelada = 1;
```

### PASSO 3: Modificar o codigo Python

Ver secoes acima para cada correcao especifica.

---

## 5. CHECKLIST DE VALIDACAO

Apos implementar as correcoes, validar:

- [ ] `luciano_top100` contem apenas empresas com `flag_atualmente_cancelada = 0`?
- [ ] `luciano_scores` tem a flag atualizada com base no ODS?
- [ ] Pagina de Ranking mostra apenas empresas ATIVAS?
- [ ] Pagina de Alertas mostra apenas empresas ATIVAS?
- [ ] Funcao `carregar_empresas_ativas()` retorna empresas realmente ativas?
- [ ] Modelo ML esta sendo aplicado apenas em empresas ATIVAS?
- [ ] Existe opcao de ver historico de canceladas separadamente?

---

## 6. RESUMO DAS ALTERACOES NECESSARIAS

| Arquivo | Linha | Alteracao |
|---------|-------|-----------|
| LUCIANO (1).py | 335 | Adicionar WHERE flag_atualmente_cancelada = 0 |
| LUCIANO (1).py | 366 | Adicionar WHERE flag_atualmente_cancelada = 0 |
| LUCIANO (1).py | 417 | Criar funcao separada para scores de ativas |
| LUCIANO (1).py | 462-481 | Reescrever query com LEFT JOIN e validacao |
| LUCIANO (1).py | 2614 | Adicionar filtro apos carregar dados |
| LUCIANO (1).py | 3266 | Adicionar filtro apos carregar dados |
| Impala | - | Criar views separadas para ativas/canceladas |
| Impala | - | Garantir que flag esta atualizada em todas tabelas |

---

## 7. CONCLUSAO

O colega estava **CORRETO** ao questionar a logica. O sistema possui varios pontos onde empresas JA CANCELADAS podem aparecer como "candidatas" ou em rankings/alertas. As correcoes propostas garantem:

1. **Separacao clara** entre analise de ativas e historico de canceladas
2. **Validacao em tempo real** da situacao cadastral via ODS
3. **Queries otimizadas** com filtros apropriados
4. **Transparencia** para o usuario sobre o que esta sendo exibido

**Prioridade de implementacao:**
1. ALTA: Corrigir queries de top100 e alertas (impacto imediato)
2. ALTA: Corrigir funcao carregar_empresas_ativas (impacto no ML)
3. MEDIA: Criar views separadas no Impala
4. MEDIA: Adicionar validacao de situacao atual
5. BAIXA: Melhorar treinamento do modelo com classe negativa real
