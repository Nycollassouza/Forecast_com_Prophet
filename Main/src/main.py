import argparse
import yaml
import sys
from pathlib import Path
import pandas as pd
from typing import Dict
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from sazonalidade_agro import SazonalidadeAgro
from model_trainer import ModelTrainerAgro
from forecaster import ForecasterAgro
from output_generator import OutputGenerator

# Participação alvo (Jan/2025) – use os nomes exatamente como aparecem no seu output
TARGET_SHARE_DIRETORIA_JAN = {
    'MG': 0.450,
    'CERRADOS LESTE': 0.320,
    'SP/ES/CANA': 0.080,
    'CENTRO SUL': 0.050,
    'CERRADOS OESTE': 0.090,
    'OUTROS AGRICOLAS': 0.010
}


def carregar_config(config_path=None):
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'venv' / 'config' / 'config.yaml'
    if isinstance(config_path, str):
        config_path = Path(config_path)
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


class DataLoader:
    """Classe para carregamento de dados (com validações extras)"""

    def __init__(self, config):
        self.config = config
        self.input_file = config['data']['input_file']

    def carregar_csv(self):
        """Carrega arquivo CSV com cuidado nas colunas ANO e MÊS"""
        df = pd.read_csv(
            self.input_file,
            encoding=self.config['data'].get('encoding', 'utf-8-sig'),
            dtype=str,
            low_memory=False
        )

        df.columns = [c.strip() for c in df.columns]
        if 'ANO' not in df.columns or 'MÊS' not in df.columns:
            raise KeyError("O arquivo CSV precisa conter as colunas 'ANO' e 'MÊS'.")

        df['ANO'] = df['ANO'].astype(str).str.strip()
        df['MÊS'] = df['MÊS'].astype(str).str.strip().str.zfill(2)

        try:
            df['ANO_INT'] = df['ANO'].astype(int)
            df['MES_INT'] = df['MÊS'].astype(int)
        except Exception:
            df['ANO_INT'] = pd.to_numeric(df['ANO'], errors='coerce')
            df['MES_INT'] = pd.to_numeric(df['MÊS'], errors='coerce')
            print(" ⚠ Aviso: Existem valores não-numéricos em ANO ou MÊS; eles serão tratados como NaN.")

        df = df.dropna(subset=['ANO_INT', 'MES_INT']).copy()
        df['ANO'] = df['ANO_INT'].astype(int).astype(str)
        df['MÊS'] = df['MES_INT'].astype(int).astype(str).str.zfill(2)

        return df

    def criar_campo_data(self, df):
        """Cria campo DATA a partir de MÊS e ANO"""
        df['DATA'] = pd.to_datetime(
            df['ANO'].astype(str) + '-' + df['MÊS'].astype(str).str.zfill(2) + '-01',
            errors='coerce'
        )

        n_before = len(df)
        df = df.dropna(subset=['DATA']).copy()
        n_after = len(df)

        if n_after < n_before:
            print(f" ⚠ Removidas {n_before - n_after} linhas sem DATA válida.")

        return df

    def filtrar_diretoria(self, df, diretoria):
        """Filtra por diretoria específica"""
        return df[df['DIRETORIA'] == diretoria].copy()


# =========================================================================
# === FUNÇÃO AUXILIAR PARA FALLBACK =======================================
# =========================================================================

def _get_fallback_forecast(
    df_historico_grupo: pd.DataFrame,
    metric_map: Dict[str, str],
    total_meses_historico: int,
    mes_alvo: int,
    ano_alvo: int,
    info_cols: list
) -> Dict:
    """Fallback SEGURO (com info robusto: pega último valor válido do histórico)"""
    if df_historico_grupo.empty:
        return {}

    # === LER O PESO DO ARQUIVO YAML ===
    peso_fallback = 1.0
    try:
        caminho_saz = Path("venv/config/sazonalidade_agro.yaml")
        if not caminho_saz.exists():
            caminho_saz = Path("config/sazonalidade_agro.yaml")
        if caminho_saz.exists():
            with open(caminho_saz, 'r', encoding='utf-8') as f:
                saz_config = yaml.safe_load(f) or {}
            pesos = saz_config.get('pesos_mensais', {}) or {}
            val = pesos.get(mes_alvo)
            if val is None:
                val = pesos.get(str(mes_alvo))
            if isinstance(val, (int, float)):
                peso_fallback = float(val)
            else:
                peso_fallback = 1.2 if mes_alvo in [10, 11, 12] else 0.5
    except Exception:
        peso_fallback = 1.0

    # INFO ROBUSTO: último valor NÃO vazio
    def _last_valid(series: pd.Series) -> str:
        s = series.dropna().astype(str).str.strip()
        s = s[(s != '') & (s.str.lower() != 'nan')]
        return s.iloc[-1] if len(s) else ''

    info_dict = {}
    for col in info_cols:
        if col in df_historico_grupo.columns:
            info_dict[col] = _last_valid(df_historico_grupo[col])

    if 'CODIGO' not in info_dict and 'CÓDIGO' in df_historico_grupo.columns:
        codigo_val = _last_valid(df_historico_grupo['CÓDIGO'])
        info_dict['CÓDIGO'] = codigo_val
        info_dict['CODIGO'] = codigo_val

    fallback_entry = {
        'mes': mes_alvo,
        'ano': ano_alvo,
        'info': info_dict,
        'fallback': True
    }

    for metric_nome, metric_tipo in metric_map.items():
        if metric_nome not in df_historico_grupo.columns:
            fallback_entry[metric_tipo] = {
                'valor_previsto': 0.0,
                'fator_ajuste': peso_fallback,
                'metodo': 'SEM_COLUNA'
            }
            continue

        y_str = df_historico_grupo[metric_nome].astype(str).str.strip()
        y_str = y_str[(y_str != 'nan') & (y_str != '')]

        if y_str.empty:
            media_mensal = 0.0
        else:
            if y_str.str.contains(',', regex=False).any():
                y_clean = y_str.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
            else:
                y_clean = y_str.str.replace(',', '', regex=False)
            y_vals = pd.to_numeric(y_clean, errors='coerce').fillna(0.0)
            if len(y_vals) > 3:
                media_mensal = y_vals.tail(3).mean()
            else:
                media_mensal = y_vals.mean()

        # CORREÇÃO: Ajuste do threshold para valores em milhares
        if media_mensal > 50_000_000:
            media_mensal = media_mensal / 1000.0

        # NÃO MULTIPLICA POR PESO AQUI - peso será aplicado apenas no modelo Prophet
        valor_final = float(media_mensal)

        fallback_entry[metric_tipo] = {
            'valor_previsto': valor_final,
            'fator_ajuste': 1.0,  # Removido multiplicação por peso_fallback
            'metodo': 'MEDIA_RECENTE'
        }

    return fallback_entry


# =========================================================================
# === HELPER: CALCULAR RECEITA TOTAL ======================================
# =========================================================================

def _calcular_receita_total(previsoes: Dict) -> float:
    """Helper para calcular receita total de forma consistente"""
    total = 0.0
    for _, data in previsoes.get('produto', {}).items():
        valor = (data.get('receita') or {}).get('valor_previsto', 0.0)
        total += float(valor)
    return total


# =========================================================================
# === FUNÇÕES DE DISTRIBUIÇÃO DMA POR BU ==================================
# =========================================================================

def carregar_pesos_distribuicao(config_path: str = "config/pesos_distribuicao.yaml"):
    """
    Carrega o arquivo de configuração de pesos de distribuição por subdiretoria.
    Suporta caminhos relativos à pasta venv, semelhante à sazonalidade_agro.
    """
    import os

    # suporta prefixo "venv/"
    if config_path.startswith("venv/") or config_path.startswith("venv\\"):
        config_path = config_path[len("venv/"):]

    # se ainda for relativo, monta a partir da raiz da venv
    if not os.path.isabs(config_path):
        venvroot = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(venvroot, config_path)

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def aplicar_distribuicao_bu(previsoes: dict, mes_alvo: int, pesos_cfg: dict) -> dict:
    """
    Redistribui previsão entre BU1/BU2/BU3 MANTENDO o total de receita.
    Os pesos do YAML definem a proporção ALVO de cada BU sobre o total.
    fator_bu = (peso_alvo * total_receita) / receita_atual_bu
    """
    if not isinstance(previsoes, dict) or 'produto' not in previsoes:
        return previsoes

    diretorias_cfg = pesos_cfg.get("diretorias", {})
    if not diretorias_cfg:
        print("  ⚠ Nenhuma diretoria encontrada no YAML de pesos.")
        return previsoes

    # monta mapa: prefixo_bu -> peso_alvo
    pesos_alvo = {}
    for dir_nome, dir_info in diretorias_cfg.items():
        sub_cfg = dir_info.get("subdiretorias", {})
        if not sub_cfg:
            continue
        for bu_nome, bu_info in sub_cfg.items():
            pesos_mensais = bu_info.get("pesos_mensais", {})
            peso = pesos_mensais.get(mes_alvo) or pesos_mensais.get(str(mes_alvo))
            if peso is not None:
                pesos_alvo[bu_nome.upper()] = float(peso)

    if not pesos_alvo:
        print("  ⚠ Nenhum peso encontrado para o mês alvo no YAML de pesos.")
        return previsoes

    print(f"  DEBUG: pesos_alvo={pesos_alvo}")

    # 1) identifica BU de cada produto e soma receita atual por BU
    receita_por_bu = defaultdict(float)
    produto_bu_map = {}

    for codigo, data in previsoes.get("produto", {}).items():
        info = data.get("info") or {}
        area_neg = str(info.get("AREA_NEGOCIO", "")).strip().upper()

        bu_prefix = None
        for bu_nome in pesos_alvo.keys():
            if area_neg.startswith(bu_nome):
                bu_prefix = bu_nome
                break

        if not bu_prefix:
            continue

        produto_bu_map[codigo] = bu_prefix
        receita_atual = float((data.get("receita") or {}).get("valor_previsto", 0.0))
        receita_por_bu[bu_prefix] += receita_atual

    if not produto_bu_map:
        print("  ⚠ Nenhum produto com BU identificado.")
        return previsoes

    total_receita_bu = sum(receita_por_bu.values())
    if total_receita_bu <= 0:
        print("  ⚠ Receita total dos BUs é 0. Pulando distribuição.")
        return previsoes

    print(f"  DEBUG: receita_por_bu={dict(receita_por_bu)}")
    print(f"  DEBUG: total_receita_bu={total_receita_bu:,.2f}")

    # 2) calcula fator por BU para atingir proporção alvo mantendo total
    fatores_bu = {}
    for bu_nome, peso_alvo in pesos_alvo.items():
        receita_atual_bu = receita_por_bu.get(bu_nome, 0.0)
        if receita_atual_bu > 0:
            fatores_bu[bu_nome] = (peso_alvo * total_receita_bu) / receita_atual_bu
        else:
            fatores_bu[bu_nome] = 1.0

    print(f"  DEBUG: fatores_bu={fatores_bu}")

    # 3) aplica fatores nos produtos
    produtos_afetados = 0
    for codigo, data in previsoes.get("produto", {}).items():
        bu_prefix = produto_bu_map.get(codigo)
        if not bu_prefix:
            continue

        fator = fatores_bu.get(bu_prefix, 1.0)
        produtos_afetados += 1

        for metrica, bloco in list(data.items()):
            if metrica == "info":
                continue
            if isinstance(bloco, dict) and "valor_previsto" in bloco:
                bloco["valor_previsto"] = float(bloco.get("valor_previsto", 0.0)) * fator
                if "limite_inferior" in bloco:
                    bloco["limite_inferior"] = float(bloco.get("limite_inferior", 0.0)) * fator
                if "limite_superior" in bloco:
                    bloco["limite_superior"] = float(bloco.get("limite_superior", 0.0)) * fator

    print(f"  ✓ Distribuição BU: {produtos_afetados} produtos ajustados.")
    print(f"  ✓ Total mantido: R$ {total_receita_bu:,.2f}")
    return previsoes

# =========================================================================
# =========================================================================
# =========================================================================

def enriquecer_area_negocio_por_linha(previsoes: dict) -> dict:
    """
    Para produtos sem AREA_NEGOCIO, herda o valor de outro produto
    da mesma LINHA que já tenha AREA_NEGOCIO preenchida.
    """
    if 'produto' not in previsoes:
        return previsoes

    # 1) monta mapa: LINHA -> AREA_NEGOCIO (pega o primeiro válido encontrado)
    linha_para_area = {}
    for codigo, data in previsoes['produto'].items():
        info = data.get('info') or {}
        linha = str(info.get('LINHA', '')).strip()
        area = str(info.get('AREA_NEGOCIO', '')).strip()
        if linha and area and area.lower() not in ('', 'nan'):
            linha_para_area[linha] = area

    # 2) aplica nos produtos sem AREA_NEGOCIO
    preenchidos = 0
    nao_preenchidos = 0
    for codigo, data in previsoes['produto'].items():
        info = data.get('info') or {}
        area_atual = str(info.get('AREA_NEGOCIO', '')).strip()

        if area_atual and area_atual.lower() != 'nan':
            continue  # já tem, ignora

        linha = str(info.get('LINHA', '')).strip()
        area_herdada = linha_para_area.get(linha, '')

        if area_herdada:
            info['AREA_NEGOCIO'] = area_herdada
            preenchidos += 1
        else:
            nao_preenchidos += 1

    print(f"  ✓ AREA_NEGOCIO herdada por LINHA: {preenchidos} produtos preenchidos.")
    if nao_preenchidos > 0:
        print(f"  ⚠ {nao_preenchidos} produtos ainda sem AREA_NEGOCIO (linha não mapeável).")

    return previsoes


# =========================================================================
# === CALIBRAÇÃO POR DIRETORIA + AREA (SHRINKAGE) ==========================
# =========================================================================
def _calibrar_previsoes_por_dir_area(
    previsoes: Dict,
    df_historico: pd.DataFrame,
    metric_tipos: list,
    shrinkage_alpha: float = 0.3,
    coluna_receita_historica: str = 'RECEITA LÍQ.'
) -> Dict:
    """
    Calibra previsões por (DIRETORIA_NA, AREA_NEGOCIO) usando shrinkage:
    part_ajustada = (1-alpha)*part_hist + alpha*part_prev
    """
    print("\n📊 ETAPA 3.6: Calibração por Diretoria + Área Negócio (Shrinkage)")
    print("-" * 70)

    if df_historico is None or df_historico.empty:
        print(" ⚠ Histórico vazio. Pulando calibração.")
        return previsoes

    if 'DIRETORIA_NA' not in df_historico.columns or 'AREA_NEGOCIO' not in df_historico.columns:
        print(" ⚠ Histórico sem DIRETORIA_NA ou AREA_NEGOCIO. Pulando calibração.")
        return previsoes

    if coluna_receita_historica not in df_historico.columns:
        print(f" ⚠ Histórico sem coluna '{coluna_receita_historica}'. Pulando calibração.")
        return previsoes

    # Log do valor ANTES da calibração
    receita_antes = _calcular_receita_total(previsoes)
    print(f" 📍 Receita ANTES da calibração DIR+AREA: R$ {receita_antes:,.2f}")

    df_hist = df_historico.copy()
    df_hist['DIRETORIA_NA'] = df_hist['DIRETORIA_NA'].fillna('').astype(str).str.strip()
    df_hist['AREA_NEGOCIO'] = df_hist['AREA_NEGOCIO'].fillna('').astype(str).str.strip()

    receita_hist = pd.to_numeric(
        df_hist[coluna_receita_historica].astype(str)
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False),
        errors='coerce'
    ).fillna(0.0)
    df_hist['__RECEITA_HIST__'] = receita_hist

    df_hist = df_hist[(df_hist['DIRETORIA_NA'] != '') & (df_hist['AREA_NEGOCIO'] != '')].copy()
    hist_agg = df_hist.groupby(['DIRETORIA_NA', 'AREA_NEGOCIO'])['__RECEITA_HIST__'].sum()
    total_hist = float(hist_agg.sum())

    if total_hist <= 0:
        print(" ⚠ Receita histórica total <= 0. Pulando calibração.")
        return previsoes

    participacao_hist = hist_agg / total_hist
    print(f" ✓ Histórico calculado: {len(participacao_hist)} combinações (DIR, AREA)")

    previsoes_agg = defaultdict(float)
    for _, data in previsoes.get('produto', {}).items():
        info = data.get('info', {}) or {}
        if not isinstance(info, dict):
            continue
        dir_val = str(info.get('DIRETORIA_NA', '')).strip()
        area_val = str(info.get('AREA_NEGOCIO', '')).strip()
        if not dir_val or not area_val:
            continue
        for metric_tipo in metric_tipos:
            v = (data.get(metric_tipo) or {}).get('valor_previsto', 0.0)
            previsoes_agg[(dir_val, area_val, metric_tipo)] += float(v)

    fatores = {}
    for metric_tipo in metric_tipos:
        total_prev = float(sum(v for (d, a, m), v in previsoes_agg.items() if m == metric_tipo))
        if total_prev <= 0:
            continue
        for (dir_val, area_val) in participacao_hist.index:
            prev_atual = float(previsoes_agg.get((dir_val, area_val, metric_tipo), 0.0))
            part_prev = (prev_atual / total_prev) if total_prev > 0 else 0.0
            part_hist = float(participacao_hist.get((dir_val, area_val), 0.0))
            part_ajustada = (1 - shrinkage_alpha) * part_hist + shrinkage_alpha * part_prev
            fator = (part_ajustada / part_prev) if part_prev > 0 else 1.0
            fatores[(dir_val, area_val, metric_tipo)] = float(fator)

    print(f" ✓ Fatores calculados: {len(fatores)} (DIR, AREA, MÉTRICA)")

    for _, data in previsoes.get('produto', {}).items():
        info = data.get('info', {}) or {}
        if not isinstance(info, dict):
            continue
        dir_val = str(info.get('DIRETORIA_NA', '')).strip()
        area_val = str(info.get('AREA_NEGOCIO', '')).strip()
        if not dir_val or not area_val:
            continue
        for metric_tipo in metric_tipos:
            fator = fatores.get((dir_val, area_val, metric_tipo), 1.0)
            if metric_tipo in data and isinstance(data[metric_tipo], dict):
                data[metric_tipo]['valor_previsto'] = float(data[metric_tipo].get('valor_previsto', 0.0)) * fator
                data[metric_tipo]['limite_inferior'] = float(data[metric_tipo].get('limite_inferior', 0.0)) * fator
                data[metric_tipo]['limite_superior'] = float(data[metric_tipo].get('limite_superior', 0.0)) * fator

    # Log do valor DEPOIS da calibração
    receita_depois = _calcular_receita_total(previsoes)
    print(f" 📍 Receita DEPOIS da calibração DIR+AREA: R$ {receita_depois:,.2f}")
    print(f" 📍 Variação: {((receita_depois/receita_antes - 1) * 100):+.2f}%")
    print(" ✓ Calibração aplicada nos produtos.")

    return previsoes


def calibrar_por_share_diretoria(previsoes: Dict, target_share: Dict[str, float], alpha: float = 0.3) -> Dict:
    """
    Ajusta a receita prevista por DIRETORIA_NA para aproximar do target_share.
    Mantém o total previsto e redistribui multiplicando produtos por um fator da diretoria.
    alpha:
      0.0 -> força 100% target_share (mais agressivo)
      1.0 -> não altera nada (mantém o modelo)
      0.3 -> 70% target + 30% modelo (recomendado)
    """
    if 'produto' not in previsoes:
        return previsoes

    # Log do valor ANTES
    receita_antes = _calcular_receita_total(previsoes)
    print(f" 📍 Receita ANTES da calibração SHARE: R$ {receita_antes:,.2f}")

    # Soma receita atual por diretoria (com base no info enriquecido)
    soma_dir = defaultdict(float)
    total = 0.0
    for _, data in previsoes['produto'].items():
        info = data.get('info', {}) or {}
        if not isinstance(info, dict):
            continue
        diretoria = str(info.get('DIRETORIA_NA', '')).strip().upper()
        valor = float((data.get('receita') or {}).get('valor_previsto', 0.0))
        if diretoria:
            soma_dir[diretoria] += valor
            total += valor

    if total <= 0:
        return previsoes

    # Participação atual e participação ajustada (shrinkage para o target)
    fatores = {}
    for dir_name, target in target_share.items():
        d = str(dir_name).strip().upper()
        target = float(target)
        atual = soma_dir.get(d, 0.0)
        part_atual = atual / total if total > 0 else 0.0
        part_ajustada = (1 - alpha) * target + alpha * part_atual
        if part_atual > 0:
            fatores[d] = part_ajustada / part_atual
        else:
            fatores[d] = 1.0

    # Aplica fatores em todos os produtos (receita e limites)
    for _, data in previsoes['produto'].items():
        info = data.get('info', {}) or {}
        if not isinstance(info, dict):
            continue
        diretoria = str(info.get('DIRETORIA_NA', '')).strip().upper()
        fator = fatores.get(diretoria, 1.0)
        if 'receita' in data and isinstance(data['receita'], dict):
            data['receita']['valor_previsto'] = float(data['receita'].get('valor_previsto', 0.0)) * fator
            data['receita']['limite_inferior'] = float(data['receita'].get('limite_inferior', 0.0)) * fator
            data['receita']['limite_superior'] = float(data['receita'].get('limite_superior', 0.0)) * fator

    # Log do valor DEPOIS
    receita_depois = _calcular_receita_total(previsoes)
    print(f" 📍 Receita DEPOIS da calibração SHARE: R$ {receita_depois:,.2f}")
    print(f" 📍 Variação: {((receita_depois/receita_antes - 1) * 100):+.2f}%")

    return previsoes


# =========================================================================
# === RECALCULAR HIERARQUIA (FUNÇÃO ÚNICA) ================================
# =========================================================================

def _recalcular_hierarquia(previsoes: Dict, product_to_line_map: Dict, line_to_group_map: Dict,
                           metric_tipos: list, mes_alvo: int, ano_alvo: int):
    """Recalcula hierarquia Linha e Grupo a partir dos produtos (Bottom-Up)"""
    previsoes['linha'] = {}
    previsoes['grupo'] = {}

    linha_totals = defaultdict(lambda: {tipo: 0.0 for tipo in metric_tipos})
    grupo_totals = defaultdict(lambda: {tipo: 0.0 for tipo in metric_tipos})

    for codigo, data in previsoes['produto'].items():
        codigo_str = str(codigo).strip()
        linha_nome = product_to_line_map.get(codigo_str, 'DESCONHECIDO')
        for metric_tipo in metric_tipos:
            valor = (data.get(metric_tipo) or {}).get('valor_previsto', 0.0)
            linha_totals[linha_nome][metric_tipo] += float(valor)

    for linha_nome, totals in linha_totals.items():
        previsoes['linha'][linha_nome] = {'mes': mes_alvo, 'ano': ano_alvo}
        for metric_tipo, total_val in totals.items():
            previsoes['linha'][linha_nome][metric_tipo] = {'valor_previsto': float(total_val)}

        grupo_nome = line_to_group_map.get(linha_nome, 'DESCONHECIDO')
        for metric_tipo, total_val in totals.items():
            grupo_totals[grupo_nome][metric_tipo] += float(total_val)

    for grupo_nome, totals in grupo_totals.items():
        previsoes['grupo'][grupo_nome] = {'mes': mes_alvo, 'ano': ano_alvo}
        for metric_tipo, total_val in totals.items():
            previsoes['grupo'][grupo_nome][metric_tipo] = {'valor_previsto': float(total_val)}


# =========================================================================
# === VALIDAÇÃO DE ÁREAS NEGÓCIO vs DIRETORIA ============================
# =========================================================================

def corrigir_diretoria_por_regional(previsoes: Dict) -> Dict:
    """
    Corrige DIRETORIA_NA baseado em AREA_NEGOCIO (regional).
    DE-PARA completo com 59 mapeamentos.
    Atualizado com as 5 novas regionais solicitadas.
    """
    print("\nETAPA 3.6: Correção de Diretoria por Regional (DE-PARA)")
    print("-" * 70)

    # DE-PARA DEFINITIVO: Regional → Diretoria
    DE_PARA = {
        # CENTRO SUL
        'Regional MS': 'CENTRO SUL',
        'Regional MS/Norte': 'CENTRO SUL',
        'Regional Norte do RS': 'CENTRO SUL',
        'Regional Sul do RS': 'CENTRO SUL',
        'Regional PR Leste': 'CENTRO SUL',
        'Regional PR Oeste': 'CENTRO SUL',
        'Regional PR oeste': 'CENTRO SUL',
        'Regional SC': 'CENTRO SUL',

        # CERRADO LESTE
        'KEY ACCOUNT - NO/NE': 'CERRADO LESTE',
        'Reg GO/DF': 'CERRADO LESTE',
        'Reg Oeste da Bahia': 'CERRADO LESTE',
        'Reg Centro-Sul GO': 'CERRADO LESTE',
        'Reg. Centro-Sul GO': 'CERRADO LESTE',
        'Regional Go Centro': 'CERRADO LESTE',
        'Regional GO Centro': 'CERRADO LESTE',
        'Regional Go Norte': 'CERRADO LESTE',
        'Regional Grandes Go': 'CERRADO LESTE',
        'Regional Grandes GO': 'CERRADO LESTE',
        'Regional MA': 'CERRADO LESTE',
        'Regional PI': 'CERRADO LESTE',
        'Regional TO/PA/RR': 'CERRADO LESTE',

        # CERRADO OESTE
        'Key Account - MT': 'CERRADO OESTE',
        'Key account - MT': 'CERRADO OESTE',
        'Key account – MT': 'CERRADO OESTE',
        'Reg MT Oeste': 'CERRADO OESTE',
        'Reg MT/Centro Norte': 'CERRADO OESTE',
        'Regional MT/Norte': 'CERRADO OESTE',
        'Regional MT/RO': 'CERRADO OESTE',
        'Regional MT/Sul': 'CERRADO OESTE',
        'Regional MT/Vale': 'CERRADO OESTE',
        'Reg Guaporé/Rondônia': 'CERRADO OESTE',

        # MG
        'Grandes Contas MG': 'MG',
        'Neg Agríc-Floresta': 'MG',
        'Neg Agríc–Floresta': 'MG',
        'Reg Alto-Paranaiba': 'MG',
        'Reg Alto Paranaiba': 'MG',
        'Reg Triangulo/Mogia': 'MG',
        'Regional Norte MG': 'MG',
        'Regional Sudoeste MG': 'MG',
        'Regional Sul de MG': 'MG',
        'REG CENTRO OESTE MG': 'MG',

        # OUTROS AGRÍCOLAS
        'CORPORATIVO AGRÍCOLA': 'OUTROS AGRÍCOLAS',
        'Corporativo Agrícola': 'OUTROS AGRÍCOLAS',
        '(+) Potash Plus B2C': 'OUTROS AGRÍCOLAS',

        # SP/ES/CANA
        'Reg Norte ES/Sul BA': 'SP/ES/CANA',
        'Reg. SP Sudoeste': 'SP/ES/CANA',
        'Regional Cana Norte': 'SP/ES/CANA',
        'Regional Cana Sul/NE': 'SP/ES/CANA',
        'Regional ES/RJ': 'SP/ES/CANA',
        'Regional SH': 'SP/ES/CANA',
        'Regional SP HF': 'SP/ES/CANA',
        'REGIONAL SP HF': 'SP/ES/CANA',
        'Regional SP Noroeste': 'SP/ES/CANA',
        'Regional SP': 'SP/ES/CANA',
    }

    if 'produto' not in previsoes:
        print(" ⚠️ Sem previsões de produto")
        return previsoes

    corrigidos = 0
    nao_encontrados = set()

    for codigo, data in previsoes['produto'].items():
        info = data.get('info', {})
        if not isinstance(info, dict):
            continue
        regional = str(info.get('AREA_NEGOCIO', '')).strip()
        diretoria_atual = str(info.get('DIRETORIA_NA', '')).strip()

        if regional in DE_PARA:
            diretoria_correta = DE_PARA[regional]
            if diretoria_atual != diretoria_correta:
                info['DIRETORIA_NA'] = diretoria_correta
                corrigidos += 1
        elif regional and regional not in ['', 'nan', 'None']:
            nao_encontrados.add(regional)

    print(f" ✓ Correções aplicadas: {corrigidos}")
    if nao_encontrados:
        print(f" ⚠️ Regionais não mapeadas: {len(nao_encontrados)}")
        for reg in sorted(list(nao_encontrados))[:5]:
            print(f" • {reg}")
        if len(nao_encontrados) > 5:
            print(f" ... e mais {len(nao_encontrados) - 5}")
    else:
        print(" ✅ Todas as regionais mapeáveis foram processadas!")

    return previsoes


def executar_pipeline(config, mes_alvo=None, ano_alvo=None, retreinar=True,
                      shrinkage_alpha=0.3, usar_calibracao_share=False):
    """Executa pipeline completo de previsão"""

    if mes_alvo is None:
        mes_alvo = config['forecast']['mes_alvo']
    if ano_alvo is None:
        ano_alvo = config['forecast']['ano_alvo']

    print("=" * 70)
    print("🚀 PIPELINE DE PREVISÃO B2C - MERCADO AGRO")
    print(f"📅 Mês Alvo: {mes_alvo}/{ano_alvo}")
    print(f"📊 Shrinkage alpha: {shrinkage_alpha}")
    print(f"📊 Calibração por SHARE: {'SIM' if usar_calibracao_share else 'NÃO'}")
    print("=" * 70)

    # 1. Carregar dados
    print("\n📂 ETAPA 1: Carregamento de Dados")
    print("-" * 70)

    loader = DataLoader(config)
    df = loader.carregar_csv()
    df = loader.criar_campo_data(df)
    df_b2c = loader.filtrar_diretoria(df, config['data']['diretoria_filtro'])

    # Normaliza CÓDIGO (evita ' ' e inconsistências)
    if 'CÓDIGO' in df_b2c.columns:
        df_b2c['CÓDIGO'] = df_b2c['CÓDIGO'].astype(str).str.strip()
        df_b2c = df_b2c[df_b2c['CÓDIGO'] != ''].copy()

    print(f" ✓ Registros carregados: {len(df):,}")
    print(f" ✓ Registros {config['data']['diretoria_filtro']}: {len(df_b2c):,}")

    min_date = df_b2c['DATA'].min()
    max_date = df_b2c['DATA'].max()
    print(f" ✓ Período detectado no arquivo: {min_date} a {max_date}")

    filtro_inicio = pd.to_datetime(config['data']['history_start_date'])
    filtro_fim_desejado = pd.to_datetime(config['data']['history_end_date'])

    if max_date >= filtro_fim_desejado:
        df_b2c = df_b2c[
            (df_b2c['DATA'] >= filtro_inicio) &
            (df_b2c['DATA'] <= filtro_fim_desejado)
        ].copy()
    else:
        df_b2c = df_b2c[df_b2c['DATA'] >= filtro_inicio].copy()
        print(f" ⚠ O arquivo NÃO contém dados até {filtro_fim_desejado.date()} (max encontrado: {max_date.date()}).")

    print(f" ✓ Período considerado: {df_b2c['DATA'].min().date()} a {df_b2c['DATA'].max().date()}")

    # Guardar histórico JÁ filtrado para calibração
    df_historico_completo = df_b2c.copy()
    total_meses_historico = (
        (filtro_fim_desejado.year - filtro_inicio.year) * 12 +
        (filtro_fim_desejado.month - filtro_inicio.month) + 1
    )
    print(f" ✓ Total de meses no histórico: {total_meses_historico}")

    # 2. Treinar ou carregar modelos
    trainer = ModelTrainerAgro(config)

    if retreinar:
        print("\n🤖 ETAPA 2: Treinamento de Modelos")
        print("-" * 70)

        metricas_nomes = [m['nome'] for m in config['metricas']]
        modelos = {}
        for nivel in config['niveis_modelagem']:
            print(f" Treinando nível: {nivel}...")
            modelos[nivel] = trainer.treinar_por_nivel(
                df_b2c,
                nivel=nivel,
                metricas=metricas_nomes
            )

        print("\n💾 Modelos treinados e salvos.")
        trainer.modelos = modelos
    else:
        print("\n📦 ETAPA 2: Carregando Modelos Salvos")
        print("-" * 70)
        modelos = trainer.carregar_modelos()
        print(f" ✓ Modelos carregados: {dict((nivel, len(modelos_nivel)) for nivel, modelos_nivel in modelos.items())}")

    # 3. Gerar previsões
    print("\n🔮 ETAPA 3: Geração de Previsões")
    print("-" * 70)

    forecaster = ForecasterAgro(config, modelos)
    previsoes = {}
    for nivel in config['niveis_modelagem']:
        print(f" Prevendo {nivel}...")
        previsoes[nivel] = forecaster.prever_mes(mes_alvo, ano_alvo, nivel)
        print(f" ✓ {len(previsoes[nivel])} previsões geradas")

    # =========================================================================
    # === ETAPA 3.5: RECONCILIAÇÃO "BOTTOM-UP PURA" ==========================
    # =========================================================================

    print("\n🤝 ETAPA 3.5: Reconciliação Híbrida (Bottom-Up Pura)")
    print("-" * 70)

    metric_map = {m['nome']: m['tipo'] for m in config['metricas']}
    metric_tipos = [m['tipo'] for m in config['metricas']]

    # --- PASSO 1: FALLBACK PARA PRODUTOS FALTANTES ---
    all_products_historico = df_b2c['CÓDIGO'].unique()
    forecasted_products = set(previsoes.get('produto', {}).keys())

    # CORREÇÃO: Normaliza códigos para comparação
    forecasted_products_normalized = set(str(x).strip() for x in forecasted_products)
    all_products_normalized = set(str(x).strip() for x in all_products_historico)
    missing_products = all_products_normalized - forecasted_products_normalized

    print(f" Calculando fallback para {len(missing_products)} produtos (que falharam no treino)...")

    fallback_prod_forecasts = {}
    info_cols = ['CÓDIGO', 'CODIGO', 'PRODUTO', 'GRUPO_LINHA', 'LINHA', 'DIRETORIA_NA', 'AREA_NEGOCIO']

    for codigo in missing_products:
        codigo_str = str(codigo).strip()
        if not codigo_str:
            continue
        df_prod = df_b2c[df_b2c['CÓDIGO'] == codigo_str]
        fallback_entry = _get_fallback_forecast(
            df_prod,
            metric_map,
            total_meses_historico,
            mes_alvo,
            ano_alvo,
            info_cols=info_cols
        )
        if fallback_entry:
            fallback_entry.setdefault('info', {})
            fallback_entry['info'].setdefault('CÓDIGO', codigo_str)
            fallback_entry['info'].setdefault('CODIGO', codigo_str)
            fallback_prod_forecasts[codigo_str] = fallback_entry

    previsoes.setdefault('produto', {})
    previsoes['produto'].update(fallback_prod_forecasts)
    print(f" ✓ Previsão 'Produto' agora tem {len(previsoes['produto'])} itens.")

    # Log inicial
    receita_inicial = _calcular_receita_total(previsoes)
    print(f" 📍 Receita INICIAL (após fallback): R$ {receita_inicial:,.2f}")

    # --- PASSO 1.5: ENRIQUECER INFO DE TODOS OS PRODUTOS ---
    def _clean_str(x):
        if x is None:
            return ''
        s = str(x).strip()
        if s.lower() == 'nan':
            return ''
        return s

    def _pick_last_valid(df_subset: pd.DataFrame, col: str) -> str:
        if col not in df_subset.columns or df_subset.empty:
            return ''
        s = df_subset[col].dropna().astype(str).str.strip()
        s = s[(s != '') & (s.str.lower() != 'nan')]
        return s.iloc[-1] if len(s) else ''

    def _set_force(info: dict, k: str, v: str):
        v = _clean_str(v)
        if v != '':
            info[k] = v

    # garante CÓDIGO limpo no histórico
    df_b2c['CÓDIGO'] = df_b2c['CÓDIGO'].astype(str).str.strip()

    # df_info: último registro por código (rápido)
    df_info = df_b2c.sort_values('DATA').groupby('CÓDIGO', as_index=True).last()

    for codigo, data in previsoes['produto'].items():
        codigo_str = _clean_str(codigo)
        data.setdefault('info', {})
        if not isinstance(data['info'], dict):
            data['info'] = {}
        info = data['info']

        # Sempre seta código (nunca pode ficar em branco)
        info['CÓDIGO'] = codigo_str
        info['CODIGO'] = codigo_str

        # 1) Tenta pelo df_info (último registro por código)
        if codigo_str in df_info.index:
            row = df_info.loc[codigo_str]
            _set_force(info, 'PRODUTO', row.get('PRODUTO', ''))
            _set_force(info, 'LINHA', row.get('LINHA', ''))
            _set_force(info, 'GRUPO_LINHA', row.get('GRUPO_LINHA', ''))
            _set_force(info, 'DIRETORIA_NA', row.get('DIRETORIA_NA', ''))
            _set_force(info, 'AREA_NEGOCIO', row.get('AREA_NEGOCIO', ''))

        # 2) Se ainda estiver faltando, tenta "último válido" direto do histórico
        if _clean_str(info.get('DIRETORIA_NA', '')) == '' or _clean_str(info.get('AREA_NEGOCIO', '')) == '':
            df_cod = df_b2c[df_b2c['CÓDIGO'] == codigo_str].sort_values('DATA')
            if not df_cod.empty:
                if _clean_str(info.get('DIRETORIA_NA', '')) == '':
                    info['DIRETORIA_NA'] = _pick_last_valid(df_cod, 'DIRETORIA_NA')
                if _clean_str(info.get('AREA_NEGOCIO', '')) == '':
                    info['AREA_NEGOCIO'] = _pick_last_valid(df_cod, 'AREA_NEGOCIO')
                if _clean_str(info.get('PRODUTO', '')) == '':
                    info['PRODUTO'] = _pick_last_valid(df_cod, 'PRODUTO')
                if _clean_str(info.get('LINHA', '')) == '':
                    info['LINHA'] = _pick_last_valid(df_cod, 'LINHA')
                if _clean_str(info.get('GRUPO_LINHA', '')) == '':
                    info['GRUPO_LINHA'] = _pick_last_valid(df_cod, 'GRUPO_LINHA')

    print(" ✓ Info enriquecido (FORÇADO) para todos os produtos: DIRETORIA_NA e AREA_NEGOCIO.")

    sem_dir = 0
    sem_area = 0
    for _, data in previsoes['produto'].items():
        info = data.get('info', {}) or {}
        if not isinstance(info, dict):
            continue
        if str(info.get('DIRETORIA_NA', '')).strip() == '':
            sem_dir += 1
        if str(info.get('AREA_NEGOCIO', '')).strip() == '':
            sem_area += 1
    print(f" 🔎 Checagem pós-enriquecimento: sem DIRETORIA_NA={sem_dir}, sem AREA_NEGOCIO={sem_area}")

    # --- NOVO: herdar AREA_NEGOCIO por LINHA para produtos sem BU ---
    previsoes = enriquecer_area_negocio_por_linha(previsoes)

    # --- PASSO 2: MAPEAR HIERARQUIA ---
    product_to_line_map = {}
    line_to_group_map = {}

    for codigo, data in previsoes['produto'].items():
        linha_nome = 'DESCONHECIDO'
        grupo_nome = 'DESCONHECIDO'
        info = data.get('info', {}) or {}
        if isinstance(info, dict):
            linha_nome = info.get('LINHA', 'DESCONHECIDO')
            grupo_nome = info.get('GRUPO_LINHA', 'DESCONHECIDO')

        if (linha_nome == 'DESCONHECIDO' or grupo_nome == 'DESCONHECIDO') and 'CÓDIGO' in df_b2c.columns:
            try:
                info_row = df_b2c[df_b2c['CÓDIGO'] == str(codigo).strip()].iloc[0]
                linha_nome = info_row.get('LINHA', linha_nome)
                grupo_nome = info_row.get('GRUPO_LINHA', grupo_nome)
            except Exception:
                pass

        product_to_line_map[str(codigo).strip()] = linha_nome
        if linha_nome != 'DESCONHECIDO':
            line_to_group_map[linha_nome] = grupo_nome

    # --- PASSO 3: RECALCULAR 'LINHA' E 'GRUPO' (Bottom-Up) - PRIMEIRA VEZ ---
    print(" Recalculando 'Linha' e 'Grupo' a partir da soma dos produtos...")
    _recalcular_hierarquia(previsoes, product_to_line_map, line_to_group_map, metric_tipos, mes_alvo, ano_alvo)
    print(" ✓ Reconciliação 'Bottom-Up' concluída. Hierarquia 100% consistente.")

    # Corrigir Diretoria por Regional (DE-PARA)
    previsoes = corrigir_diretoria_por_regional(previsoes)

    # =========================================================================
    # === ETAPA 3.6: CALIBRAÇÃO - ESCOLHE APENAS UMA =========================
    # =========================================================================
    if usar_calibracao_share:
        # OPÇÃO 1: Calibração por Share de Diretoria
        print("\n📊 ETAPA 3.6: Calibração por Share de Diretoria")
        print("-" * 70)
        previsoes = calibrar_por_share_diretoria(
            previsoes,
            target_share=TARGET_SHARE_DIRETORIA_JAN,
            alpha=0.3
        )
        print(" ✓ Calibração por share de diretoria aplicada.")
    else:
        # OPÇÃO 2: Calibração por Diretoria + Área (DEFAULT)
        previsoes = _calibrar_previsoes_por_dir_area(
            previsoes=previsoes,
            df_historico=df_historico_completo,
            metric_tipos=metric_tipos,
            shrinkage_alpha=shrinkage_alpha,
            coluna_receita_historica='RECEITA LÍQ.'
        )

    # --- NOVO PASSO: DISTRIBUIÇÃO DMA EM BU1/BU2/BU3 ------------------------
    try:
        pesos_distribuicao = carregar_pesos_distribuicao("venv/config/pesos_distribuicao.yaml")
        previsoes = aplicar_distribuicao_bu(previsoes, mes_alvo, pesos_distribuicao)
        print(" ✓ Distribuição por BU aplicada com sucesso.")
    except Exception as e:
        print(f" ⚠ Falha ao aplicar distribuição por BU: {e}")



    # Recalcular LINHA e GRUPO após calibração + distribuição
    print(" Recalculando 'Linha' e 'Grupo' após calibração e distribuição DMA...")
    _recalcular_hierarquia(previsoes, product_to_line_map, line_to_group_map, metric_tipos, mes_alvo, ano_alvo)
    print(" ✓ Hierarquia recalculada.")

    receita_final = _calcular_receita_total(previsoes)
    print(f" 📍 Receita FINAL: R$ {receita_final:,.2f}")
    print(f" 📍 Variação total: {((receita_final/receita_inicial - 1) * 100):+.2f}%")

    # 4. Gerar outputs
    print("\n📊 ETAPA 4: Geração de Outputs")
    print("-" * 70)
    output_gen = OutputGenerator(config)
    output_gen.gerar_outputs(previsoes, mes_alvo, ano_alvo)

    # 5. Resumo final
    print("\n" + "=" * 70)
    print("✅ PIPELINE CONCLUÍDO COM SUCESSO!")
    print("=" * 70)

    if 'produto' in previsoes:
        receita_total = sum((p.get('receita') or {}).get('valor_previsto', 0.0) for p in previsoes['produto'].values())
        volume_total = sum((p.get('volume') or {}).get('valor_previsto', 0.0) for p in previsoes['produto'].values())
        print("\n📈 RESUMO DA PREVISÃO:")
        print(f" • Receita Líquida Total: R$ {receita_total:,.2f}")
        print(f" • Volume Total: {volume_total:,.2f} mil unidades")
        print(f" • Produtos previstos: {len(previsoes['produto'])}")

    sazonalidade = SazonalidadeAgro(config['sazonalidade_agro']['pesos_arquivo'])
    fase = sazonalidade.obter_fase_mes(mes_alvo)
    peso = sazonalidade.pesos_mensais[mes_alvo]
    print("\n🌾 CONTEXTO SAZONAL:")
    print(f" • Fase: {fase.upper()}")
    print(f" • Peso sazonal: {peso}")
    print(f" • Descrição: {sazonalidade.obter_descricao_mes(mes_alvo)}")
    print("\n")


def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description='Pipeline de Previsão de Faturamento B2C - Mercado Agro'
    )
    parser.add_argument('--mes', type=int, help='Mês para previsão (1-12)', default=None)
    parser.add_argument('--ano', type=int, help='Ano para previsão', default=None)
    parser.add_argument('--config', type=str, help='Caminho para arquivo de configuração',
                        default='venv/config/config.yaml')
    parser.add_argument('--no-retreinar', action='store_true', help='Não retreinar modelos, usar salvos')
    parser.add_argument('--meses', type=str, help='Múltiplos meses separados por vírgula (ex: 10,11,12)', default=None)
    parser.add_argument('--shrinkage', type=float, help='Shrinkage alpha (0-1). Padrão=0.3', default=0.3)
    parser.add_argument('--usar-share', action='store_true', help='Usar calibração por SHARE em vez de DIR+AREA')
    parser.add_argument('--visualizar-sazonalidade', action='store_true', help='Gerar gráfico de sazonalidade')

    args = parser.parse_args()
    config = carregar_config(args.config)

    if args.visualizar_sazonalidade:
        import matplotlib.pyplot as plt
        sazonalidade = SazonalidadeAgro(config['sazonalidade_agro']['pesos_arquivo'])
        fig = sazonalidade.plotar_sazonalidade()
        plt.show()
        return

    if args.meses:
        meses = [int(m.strip()) for m in args.meses.split(',') if m.strip()]
        ano = args.ano if args.ano else config['forecast']['ano_alvo']
        for mes in meses:
            print(f"\n{'=' * 70}")
            print(f"Executando para {mes}/{ano}")
            print('=' * 70)
            executar_pipeline(
                config,
                mes_alvo=mes,
                ano_alvo=ano,
                retreinar=(not args.no_retreinar),
                shrinkage_alpha=args.shrinkage,
                usar_calibracao_share=args.usar_share
            )
    else:
        executar_pipeline(
            config,
            mes_alvo=args.mes,
            ano_alvo=args.ano,
            retreinar=(not args.no_retreinar),
            shrinkage_alpha=args.shrinkage,
            usar_calibracao_share=args.usar_share
        )


if __name__ == '__main__':
    main()
