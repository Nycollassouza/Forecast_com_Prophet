from prophet import Prophet
from joblib import Parallel, delayed, dump, load
import pandas as pd
from sazonalidade_agro import SazonalidadeAgro
from pathlib import Path
import multiprocessing
import traceback
import os
from joblib import load
import re


class ModelTrainerAgro:
    """Treina modelos Prophet customizados para o mercado Agro (versão otimizada)"""

    def __init__(self, config: dict):
        self.config = config
        self.prophet_config = config['prophet']

        # instancia a sazonalidade AGRO (usada em preparar_dados_prophet)
        self.sazonalidade = SazonalidadeAgro(
            config['sazonalidade_agro']['pesos_arquivo']
        )

        # diretório base dos modelos
        self.model_output_path = Path(config['output']['model_dir'])
        self.modelos = {}

    # =====================================================================
    # PREPARAÇÃO DE DADOS
    # =====================================================================
    def preparar_dados_prophet(self, df: pd.DataFrame, metrica: str) -> pd.DataFrame:
        """
        Prepara dados para o Prophet.
        Recebe um 'df_grupo' (dados brutos) e o agrega por mês.
        """
        df_prophet = pd.DataFrame({
            'ds': pd.to_datetime(df['DATA']),
            'y_str': df[metrica].astype(str)  # Garante que é string
        })

        # Lógica de conversão "inteligente" (BR ou US)
        y_str_valid = df_prophet['y_str'].dropna()
        if y_str_valid.str.contains(',').any():
            # Formato BR ("1.234,56")
            y_numeric_str = df_prophet['y_str'].str.replace('.', '', regex=False)
            y_numeric_str = y_numeric_str.str.replace(',', '.', regex=False)
        else:
            # Formato US ("1,234.56" ou "1234.56")
            y_numeric_str = df_prophet['y_str'].str.replace(',', '', regex=False)

        df_prophet['y'] = pd.to_numeric(y_numeric_str, errors='coerce')
        df_prophet = df_prophet.fillna({'y': 0.0})
        df_prophet = df_prophet.groupby('ds').agg({'y': 'sum'}).reset_index()
        df_prophet = self.sazonalidade.adicionar_features(df_prophet)

        # =================================================================
        # === MUDANÇA DE AJUSTE (TUNING) ==================================
        # Apertamos o "freio" (cap) ainda mais.
        # Mudamos de 1.2x (R$ 532M) para 1.05x (Max Histórico + 5%).
        # =================================================================
        max_historico = df_prophet['y'].max()
        df_prophet['cap'] = max_historico * 1.05 + 1
        df_prophet['y'] = df_prophet['y'].clip(lower=0.01)

        return self._tratar_outliers(df_prophet)

    def _tratar_outliers(self, df: pd.DataFrame):
        Q1, Q3 = df['y'].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        limite_inferior = Q1 - 3 * IQR
        limite_superior = Q3 + 3 * IQR
        df['y'] = df['y'].clip(lower=max(0.01, limite_inferior), upper=limite_superior)
        return df

    # =====================================================================
    # CRIAÇÃO E TREINAMENTO DE MODELOS
    # =====================================================================
    def criar_modelo_prophet(self) -> Prophet:
        cfg = self.prophet_config
        model = Prophet(
            growth='logistic',  # Usando o "freio"
            yearly_seasonality=cfg['yearly_seasonality'],
            weekly_seasonality=cfg['weekly_seasonality'],
            daily_seasonality=cfg['daily_seasonality'],
            seasonality_mode=cfg['seasonality_mode'],  # 'additive'
            changepoint_prior_scale=cfg['changepoint_prior_scale'],
            seasonality_prior_scale=cfg['seasonality_prior_scale'],
            interval_width=cfg['interval_width'],
            mcmc_samples=cfg['mcmc_samples'],
            uncertainty_samples=cfg['uncertainty_samples']
        )

        model.add_seasonality(
            name='safra_anual', period=365.25,
            fourier_order=self.config['sazonalidade_agro']['fourier_order_anual'],
            prior_scale=15.0,
            mode=cfg['seasonality_mode']  # 'additive'
        )

        model.add_seasonality(
            name='ciclo_safra', period=182.625,
            fourier_order=self.config['sazonalidade_agro']['fourier_order_semestral'],
            prior_scale=10.0,
            mode=cfg['seasonality_mode']  # 'additive'
        )

        model.add_regressor('safra_alta', mode=cfg['seasonality_mode'])
        model.add_regressor('intensidade_agro', mode=cfg['seasonality_mode'])
        model.add_regressor('fase_safra_num', mode='additive')

        return model

    def treinar_modelo(self, df_prophet: pd.DataFrame, identificador: str) -> Prophet:
        if len(df_prophet) < 6:
            raise ValueError(f"Dados insuficientes ({len(df_prophet)} meses) para {identificador}")
        model = self.criar_modelo_prophet()
        model.fit(df_prophet)
        return model

    # =====================================================================
    # PARALELIZAÇÃO E CHECKPOINT
    # =====================================================================
    def _treinar_modelo_unico(self, grupo_id, df_grupo, metricas, nivel, base_dir):
        """Treina os modelos (receita e volume) para um grupo específico."""
        try:
            modelos_grupo = {}

            for metrica in metricas:
                tipo_metrica = 'receita' if 'RECEITA' in metrica else 'volume'

                # Correção do bug "1484 produtos"
                if isinstance(grupo_id, tuple) and len(grupo_id) == 1:
                    safe_name = str(grupo_id[0])
                elif isinstance(grupo_id, tuple):
                    safe_name = "__".join(str(x) for x in grupo_id)
                else:
                    safe_name = str(grupo_id)

                safe_name = safe_name.replace('/', '_').replace('\\', '_')
                caminho = base_dir / f"{safe_name}_{tipo_metrica}.pkl"

                if caminho.exists():
                    modelos_grupo[tipo_metrica] = load(caminho)
                    continue

                df_prophet = self.preparar_dados_prophet(df_grupo, metrica)
                modelo = self.treinar_modelo(df_prophet, f"{grupo_id}_{metrica}")
                dump(modelo, caminho, compress=3)
                modelos_grupo[tipo_metrica] = modelo

            # === INFO ENRIQUECIDO (com Diretoria/Área) ===
            if nivel == 'produto':
                cols_existentes = [c for c in [
                    'DIRETORIA_NA',
                    'ÁREA DE NEGÓCIO',
                    'CÓDIGO',
                    'PRODUTO',
                    'GRUPO_LINHA',
                    'LINHA'
                ] if c in df_grupo.columns]

                info = df_grupo.iloc[0][cols_existentes].to_dict()

                # Normaliza chave AREA_NEGOCIO
                if 'ÁREA DE NEGÓCIO' in info:
                    info['AREA_NEGOCIO'] = info.pop('ÁREA DE NEGÓCIO')

                modelos_grupo['info'] = info

            return grupo_id, modelos_grupo, None

        except Exception as e:
            return grupo_id, None, traceback.format_exc()

    def treinar_por_nivel(self, df: pd.DataFrame, nivel: str,
                          metricas=['RECEITA LÍQ.', 'QTDE P/1.000']):
        """Treina modelos Prophet em paralelo para o nível informado."""
        print(f"\nTreinando modelos por {nivel.upper()} (modo paralelo)...")

        colunas_grupo = self._obter_colunas_grupo(nivel)
        grupos = list(df.groupby(colunas_grupo))
        total = len(grupos)
        print(f" (Total de grupos a treinar: {total})")

        n_jobs = max(1, multiprocessing.cpu_count() - 1)
        base_dir = self.model_output_path / nivel.capitalize()
        base_dir.mkdir(parents=True, exist_ok=True)

        resultados = Parallel(n_jobs=n_jobs, prefer="processes", verbose=10)(
            delayed(self._treinar_modelo_unico)(grupo_id, df_grupo, metricas, nivel, base_dir)
            for grupo_id, df_grupo in grupos
        )

        modelos = {}
        erros = {}

        for grupo_id, modelo_dict, erro in resultados:
            # Correção do bug "1484 produtos"
            chave_limpa = grupo_id
            if isinstance(grupo_id, tuple) and len(grupo_id) == 1:
                chave_limpa = grupo_id[0]
            elif isinstance(grupo_id, tuple):
                chave_limpa = "__".join(str(x) for x in grupo_id)

            if modelo_dict is not None:
                modelos[chave_limpa] = modelo_dict
            else:
                erros[chave_limpa] = erro

        print(f" ✓ {len(modelos)}/{total} modelos treinados com sucesso.")

        if erros:
            print(f" ⚠ {len(erros)} grupos com erro. Exemplo de IDs:\n{list(erros.keys())[:3]}")
            # Mostra stack trace completo do primeiro erro
            first_key = list(erros.keys())[0]
            print(f"\n--- STACK TRACE DO PRIMEIRO ERRO ({first_key}) ---")
            print(erros[first_key])
            print("--- FIM STACK TRACE ---\n")

        return modelos

    # =====================================================================
    # UTILITÁRIOS
    # =====================================================================
    def _obter_colunas_grupo(self, nivel: str):
        if nivel == 'produto':
            return ['CÓDIGO']
        elif nivel == 'linha':
            return ['LINHA']
        elif nivel == 'grupo':
            return ['GRUPO_LINHA']
        raise ValueError(f"Nível inválido: {nivel}")

    def _agregar_dados(self, df: pd.DataFrame, colunas_grupo):
        # (não está sendo usada, mantida por compatibilidade)
        return df.groupby(colunas_grupo + ['MÊS', 'ANO', 'DATA']).agg({
            'RECEITA LÍQ.': 'sum',
            'QTDE P/1.000': 'sum'
        }).reset_index()

    # =====================================================================
    # CARREGAR MODELOS DO DISCO (SEM RETREINAR)
    # =====================================================================
    def carregar_modelos(self):
        base = self.model_output_path
        print("DEBUG MODELOS - base:", base)

        resultado = {}
        for nivel_dir in ['Produto', 'Linha', 'Grupo']:
            pasta = base / nivel_dir
            print("DEBUG MODELOS - verificando pasta:", pasta, "existe?", pasta.exists())
            modelos_nivel = {}
            if not pasta.exists():
                resultado[nivel_dir.lower()] = {}
                continue

            for p in pasta.glob("*.pkl"):
                nome = p.stem
                m = re.match(r"(.+)_(receita|volume)$", nome, flags=re.IGNORECASE)

                if m:
                    chave_str = m.group(1)
                    tipo = m.group(2).lower()
                    grupo_key = chave_str
                    try:
                        modelo_obj = load(p)
                    except Exception as e:
                        print(f" ⚠ Erro ao carregar {p}: {e}")
                        continue

                    if grupo_key not in modelos_nivel:
                        modelos_nivel[grupo_key] = {}
                    modelos_nivel[grupo_key][tipo] = modelo_obj
                else:
                    grupo_key = nome
                    try:
                        modelo_obj = load(p)
                    except Exception as e:
                        print(f" ⚠ Erro ao carregar {p}: {e}")
                        continue
                    modelos_nivel.setdefault(grupo_key, {})['receita_or_volume_unknown'] = modelo_obj

            # Cria um info mínimo para compatibilidade com o Forecaster/Output
            for grupo_key, d in modelos_nivel.items():
                if 'info' not in d:
                    if nivel_dir.lower() == 'produto':
                        d['info'] = {
                            'CODIGO': grupo_key,
                            'PRODUTO': grupo_key,
                            'GRUPO_LINHA': '',
                            'LINHA': ''
                        }
                    elif nivel_dir.lower() == 'linha':
                        d['info'] = {
                            'LINHA': grupo_key,
                            'GRUPO_LINHA': ''
                        }
                    elif nivel_dir.lower() == 'grupo':
                        d['info'] = {
                            'GRUPO_LINHA': grupo_key
                        }

            resultado[nivel_dir.lower()] = modelos_nivel

        self.modelos = resultado
        return resultado
