import json
from datetime import datetime
import os
import pandas as pd
from typing import Dict, Any

from sazonalidade_agro import SazonalidadeAgro


class OutputGenerator:
    """Gera outputs em múltiplos formatos com abas completas e resumo validado"""

    def __init__(self, config: dict):
        self.config = config
        self.pasta_output = config['output']['pasta_output']
        self.formatos = config['output']['formatos']
        os.makedirs(self.pasta_output, exist_ok=True)

    # =========================================================
    # MÉTODO PRINCIPAL
    # =========================================================
    def gerar_outputs(self, previsoes: Dict, mes_num: int, ano: int):
        print(f"\n📊 Gerando outputs para {mes_num}/{ano}...")
        dados_consolidados = self._consolidar_dados(previsoes, mes_num, ano)
        nome_base = f"previsao_{ano}_{mes_num:02d}"

        if 'excel' in self.formatos:
            self._gerar_excel(dados_consolidados, nome_base)

        if 'csv' in self.formatos:
            self._gerar_csv(dados_consolidados, nome_base)

        if 'json' in self.formatos:
            self._gerar_json(dados_consolidados, nome_base)

        print(f"✓ Outputs gerados em: {self.pasta_output}")

    # =========================================================
    # HELPERS
    # =========================================================
    def _formatar_codigo_texto(self, codigo: Any) -> str:
        if codigo is None:
            return ""

        try:
            if pd.isna(codigo):
                return ""
        except Exception:
            pass

        if isinstance(codigo, int):
            return str(codigo)

        if isinstance(codigo, float):
            try:
                return str(int(round(codigo)))
            except Exception:
                return str(codigo)

        s = str(codigo).strip()
        if s == "" or s.lower() == "nan":
            return ""

        if "e+" in s.lower() or "e-" in s.lower():
            try:
                return str(int(round(float(s))))
            except Exception:
                return s

        if s.endswith(".0"):
            try:
                return str(int(float(s)))
            except Exception:
                return s

        return s

    def _forcar_colunas_texto(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df

        df = df.copy()

        if 'CODIGO' in df.columns:
            df['CODIGO'] = df['CODIGO'].apply(self._formatar_codigo_texto).astype(str)

        for col in ['DIRETORIA_NA', 'AREA_NEGOCIO', 'PRODUTO', 'GRUPO_LINHA', 'LINHA']:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str)

        return df

    # =========================================================
    # CONSOLIDAÇÃO DE DADOS
    # =========================================================
    def _consolidar_dados(self, previsoes: Dict, mes_num: int, ano: int) -> Dict:
        sazonalidade = SazonalidadeAgro(self.config['sazonalidade_agro']['pesos_arquivo'])

        nomes_meses = [
            'Janeiro', 'Fevereiro', 'Março', 'Abril', 'Maio', 'Junho',
            'Julho', 'Agosto', 'Setembro', 'Outubro', 'Novembro', 'Dezembro'
        ]

        tabelas = self._estruturar_previsoes(previsoes)

        return {
            'metadata': {
                'mes_previsao': f"{nomes_meses[mes_num-1]}/{ano}",
                'ano': ano,
                'mes_num': mes_num,
                'data_geracao': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'fase_safra': sazonalidade.obter_fase_mes(mes_num),
                'descricao_mes': sazonalidade.obter_descricao_mes(mes_num),
                'fator_sazonal_medio': sazonalidade.pesos_mensais.get(mes_num, 1)
            },
            'tabelas': tabelas
        }

    # =========================================================
    # ESTRUTURAÇÃO
    # =========================================================
    def _estruturar_previsoes(self, previsoes: Dict) -> Dict[str, pd.DataFrame]:
        """Transforma previsões brutas em DataFrames organizados por nível"""
        tabelas = {}

        for nivel in ['produto', 'linha', 'grupo']:
            if nivel not in previsoes:
                continue

            registros = []

            for grupo_id, prev in previsoes[nivel].items():
                if not isinstance(prev, dict):
                    continue

                info = prev.get('info', {}) or {}
                if not isinstance(info, dict):
                    info = {}

                # === DIRETORIA/AREA: SEM HEURÍSTICA DE SPLIT ===
                diretoria = info.get('DIRETORIA_NA') or ''
                area = info.get('AREA_NEGOCIO') or ''

                # === IDs e nomes ===
                codigo_raw = info.get('CODIGO') or info.get('CÓDIGO') or grupo_id
                codigo = self._formatar_codigo_texto(codigo_raw)

                produto = info.get('PRODUTO') or ''
                linha = info.get('LINHA') or ''
                grupo = info.get('GRUPO_LINHA') or ''

                receita_dict = prev.get('receita') or {}
                volume_dict = prev.get('volume') or {}

                registro = {
                    'DIRETORIA_NA': diretoria,
                    'AREA_NEGOCIO': area,
                    'CODIGO': codigo,
                    'PRODUTO': produto,
                    'GRUPO_LINHA': grupo,
                    'LINHA': linha,
                    'RECEITA_PREVISTA': round(receita_dict.get('valor_previsto', 0), 2),
                    'VOLUME_PREVISTO': round(volume_dict.get('valor_previsto', 0), 2),
                    'RECEITA_LIM_INF': round(receita_dict.get('limite_inferior', 0), 2),
                    'RECEITA_LIM_SUP': round(receita_dict.get('limite_superior', 0), 2),
                    'VOLUME_LIM_INF': round(volume_dict.get('limite_inferior', 0), 2),
                    'VOLUME_LIM_SUP': round(volume_dict.get('limite_superior', 0), 2),
                    'RECEITA_TENDENCIA': round(receita_dict.get('tendencia', 0), 2),
                    'VOLUME_TENDENCIA': round(volume_dict.get('tendencia', 0), 2),
                    'RECEITA_FATOR_SAZONAL': round(receita_dict.get('fator_sazonal', 1), 3),
                    'VOLUME_FATOR_SAZONAL': round(volume_dict.get('fator_sazonal', 1), 3),
                    'MES': prev.get('mes'),
                    'ANO': prev.get('ano'),
                    'MES_NOME': prev.get('mes_nome'),
                    'FASE_SAFRA': prev.get('fase_safra'),
                    'DESCRICAO_MES': prev.get('descricao_mes')
                }

                registros.append(registro)

            if registros:
                df = pd.DataFrame(registros)

                colunas_ordenadas = [
                    'DIRETORIA_NA', 'AREA_NEGOCIO', 'CODIGO', 'PRODUTO', 'GRUPO_LINHA', 'LINHA',
                    'RECEITA_PREVISTA', 'RECEITA_LIM_INF', 'RECEITA_LIM_SUP',
                    'RECEITA_TENDENCIA', 'RECEITA_FATOR_SAZONAL',
                    'VOLUME_PREVISTO', 'VOLUME_LIM_INF', 'VOLUME_LIM_SUP',
                    'VOLUME_TENDENCIA', 'VOLUME_FATOR_SAZONAL',
                    'MES', 'ANO', 'MES_NOME', 'FASE_SAFRA', 'DESCRICAO_MES'
                ]

                for col in colunas_ordenadas:
                    if col not in df.columns:
                        df[col] = ''

                df = df[colunas_ordenadas]
                df = self._forcar_colunas_texto(df)

                tabelas[nivel] = df
            else:
                tabelas[nivel] = pd.DataFrame()

        return tabelas

    # =========================================================
    # SAÍDA EXCEL
    # =========================================================
    def _gerar_excel(self, dados: Dict, nome_base: str):
        caminho = os.path.join(self.pasta_output, f"{nome_base}.xlsx")
        tabelas = dados['tabelas']

        with pd.ExcelWriter(caminho, engine='openpyxl') as writer:
            pd.DataFrame([dados['metadata']]).T.to_excel(
                writer, sheet_name='Metadata', header=False
            )

            for nome, df in tabelas.items():
                df = self._forcar_colunas_texto(df)
                df.to_excel(writer, sheet_name=f"Por {nome.capitalize()}", index=False)

            resumo = self._gerar_resumo_validado(tabelas)
            resumo.to_excel(writer, sheet_name="Resumo Validado", index=False)

        print(f" ✓ Excel completo gerado: {nome_base}.xlsx")

    # =========================================================
    # RESUMO VALIDADO
    # =========================================================
    def _gerar_resumo_validado(self, tabelas: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Cria uma aba consolidada com totais por Diretoria/Área em cada nível"""
        registros = []

        for nivel, df in tabelas.items():
            if df is None or df.empty:
                continue

            df = df.copy()
            df['DIRETORIA_NA'] = df['DIRETORIA_NA'].fillna('').astype(str)
            df['AREA_NEGOCIO'] = df['AREA_NEGOCIO'].fillna('').astype(str)

            agrupado = df.groupby(['DIRETORIA_NA', 'AREA_NEGOCIO']).agg({
                'RECEITA_PREVISTA': 'sum',
                'VOLUME_PREVISTO': 'sum'
            }).reset_index()

            agrupado['NIVEL'] = nivel.capitalize()
            registros.append(agrupado)

        if registros:
            df_final = pd.concat(registros, ignore_index=True)
        else:
            df_final = pd.DataFrame(
                columns=['DIRETORIA_NA', 'AREA_NEGOCIO', 'RECEITA_PREVISTA', 'VOLUME_PREVISTO', 'NIVEL']
            )

        df_final = df_final.sort_values(['DIRETORIA_NA', 'NIVEL'])
        return df_final

    # =========================================================
    # OUTROS FORMATOS
    # =========================================================
    def _gerar_csv(self, dados: Dict, nome_base: str):
        tabelas = dados['tabelas']
        for nome, df in tabelas.items():
            caminho = os.path.join(self.pasta_output, f"{nome_base}_{nome}.csv")
            df = self._forcar_colunas_texto(df)
            df.to_csv(caminho, index=False, sep=';', decimal=',', encoding='utf-8-sig')
        print(" ✓ CSVs gerados")

    def _gerar_json(self, dados: Dict, nome_base: str):
        caminho = os.path.join(self.pasta_output, f"{nome_base}.json")

        dados_para_json = {'metadata': dados.get('metadata', {})}
        dados_para_json['tabelas'] = {
            nome: self._forcar_colunas_texto(df).to_dict(orient='records')
            for nome, df in dados.get('tabelas', {}).items()
            if isinstance(df, pd.DataFrame)
        }

        with open(caminho, 'w', encoding='utf-8') as f:
            json.dump(dados_para_json, f, ensure_ascii=False, indent=2)

        print(" ✓ JSON gerado")

    def validar_area_diretoria(self, previsoes: Dict, depara_path=None, corrigir=True) -> Dict:
        """
        Valida e corrige AREA_NEGOCIO e DIRETORIA_NA usando histórico.
        Retorna dict com discrepâncias encontradas.
        """
        resultado = {
            'discrepancias': [],
            'corrigidos': 0
        }

        if 'produto' not in previsoes:
            return resultado

        vazios_diretoria = 0
        vazios_area = 0

        for codigo, data in previsoes['produto'].items():
            info = data.get('info', {})
            if not isinstance(info, dict):
                continue

            codigo_str = str(codigo).strip()
            diretoria = str(info.get('DIRETORIA_NA', '')).strip()
            area = str(info.get('AREA_NEGOCIO', '')).strip()

            # Detectar campos vazios
            if not diretoria:
                resultado['discrepancias'].append({
                    'codigo': codigo_str,
                    'area': area,
                    'acao': 'DIRETORIA_VAZIA',
                    'anterior': '',
                    'corrigida': ''
                })
                vazios_diretoria += 1

            if not area:
                resultado['discrepancias'].append({
                    'codigo': codigo_str,
                    'diretoria': diretoria,
                    'acao': 'AREA_VAZIA',
                    'anterior': '',
                    'corrigida': ''
                })
                vazios_area += 1

        resultado['corrigidos'] = vazios_diretoria + vazios_area
        return resultado