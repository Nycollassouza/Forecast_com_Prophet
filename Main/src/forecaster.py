from typing import Dict, List
import pandas as pd
from sazonalidade_agro import SazonalidadeAgro

class ForecasterAgro:
    """Gera previsões usando modelos Prophet treinados (Modo Aditivo)"""

    def __init__(self, config: dict, modelos: dict):
        self.config = config
        self.modelos = modelos
        self.sazonalidade = SazonalidadeAgro(config['sazonalidade_agro']['pesos_arquivo'])

    def prever_mes(self, mes_num: int, ano: int, nivel: str = 'produto') -> Dict:
        if nivel not in self.modelos:
            return {}

        previsoes = {}
        for grupo_id, modelos_grupo in self.modelos[nivel].items():
            try:
                # Normaliza estrutura
                if not isinstance(modelos_grupo, dict): continue

                info = modelos_grupo.get('info', {})
                res_grupo = {
                    'mes': mes_num, 'ano': ano,
                    'info': info,
                    'fase_safra': self.sazonalidade.obter_fase_mes(mes_num)
                }

                # Previsão Receita
                if 'receita' in modelos_grupo:
                    res_grupo['receita'] = self._prever_modelo(modelos_grupo['receita'], mes_num, ano)
                
                # Previsão Volume
                if 'volume' in modelos_grupo:
                    res_grupo['volume'] = self._prever_modelo(modelos_grupo['volume'], mes_num, ano)

                if 'receita' in res_grupo or 'volume' in res_grupo:
                    previsoes[grupo_id] = res_grupo

            except Exception as e:
                # print(f"Erro ao prever {grupo_id}: {e}")
                continue
                
        return previsoes

    def _prever_modelo(self, modelo, mes_num: int, ano: int) -> Dict:
        if modelo is None: return None

        try:
            # 1. Base de Cálculo (Média Recente Real)
            df_hist = modelo.history.sort_values('ds')
            
            if len(df_hist) >= 3:
                base_calculo = df_hist['y'].tail(3).mean()
            elif len(df_hist) > 0:
                base_calculo = df_hist['y'].mean()
            else:
                base_calculo = 0.0

            # 2. Peso Manual
            peso_manual = self.sazonalidade.pesos_mensais.get(mes_num, 1.0)
            
            # 3. Valor Inicial Proposto
            valor_proposto = base_calculo * peso_manual

            # === INTELIGÊNCIA DE TRAVA ANUAL (YoY CONSTRAINT) ===
            # Vamos olhar quanto foi vendido NESTE MESMO MÊS nos anos anteriores.
            # Isso impede que a previsão descole da realidade histórica do mês.
            
            # Filtra o histórico apenas para o mês alvo (ex: apenas Janeiros)
            mask_mesmo_mes = df_hist['ds'].dt.month == mes_num
            df_mesmo_mes = df_hist[mask_mesmo_mes]
            
            if not df_mesmo_mes.empty:
                # Pega o máximo que já foi vendido nesse mês na história
                max_historico_mes = df_mesmo_mes['y'].max()
                
                # Regra de Negócio: O mercado Agro raramente cresce mais que 15% 
                # sobre o recorde histórico do mês em um ano normal.
                teto_crescimento = 1.15 # Permite crescer até 15% acima do recorde
                
                teto_valor = max_historico_mes * teto_crescimento
                
                if valor_proposto > teto_valor:
                    # print(f" [YoY] Corte aplicado: {valor_proposto:.0f} -> {teto_valor:.0f} (Recorde era {max_historico_mes:.0f})")
                    valor_proposto = teto_valor

            # ====================================================

            # Recupera tendência informativa do Prophet
            try:
                future = modelo.make_future_dataframe(periods=12, freq='MS')
                forecast = modelo.predict(future)
                mask = (forecast['ds'].dt.year == ano) & (forecast['ds'].dt.month == mes_num)
                tendencia_info = forecast.loc[mask, 'trend'].values[0] if mask.any() else 0
            except:
                tendencia_info = 0

            return {
                'valor_previsto': valor_proposto,
                'limite_inferior': valor_proposto * 0.85,
                'limite_superior': valor_proposto * 1.15,
                'tendencia': tendencia_info,
                'fator_sazonal': peso_manual,
                'metodo_calculo': 'MEDIA_AJUSTADA_COM_TRAVA_YOY'
            }

        except Exception as e:
            return None
