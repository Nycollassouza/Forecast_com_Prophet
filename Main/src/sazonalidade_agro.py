import pandas as pd
import numpy as np
import yaml
from typing import Dict
import matplotlib.pyplot as plt
import os

class SazonalidadeAgro:
    """Gerencia pesos sazonais e features para o mercado Agro"""
    
class SazonalidadeAgro:
    """Gerencia pesos sazonais e features para o mercado Agro"""

    def __init__(self, config_path='config/sazonalidade_agro.yaml'):
        # 1) Normaliza o que vier do config.yaml
        #    (ex.: 'config/sazonalidade_agro.yaml' ou 'venv/config/sazonalidade_agro.yaml')
        if config_path.startswith("venv/") or config_path.startswith("venv\\"):
            # remove prefixo 'venv/' ou 'venv\'
            config_path = config_path[len("venv/"):]

        # 2) Se ainda for relativo, monta a partir da pasta venv
        if not os.path.isabs(config_path):
            # __file__ = C:\Modelo Previsão Prophet\venv\src\sazonalidade_agro.py
            venv_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            # venv_root = C:\Modelo Previsão Prophet\venv
            config_path = os.path.join(venv_root, config_path)
            # Ex.: 'config/sazonalidade_agro.yaml'
            #  -> C:\Modelo Previsão Prophet\venv\config\sazonalidade_agro.yaml

        # 3) (Opcional) debug pra garantir
        print("DEBUG SAZONALIDADE - config_path:", config_path)

        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.pesos_mensais = self._extrair_pesos()
        self.mes_inicio_safra = self.config['safra']['mes_inicio']
        self.mes_pico_safra = self.config['safra']['mes_pico']
        self.mes_fim_safra = self.config['safra']['mes_fim']
        
    def _extrair_pesos(self) -> Dict[int, float]:
        return {
            mes: info['peso'] 
            for mes, info in self.config['pesos_mensais'].items()
        }
    
    def adicionar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adiciona colunas de features sazonais ao DataFrame"""
        df = df.copy()
        df['mes'] = df['ds'].dt.month
        df['peso_agro'] = df['mes'].map(self.pesos_mensais)
        df['safra_alta'] = (
            (df['mes'] >= self.mes_inicio_safra) & 
            (df['mes'] <= self.mes_fim_safra)
        ).astype(int)
        df['intensidade_agro'] = df['peso_agro']
        df['distancia_pico'] = df['mes'].apply(self._calcular_distancia_pico)
        df['fase_safra_num'] = df['mes'].apply(self._obter_fase_numerica)
        return df
    
    def _calcular_distancia_pico(self, mes: int) -> float:
        distancia = abs(mes - self.mes_pico_safra)
        return min(distancia, 12 - distancia) / 6.0
    
    def _obter_fase_numerica(self, mes: int) -> float:
        fase = self.config['pesos_mensais'][mes]['fase']
        mapa_fases = {
            'entressafra': 0.0,
            'preparacao': 0.3,
            'safra': 1.0,
            'fim_safra': 0.7
        }
        return mapa_fases.get(fase, 0.5)
    
    def obter_descricao_mes(self, mes: int) -> str:
        return self.config['pesos_mensais'][mes]['descricao']
    
    def obter_fase_mes(self, mes: int) -> str:
        return self.config['pesos_mensais'][mes]['fase']
    
    def plotar_sazonalidade(self):
        """Gera visualização dos pesos sazonais"""
        meses = list(range(1, 13))
        pesos = [self.pesos_mensais[m] for m in meses]
        cores = ['red' if m < 6 else 'green' for m in meses]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(meses, pesos, color=cores, alpha=0.7)
        ax.axhline(y=1.0, color='black', linestyle='--', linewidth=1, label='Média')
        ax.axvline(x=self.mes_inicio_safra - 0.5, color='blue', 
                   linestyle='--', linewidth=2, label='Início Safra')
        
        ax.set_xlabel('Mês', fontsize=12)
        ax.set_ylabel('Peso Sazonal', fontsize=12)
        ax.set_title('Sazonalidade do Mercado Agro', fontsize=14, fontweight='bold')
        ax.set_xticks(meses)
        ax.set_xticklabels(['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
                            'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        return fig
