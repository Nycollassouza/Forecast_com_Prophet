import os
import sys
import time
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox

import yaml
import pandas as pd

from model_trainer import ModelTrainerAgro
from forecaster import ForecasterAgro
from output_generator import OutputGenerator


# Paleta de cores baseada na marca 
PRIMARY = "#003A5D"
PRIMARY_LIGHT = "#005F8F"
ACCENT = "#00A6CC"
BG_MAIN = "#F4F6F8"
BG_PANEL = "#FFFFFF"
TEXT_MAIN = "#1F2933"
TEXT_MUTED = "#6B7280"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # ...\venv\src
VENV_ROOT = os.path.dirname(BASE_DIR)                   # ...\venv


# -------------------------
# Funções utilitárias
# -------------------------
def carregar_config(caminho_config: str = None) -> dict:
    if caminho_config is None:
        caminho_config = os.path.join(VENV_ROOT, "config", "config.yaml")
    with open(caminho_config, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def carregar_historico(cfg: dict) -> pd.DataFrame:
    caminho_rel = cfg["data"]["input_file"]
    if not os.path.isabs(caminho_rel):
        caminho = os.path.join(VENV_ROOT, caminho_rel)
    else:
        caminho = caminho_rel
    encoding = cfg["data"].get("encoding", "utf-8-sig")
    df = pd.read_csv(caminho, sep=";", encoding=encoding)
    return df


def preparar_modelos(cfg: dict, retreinar: bool) -> dict:
    trainer = ModelTrainerAgro(cfg)

    if retreinar:
        df_hist = carregar_historico(cfg)

        start = cfg["data"].get("history_start_date")
        end = cfg["data"].get("history_end_date")
        if start:
            df_hist = df_hist[df_hist["DATA"] >= start]
        if end:
            df_hist = df_hist[df_hist["DATA"] <= end]

        diretoria_filtro = cfg["data"].get("diretoria_filtro")
        if diretoria_filtro and "DIRETORIA" in df_hist.columns:
            df_hist = df_hist[df_hist["DIRETORIA"] == diretoria_filtro]

        modelos = {}
        for nivel in cfg["niveis_modelagem"]:
            modelos[nivel] = trainer.treinar_por_nivel(df_hist, nivel)
        return modelos
    else:
        return trainer.carregar_modelos()


def gerar_previsoes(cfg: dict, modelos: dict, mes: int, ano: int) -> dict:
    forecaster = ForecasterAgro(cfg, modelos)
    previsoes_prod = forecaster.prever_mes(mes, ano, nivel="produto")
    return {"produto": previsoes_prod}


def resumo_previsoes(previsoes: dict) -> dict:
    prod = previsoes.get("produto", {})
    receita_total = 0.0
    volume_total = 0.0
    for _, dados in prod.items():
        rec = (dados.get("receita") or {}).get("valor_previsto", 0.0)
        vol = (dados.get("volume") or {}).get("valor_previsto", 0.0)
        receita_total += rec
        volume_total += vol
    return {
        "receita_total": receita_total,
        "volume_total": volume_total,
        "qtd_produtos": len(prod),
    }


def get_pasta_output(cfg: dict) -> str:
    pasta = cfg["output"]["pasta_output"]
    if not os.path.isabs(pasta):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pasta = os.path.join(project_root, pasta.replace("venv/", ""))
    return pasta


def gerar_arquivos(cfg: dict, previsoes: dict, mes: int, ano: int) -> str:
    out = OutputGenerator(cfg)
    out.gerar_outputs(previsoes, mes, ano)
    return get_pasta_output(cfg)


# -------------------------
# Interface Tkinter
# -------------------------
class App(tk.Tk):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.title("ICL - Forecast Agro B2C")
        self.geometry("900x560")
        self.resizable(False, False)

        # tracking de tempos médios
        self.last_train_time = None
        self.last_forecast_time = None
        self.last_output_time = None

        # Fundo geral
        self.configure(bg=BG_MAIN)

        # Estilos ttk
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("TNotebook", background=BG_MAIN, borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            padding=(12, 6),
            background=BG_PANEL,
            foreground=TEXT_MAIN,
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", PRIMARY)],
            foreground=[("selected", "white")],
        )

        style.configure("TFrame", background=BG_PANEL)
        style.configure(
            "Header.TLabel",
            background=BG_PANEL,
            foreground=PRIMARY,
            font=("Segoe UI", 16, "bold"),
        )
        style.configure(
            "SubHeader.TLabel",
            background=BG_PANEL,
            foreground=TEXT_MUTED,
            font=("Segoe UI", 10),
        )
        style.configure("TLabel", background=BG_PANEL, foreground=TEXT_MAIN)
        style.configure("TCheckbutton", background=BG_PANEL, foreground=TEXT_MAIN)

        style.configure(
            "Accent.TButton",
            background=ACCENT,
            foreground="white",
            font=("Segoe UI", 10, "bold"),
            borderwidth=0,
            focusthickness=0,
            padding=(10, 4),
        )
        style.map(
            "Accent.TButton",
            background=[("active", PRIMARY_LIGHT)],
        )

        style.configure(
            "Secondary.TButton",
            background=BG_PANEL,
            foreground=PRIMARY,
            borderwidth=1,
            padding=(10, 4),
        )
        style.map(
            "Secondary.TButton",
            background=[("active", "#E5E7EB")],
        )

        style.configure(
            "Treeview",
            background="white",
            foreground=TEXT_MAIN,
            fieldbackground="white",
            bordercolor="#E5E7EB",
            rowheight=22,
        )
        style.configure(
            "Treeview.Heading",
            background=PRIMARY,
            foreground="white",
            font=("Segoe UI", 9, "bold"),
        )

        self._build_widgets()
        self.instalar_console()

    # redireciona stdout/stderr para o Text de log
    def instalar_console(self):
        app = self

        class GUIConsole:
            def __init__(self, app_ref):
                self.app = app_ref

            def write(self, msg):
                if "Task" in msg:
                    return
                msg = msg.rstrip("\n")
                if msg:
                    self.app.log(msg)

            def flush(self):
                pass

        sys.stdout = GUIConsole(app)
        sys.stderr = GUIConsole(app)

    def _build_widgets(self):
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # --- Aba 1: Previsão ---
        frame_prev = ttk.Frame(notebook, padding=10, style="TFrame")
        notebook.add(frame_prev, text="Previsão")

        self._build_tab_previsao(frame_prev)

        # --- Aba 2: Arquivos ---
        frame_files = ttk.Frame(notebook, padding=10, style="TFrame")
        notebook.add(frame_files, text="Arquivos")

        self._build_tab_arquivos(frame_files)

    # ----------------- Aba Previsão -----------------
    def _build_tab_previsao(self, parent):
        frame_top = ttk.Frame(parent, style="TFrame")
        frame_top.pack(fill=tk.X)

        ttk.Label(
            frame_top,
            text="Previsão Agro B2C - Prophet",
            style="Header.TLabel",
        ).pack(anchor=tk.W)
        ttk.Label(
            frame_top,
            text="ICL | Ferramenta interna de previsão de faturamento",
            style="SubHeader.TLabel",
        ).pack(anchor=tk.W, pady=(2, 10))

        frame_inputs = ttk.Frame(parent, padding=(0, 10), style="TFrame")
        frame_inputs.pack(fill=tk.X)

        ttk.Label(frame_inputs, text="Mês inicial:").grid(row=0, column=0, sticky=tk.W)
        self.var_mes_ini = tk.StringVar(value=str(self.cfg["forecast"]["mes_alvo"]))
        ttk.Entry(frame_inputs, textvariable=self.var_mes_ini, width=5).grid(
            row=0, column=1, padx=5
        )

        ttk.Label(frame_inputs, text="Mês final:").grid(
            row=0, column=2, sticky=tk.W, padx=(15, 0)
        )
        self.var_mes_fim = tk.StringVar(value=str(self.cfg["forecast"]["mes_alvo"]))
        ttk.Entry(frame_inputs, textvariable=self.var_mes_fim, width=5).grid(
            row=0, column=3, padx=5
        )

        ttk.Label(frame_inputs, text="Ano:").grid(
            row=0, column=4, sticky=tk.W, padx=(15, 0)
        )
        self.var_ano = tk.StringVar(value=str(self.cfg["forecast"]["ano_alvo"]))
        ttk.Entry(frame_inputs, textvariable=self.var_ano, width=6).grid(
            row=0, column=5, padx=5
        )

        self.var_retreinar = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame_inputs,
            text="Retreinar modelos",
            variable=self.var_retreinar,
        ).grid(row=0, column=6, padx=(20, 0))

        frame_buttons = ttk.Frame(parent, padding=(0, 5), style="TFrame")
        frame_buttons.pack(fill=tk.X)

        ttk.Button(
            frame_buttons,
            text="Rodar previsão",
            style="Accent.TButton",
            command=self.on_rodar,
        ).pack(side=tk.LEFT)
        ttk.Button(
            frame_buttons,
            text="Sair",
            style="Secondary.TButton",
            command=self.destroy,
        ).pack(side=tk.RIGHT)

        # barra de progresso + tempo estimado/decorrido
        frame_prog = ttk.Frame(parent, padding=(0, 5), style="TFrame")
        frame_prog.pack(fill=tk.X)

        self.progress = ttk.Progressbar(
            frame_prog, orient="horizontal", mode="determinate", length=400
        )
        self.progress.pack(side=tk.LEFT, padx=(0, 10))

        self.var_time_est = tk.StringVar(value="Estimativa: -")
        ttk.Label(frame_prog, textvariable=self.var_time_est).pack(side=tk.LEFT)

        frame_status = ttk.Frame(parent, padding=(0, 5), style="TFrame")
        frame_status.pack(fill=tk.X)

        ttk.Label(frame_status, text="Status:").pack(side=tk.LEFT)
        self.var_status = tk.StringVar(value="Pronto.")
        self.lbl_status = ttk.Label(frame_status, textvariable=self.var_status)
        self.lbl_status.pack(side=tk.LEFT, padx=5)

        frame_log = ttk.Frame(parent, padding=(0, 5), style="TFrame")
        frame_log.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame_log, text="Log:").pack(anchor=tk.W)
        self.txt_log = tk.Text(
            frame_log,
            height=12,
            width=90,
            bd=0,
            highlightthickness=1,
            highlightbackground="#E5E7EB",
        )
        self.txt_log.pack(fill=tk.BOTH, expand=True)

    def log(self, msg: str):
        self.txt_log.insert(tk.END, msg + "\n")
        self.txt_log.see(tk.END)
        self.update_idletasks()

    def set_status(self, msg: str):
        self.var_status.set(msg)
        self.update_idletasks()

    def set_progress(self, value: float):
        self.progress["value"] = value
        self.update_idletasks()

    def set_time_estimate(self, text: str):
        self.var_time_est.set(text)
        self.update_idletasks()

    def on_rodar(self):
        try:
            mes_ini = int(self.var_mes_ini.get())
            mes_fim = int(self.var_mes_fim.get())
            ano = int(self.var_ano.get())
        except ValueError:
            messagebox.showerror("Erro", "Mês/Ano inválidos.")
            return

        if not (1 <= mes_ini <= 12 and 1 <= mes_fim <= 12) or mes_ini > mes_fim:
            messagebox.showerror("Erro", "Intervalo de meses inválido.")
            return

        retreinar = self.var_retreinar.get()

        try:
            start_total = time.time()
            n_meses = mes_fim - mes_ini + 1
            # 1 passo treino + 1 passo por mês
            self.progress["maximum"] = 1 + n_meses
            self.set_progress(0)
            self.set_time_estimate("Estimativa: -")

            # --- Modelos ---
            self.set_status("Preparando modelos...")
            self.log(
                f"Iniciando previsão de {mes_ini:02d} a {mes_fim:02d}/{ano} | Retreinar={retreinar}"
            )
            if self.last_train_time:
                self.set_time_estimate(
                    f"Estimativa treino: ~{self.last_train_time:.1f} s"
                )

            t0 = time.time()
            modelos = preparar_modelos(self.cfg, retreinar=retreinar)
            self.last_train_time = time.time() - t0
            self.log(f"Treino/carregamento concluído em {self.last_train_time:.1f} s")
            self.set_progress(1)

            qtd_prod_modelos = len(modelos.get("produto", {}))
            self.log(f"Modelos carregados. Produto: {qtd_prod_modelos}")

            if qtd_prod_modelos == 0:
                self.set_status("Nenhum modelo no nível produto.")
                messagebox.showwarning(
                    "Aviso", "Nenhum modelo encontrado para 'produto'."
                )
                return

            # --- Loop de meses ---
            for idx, mes in enumerate(range(mes_ini, mes_fim + 1), start=1):
                self.set_status(f"Prevendo {mes:02d}/{ano}...")
                if self.last_forecast_time and self.last_output_time:
                    est = self.last_forecast_time + self.last_output_time
                    self.set_time_estimate(
                        f"Estimativa por mês: ~{est:.1f} s (previsão+outputs)"
                    )

                # previsão
                t1 = time.time()
                previsoes = gerar_previsoes(self.cfg, modelos, mes, ano)
                self.last_forecast_time = time.time() - t1
                self.log(
                    f"Previsão {mes:02d}/{ano} concluída em {self.last_forecast_time:.1f} s"
                )

                resumo = resumo_previsoes(previsoes)
                self.log(
                    f"[{mes:02d}/{ano}] Receita: R$ {resumo['receita_total']:,.2f} | "
                    f"Volume: {resumo['volume_total']:,.2f} | Produtos: {resumo['qtd_produtos']}"
                )

                # outputs
                t2 = time.time()
                pasta = gerar_arquivos(self.cfg, previsoes, mes, ano)
                self.last_output_time = time.time() - t2
                self.log(
                    f"Outputs {mes:02d}/{ano} gerados em {self.last_output_time:.1f} s -> {pasta}"
                )

                self.set_progress(1 + idx)  # treino (1) + idx-ésimo mês

            total_time = time.time() - start_total
            # garante barra cheia
            self.set_progress(self.progress["maximum"])
            self.set_status("Previsão concluída.")
            self.set_time_estimate(f"Tempo total: {total_time:.1f} s")
            self.log(f"Tempo total da execução: {total_time:.1f} s")

            messagebox.showinfo(
                "Concluído",
                f"Previsão de {mes_ini:02d} a {mes_fim:02d}/{ano} gerada com sucesso.\n"
                f"Tempo total: {total_time:.1f} s\n"
                f"Arquivos em:\n{pasta}",
            )

            # Atualiza a aba de arquivos após gerar
            self.atualizar_lista_arquivos()

        except Exception as e:
            self.set_status("Erro.")
            self.set_time_estimate("Estimativa: -")
            self.log(f"ERRO: {str(e)}")
            messagebox.showerror("Erro", str(e))

    # ----------------- Aba Arquivos -----------------
    def _build_tab_arquivos(self, parent):
        frame_top = ttk.Frame(parent, style="TFrame")
        frame_top.pack(fill=tk.X)

        ttk.Label(
            frame_top,
            text="Arquivos de previsão (Excel)",
            style="Header.TLabel",
        ).pack(side=tk.LEFT)
        ttk.Button(
            frame_top,
            text="Atualizar lista",
            style="Secondary.TButton",
            command=self.atualizar_lista_arquivos,
        ).pack(side=tk.RIGHT)

        frame_tree = ttk.Frame(parent, padding=(0, 10), style="TFrame")
        frame_tree.pack(fill=tk.BOTH, expand=True)

        cols = ("nome", "tamanho_kb")
        self.tree = ttk.Treeview(
            frame_tree, columns=cols, show="headings", height=12
        )
        self.tree.heading("nome", text="Arquivo")
        self.tree.heading("tamanho_kb", text="Tamanho (KB)")
        self.tree.column("nome", width=600, anchor=tk.W)
        self.tree.column("tamanho_kb", width=120, anchor=tk.E)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(
            frame_tree, orient="vertical", command=self.tree.yview
        )
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        frame_bottom = ttk.Frame(parent, style="TFrame")
        frame_bottom.pack(fill=tk.X)

        ttk.Button(
            frame_bottom,
            text="Abrir arquivo selecionado",
            style="Accent.TButton",
            command=self.abrir_arquivo_selecionado,
        ).pack(side=tk.LEFT, pady=(5, 0))

        # Carrega lista inicial
        self.atualizar_lista_arquivos()

    def atualizar_lista_arquivos(self):
        pasta = get_pasta_output(self.cfg)
        print("DEBUG ARQUIVOS - pasta_output:", pasta)
        if not os.path.isdir(pasta):
            print("DEBUG ARQUIVOS - pasta não existe")
            return

        for item in self.tree.get_children():
            self.tree.delete(item)

        for nome in sorted(os.listdir(pasta)):
            if nome.lower().endswith(".xlsx"):
                caminho = os.path.join(pasta, nome)
                tamanho_kb = os.path.getsize(caminho) // 1024
                self.tree.insert("", tk.END, values=(nome, tamanho_kb))

    def abrir_arquivo_selecionado(self):
        pasta = get_pasta_output(self.cfg)
        selecionado = self.tree.focus()
        if not selecionado:
            messagebox.showwarning("Aviso", "Selecione um arquivo na lista.")
            return
        nome, _ = self.tree.item(selecionado, "values")
        caminho = os.path.join(pasta, nome)
        if not os.path.exists(caminho):
            messagebox.showerror("Erro", "Arquivo não encontrado.")
            return

        try:
            os.startfile(caminho)  # Windows
        except AttributeError:
            subprocess.Popen(["xdg-open", caminho])


def main():
    cfg = carregar_config()
    app = App(cfg)
    app.mainloop()


if __name__ == "__main__":
    main()
