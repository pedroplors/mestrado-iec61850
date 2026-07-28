# -*- coding: utf-8 -*-
"""
Diagnóstico - Rede Elétrica IEC 61850 - VERSÃO CORRIGIDA

Correções aplicadas em relação à versão original (ver relatório no chat):
  E1  - Teto matemático do softmax (tanh + layer_norm na saída) -> cabeça linear
  E2  - Limiar 0.95 inalcançável -> removido, decisão por argmax das log-probs médias
  E3  - Fitness medido no treino durante o treino -> split 3-vias, seleção em validação
  E4  - Batching intra-arquivo (seq=60) != inferência (seq=600) -> batch de ARQUIVOS
  E5  - Buffer FIFO (75) maior que a sequência de treino (60) -> agora coerente
  E6  - CNN em blocos com padding no treino vs arquivo inteiro na inferência -> contexto
  E7  - train_test_split sem stratify -> stratify
  E8  - Pesos [1,30,30,30] arbitrários -> frequência inversa
  E9  - Sem seed nas avaliações -> seed fixa por configuração
  E10 - "falsos alarmes" contabilizados errado -> matriz de confusão 4x4
  E11 - Match de classe por substring com "normal" primeiro -> chave mais longa vence
  E12 - Arquivos descartados/incompletos silenciosamente -> auditoria explícita
  E13 - Conexão residual nunca ativava -> projeção linear quando as dimensões diferem
  E14 - Loss em loop Python por timestep -> vetorizada no bloco
"""

# ==============================================================================
# CÉLULA 1: BIBLIOTECAS, SEED E CONSTANTES
# ==============================================================================
import os
import gc
import glob
import math
import copy
import random
import shutil
import zipfile
import collections

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

SEED = 42


def fixar_seed(seed=SEED):
    """E9: sem isso, a diferença entre duas configurações é ruído de inicialização."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[{'✔' if device == 'cuda' else '!'}] Dispositivo: {device}")

CAMINHO_ZIP = "/content/DATASET-MESTRADO.zip"
PASTA_EXTRAIDA = "/content/dataset_iec61850"

TAMANHO_SAIDA = 4
LOTE_ARQUIVOS = 8      # E4: o lote agora é de ARQUIVOS, não de pedaços de um arquivo
TAMANHO_BLOCO = 100    # E4/E5: TBPTT ao longo do tempo REAL do arquivo

MAPA_CLASSES = {
    "normal": 0,
    "breaker failure": 1,
    "busbar protection": 2,
    "under frequency": 3,
}
NOMES_CLASSES = {0: "Normal", 1: "Breaker Failure", 2: "Busbar Protection", 3: "Under Frequency"}

COLUNAS_FISICAS = [
    "current line 'l1'", "current line 'l2'", "current line 'l3'",
    "voltage phase 'l1-n'", "voltage phase 'l2-n'", "voltage phase 'l3-n'",
    "frequency",
    "circuit breaker open/close status",
    "circuit breaker mechanical failure",
]


# ==============================================================================
# CÉLULA 2: PRÉ-PROCESSAMENTO
# ==============================================================================
def identificar_classe(caminho):
    """
    E11: o dicionário original era percorrido com "normal" primeiro, então qualquer
    caminho contendo "normal" em qualquer lugar virava classe 0 silenciosamente.
    Aqui a chave MAIS ESPECÍFICA (mais longa) vence.
    """
    p = caminho.lower().replace("\\", "/")
    achados = [(len(chave), valor) for chave, valor in MAPA_CLASSES.items() if chave in p]
    if not achados:
        return -1
    return max(achados)[1]


def preparar_dados_reais(caminho_zip, pasta_destino, test_size=0.20, val_size=0.20):
    """
    Retorna listas de (tensor (T, F), rótulo int) — UM rótulo por arquivo.
    O rótulo por instante é criado só na hora da loss, por expansão.
    """
    print("\n--- PRÉ-PROCESSAMENTO ---")

    if os.path.exists(pasta_destino):
        shutil.rmtree(pasta_destino)
    if not os.path.exists(caminho_zip):
        raise FileNotFoundError(f"[!] '{caminho_zip}' não encontrado.")

    with zipfile.ZipFile(caminho_zip, "r") as zip_ref:
        zip_ref.extractall(pasta_destino)

    arquivos_csv = glob.glob(os.path.join(pasta_destino, "**", "*.csv"), recursive=True)
    if not arquivos_csv:
        raise ValueError("Nenhum CSV encontrado.")
    print(f"[✔] {len(arquivos_csv)} arquivos CSV encontrados.")

    # E12: auditoria explícita do que entra e do que é descartado
    auditoria = {"sem_rotulo": [], "muitas_colunas": [], "colunas_faltando": [], "vazios": []}
    dados_brutos, nomes_finais = [], None

    for arquivo in sorted(arquivos_csv):
        rotulo = identificar_classe(arquivo)
        if rotulo == -1:
            auditoria["sem_rotulo"].append(arquivo)
            continue

        df = pd.read_csv(arquivo)
        df.columns = df.columns.str.strip().str.lower()

        if len(df.columns) > 30:
            auditoria["muitas_colunas"].append((os.path.basename(arquivo), NOMES_CLASSES[rotulo]))
            continue
        if len(df) < 2:
            auditoria["vazios"].append(os.path.basename(arquivo))
            continue

        faltantes = [c for c in COLUNAS_FISICAS if c not in df.columns]
        if faltantes:
            # E12: preencher com 0.0 é fabricar sensor. Continua sendo feito,
            # mas agora aparece no relatório para você decidir se descarta.
            auditoria["colunas_faltando"].append(
                (os.path.basename(arquivo), NOMES_CLASSES[rotulo], faltantes)
            )
            for c in faltantes:
                df[c] = 0.0

        df_fis = df[COLUNAS_FISICAS].copy()
        df_fis = df_fis.astype(str).apply(lambda x: x.str.replace(",", ".").str.strip())
        df_num = df_fis.apply(pd.to_numeric, errors="coerce").fillna(0.0)

        df_delta = df_num.diff().fillna(0.0)
        df_delta.columns = [f"{c}_delta" for c in df_num.columns]

        df_final = pd.concat([df_num, df_delta], axis=1)
        if nomes_finais is None:
            nomes_finais = list(df_final.columns)

        dados_brutos.append((df_final.values.astype(np.float32), int(rotulo)))

    if nomes_finais is None:
        raise ValueError("Nenhum arquivo válido sobreviveu ao pré-processamento.")

    # ---- relatório de auditoria -------------------------------------------
    print("\n--- AUDITORIA DE ENTRADA ---")
    print(f"  Aceitos ................. {len(dados_brutos)}")
    print(f"  Sem rótulo no caminho ... {len(auditoria['sem_rotulo'])}")
    print(f"  >30 colunas (descartado)  {len(auditoria['muitas_colunas'])}")
    for nome, classe in auditoria["muitas_colunas"]:
        print(f"      - {nome}  [classe: {classe}]")
    print(f"  Com colunas ausentes .... {len(auditoria['colunas_faltando'])} (preenchidas com 0.0)")
    for nome, classe, cols in auditoria["colunas_faltando"][:10]:
        print(f"      - {nome}  [{classe}]  faltando: {cols}")

    dist = collections.Counter(r for _, r in dados_brutos)
    print("  Distribuição:", {NOMES_CLASSES[k]: v for k, v in sorted(dist.items())})

    # ---- E3/E7: split 3-vias estratificado --------------------------------
    rotulos = [r for _, r in dados_brutos]
    resto, teste = train_test_split(
        dados_brutos, test_size=test_size, random_state=SEED, stratify=rotulos
    )
    rotulos_resto = [r for _, r in resto]
    treino, val = train_test_split(
        resto, test_size=val_size / (1 - test_size), random_state=SEED, stratify=rotulos_resto
    )

    print(f"\n[✔] Treino: {len(treino)} | Validação (seleção): {len(val)} | Teste (intocado): {len(teste)}")
    for nome, conj in [("treino", treino), ("val", val), ("teste", teste)]:
        c = collections.Counter(r for _, r in conj)
        print(f"    {nome:7s}: {[c.get(i, 0) for i in range(TAMANHO_SAIDA)]}")

    # ---- normalização: ajustada SÓ no treino ------------------------------
    scaler = RobustScaler()
    scaler.fit(np.vstack([s for s, _ in treino]))

    def to_tensor(lista):
        return [
            (torch.tensor(scaler.transform(s), dtype=torch.float32, device=device), r)
            for s, r in lista
        ]

    num_features = len(nomes_finais)  # E: não depende mais de variável vazada do loop
    print(f"[✔] Features (base + delta): {num_features}")

    return to_tensor(treino), to_tensor(val), to_tensor(teste), num_features, nomes_finais


# ==============================================================================
# CÉLULA 2b: SANIDADE DOS DADOS  (rode ANTES de culpar o modelo)
# ==============================================================================
def diagnosticar_separabilidade(dados, nomes_colunas, tamanho_saida=TAMANHO_SAIDA):
    """
    Sua loss estacionou em ~1.10 (uniforme = ln(4) = 1.386). Antes de acusar a
    arquitetura, verifique se as features SEPARAM as classes.

    Para cada coluna: estatística F de ANOVA sobre a média por arquivo.
    F ~ 1 significa que a coluna não distingue nada. F alto = sinal real.
    """
    X = np.stack([s.mean(dim=0).cpu().numpy() for s, _ in dados])   # (n_arquivos, n_features)
    y = np.array([r for _, r in dados])

    print("\n--- SEPARABILIDADE POR FEATURE (média por arquivo, ANOVA F) ---")
    linhas = []
    for j, nome in enumerate(nomes_colunas):
        grupos = [X[y == c, j] for c in range(tamanho_saida) if (y == c).sum() > 1]
        if len(grupos) < 2:
            continue
        media_geral = X[:, j].mean()
        ss_entre = sum(len(g) * (g.mean() - media_geral) ** 2 for g in grupos)
        ss_dentro = sum(((g - g.mean()) ** 2).sum() for g in grupos)
        gl_entre = len(grupos) - 1
        gl_dentro = len(X) - len(grupos)
        f_stat = ((ss_entre / gl_entre) / (ss_dentro / gl_dentro)) if ss_dentro > 1e-12 else np.inf
        linhas.append((f_stat, nome, X[:, j].std()))

    for f_stat, nome, desvio in sorted(linhas, reverse=True):
        marca = "  <-- FORTE" if f_stat > 10 else ("" if f_stat > 2 else "  (inútil)")
        print(f"  F={f_stat:8.2f}  std={desvio:8.4f}  {nome}{marca}")

    constantes = [n for f, n, d in linhas if d < 1e-6]
    if constantes:
        print(f"\n[!] Colunas praticamente CONSTANTES no dataset inteiro: {constantes}")
        print("    Elas só adicionam ruído e parâmetros. Considere removê-las.")


# ==============================================================================
# CÉLULA 3: ARQUITETURA (CNN + RNN densamente conectada)
# ==============================================================================
class CamadaVetorizadaBlindada(nn.Module):
    def __init__(self, n_neuronios, total_conexoes, tamanho_memoria):
        super().__init__()
        self.n_neuronios = n_neuronios
        self.tamanho_memoria = tamanho_memoria

        escala_e = 1.0 / math.sqrt(total_conexoes) if total_conexoes > 0 else 1.0
        escala_m = 1.0 / math.sqrt(tamanho_memoria) if tamanho_memoria > 0 else 1.0
        self.pesos = nn.Parameter(torch.randn(total_conexoes, n_neuronios) * escala_e)
        self.pesos_memoria = nn.Parameter(torch.randn(tamanho_memoria, n_neuronios) * escala_m)
        self.bias = nn.Parameter(torch.zeros(n_neuronios))
        self.buffer_memoria = None

    def forward(self, entrada_completa, sinal_anterior, projecao_residual=None):
        soma = torch.matmul(entrada_completa, self.pesos)

        if self.tamanho_memoria > 0 and self.buffer_memoria is not None:
            soma = soma + torch.sum(self.buffer_memoria * self.pesos_memoria.unsqueeze(1), dim=0)

        soma = F.layer_norm(soma, normalized_shape=[self.n_neuronios]) + self.bias
        novo_estado = torch.tanh(soma)

        # E13: no original a residual só somava se as dimensões batessem por acaso.
        # Com arquitetura [32, 64, 32] ela NUNCA ativava (16!=32, 32!=64, 64!=32).
        if sinal_anterior.shape[1] == novo_estado.shape[1]:
            novo_estado = novo_estado + sinal_anterior
        elif projecao_residual is not None:
            novo_estado = novo_estado + projecao_residual(sinal_anterior)

        if self.tamanho_memoria > 0 and self.buffer_memoria is not None:
            self.buffer_memoria = torch.roll(self.buffer_memoria, shifts=-1, dims=0)
            self.buffer_memoria[-1] = novo_estado.detach()

        return novo_estado

    def resetar_memoria(self, batch_size, device):
        if self.tamanho_memoria > 0:
            self.buffer_memoria = torch.zeros(
                self.tamanho_memoria, batch_size, self.n_neuronios, device=device
            )


class RedeConvTemporal(nn.Module):
    def __init__(self, in_channels, tamanho_janela, arquitetura_ocultas, tamanho_saida,
                 tamanho_memoria, canais_conv=16):
        super().__init__()
        if tamanho_janela % 2 == 0:
            tamanho_janela += 1
        self.tamanho_janela = tamanho_janela
        self.canais_conv = canais_conv

        self.conv1 = nn.Conv1d(in_channels, canais_conv,
                               kernel_size=tamanho_janela, padding="same")

        self.ocultas = list(arquitetura_ocultas)
        self.camadas = nn.ModuleList()
        self.projecoes = nn.ModuleList()

        for i, n in enumerate(self.ocultas):
            entradas = canais_conv + sum(self.ocultas[:i]) + sum(self.ocultas)
            self.camadas.append(CamadaVetorizadaBlindada(n, entradas, tamanho_memoria))
            largura_anterior = canais_conv if i == 0 else self.ocultas[i - 1]
            self.projecoes.append(
                nn.Identity() if largura_anterior == n else nn.Linear(largura_anterior, n, bias=False)
            )

        # ==================================================================
        # E1: A CORREÇÃO CENTRAL.
        # No original a última camada era CamadaVetorizadaBlindada -> tanh,
        # logo os logits ficavam em (-1, 1) e o softmax de 4 classes tinha
        # teto de e^1 / (e^1 + 3*e^-1) = 0.711. O filtro `certeza < 0.95`
        # zerava então TODA predição de falha. Cabeça linear = logits livres.
        # ==================================================================
        self.cabeca = nn.Linear(self.ocultas[-1], tamanho_saida)

    def forward_cnn(self, x):
        """x: (B, F, T) -> (B, canais_conv, T)"""
        return torch.relu(self.conv1(x))

    def forward_rnn(self, sinal_t):
        """sinal_t: (B, canais_conv) -> logits (B, tamanho_saida)"""
        if self.estados_t_menos_1:
            historico = torch.cat(self.estados_t_menos_1, dim=1)
        else:
            historico = torch.zeros(sinal_t.shape[0], sum(self.ocultas), device=sinal_t.device)

        fwd_acum = [sinal_t]
        novos_estados = []
        for i, camada in enumerate(self.camadas):
            entrada = torch.cat([torch.cat(fwd_acum, dim=1), historico], dim=1)
            saida = camada(entrada, fwd_acum[-1], self.projecoes[i])
            fwd_acum.append(saida)
            novos_estados.append(saida)

        self.estados_t_menos_1 = novos_estados
        return self.cabeca(novos_estados[-1])   # LOGITS, não probabilidades

    def resetar_estados(self, batch_size, device):
        for camada in self.camadas:
            camada.resetar_memoria(batch_size, device)
        self.estados_t_menos_1 = [
            torch.zeros(batch_size, n, device=device) for n in self.ocultas
        ]


# ==============================================================================
# CÉLULA 4: TREINO E AVALIAÇÃO
# ==============================================================================
def montar_lotes(dados, tamanho_lote, embaralhar=True, seed=None):
    """
    E4: cada elemento do lote é UM ARQUIVO INTEIRO (600 passos), não um pedaço
    de 60. Assim o regime temporal do treino é idêntico ao da inferência, e o
    buffer FIFO de memória (75) enche de verdade durante o treino.
    """
    indices = list(range(len(dados)))
    if embaralhar:
        random.Random(seed).shuffle(indices)

    lotes = []
    for i in range(0, len(indices), tamanho_lote):
        grupo = [dados[j] for j in indices[i:i + tamanho_lote]]
        T = min(s.shape[0] for s, _ in grupo)          # trunca ao menor do lote
        x = torch.stack([s[:T] for s, _ in grupo], dim=0)                    # (B, T, F)
        y = torch.tensor([r for _, r in grupo], dtype=torch.long, device=x.device)
        lotes.append((x, y))
    return lotes


def passar_lote(rede, x, y, criterio, otimizador, tamanho_bloco, treinar):
    """
    Percorre um lote de arquivos com TBPTT em blocos de `tamanho_bloco`.
    Retorna (loss_média, logits (T, B, C) destacados).
    """
    B, T, _ = x.shape
    rede.resetar_estados(B, x.device)

    # E6: no original a CNN via blocos de 60 no treino (com zero-padding dominando
    # um kernel de 21) e o arquivo inteiro na inferência -> features diferentes nas
    # duas fases. Aqui cada bloco é convoluído COM CONTEXTO das bordas e depois
    # recortado, o que reproduz exatamente a convolução sobre a sequência inteira.
    margem = rede.tamanho_janela // 2

    perdas, logits_todos = [], []
    for ini in range(0, T, tamanho_bloco):
        fim = min(ini + tamanho_bloco, T)
        a, b = max(0, ini - margem), min(T, fim + margem)

        feats_ctx = rede.forward_cnn(x[:, a:b, :].permute(0, 2, 1))
        desloc = ini - a
        feats = feats_ctx[:, :, desloc:desloc + (fim - ini)]

        saidas = [rede.forward_rnn(feats[:, :, t]) for t in range(fim - ini)]
        logits = torch.stack(saidas, dim=0)                       # (Tb, B, C)

        # E14: uma única loss vetorizada no bloco, em vez de uma chamada por instante.
        alvo = y.unsqueeze(0).expand(logits.shape[0], -1)
        erro = criterio(logits.reshape(-1, logits.shape[-1]), alvo.reshape(-1))

        if treinar:
            otimizador.zero_grad()
            erro.backward()
            nn.utils.clip_grad_norm_(rede.parameters(), 1.0)
            otimizador.step()

        perdas.append(erro.item())
        logits_todos.append(logits.detach())
        for estado in rede.estados_t_menos_1:
            estado.detach_()

    return float(np.mean(perdas)), torch.cat(logits_todos, dim=0)


def treinar_rede(config, dados_treino, num_features, epocas, pesos_classe,
                 tamanho_saida=TAMANHO_SAIDA, tamanho_bloco=TAMANHO_BLOCO,
                 lote_arquivos=LOTE_ARQUIVOS, silencioso=True, seed=SEED,
                 dados_val=None):
    fixar_seed(seed)
    rede = RedeConvTemporal(
        num_features, config["tamanho_janela"], config["arquitetura"],
        tamanho_saida, config["memoria"]
    ).to(device)

    otimizador = optim.Adam(rede.parameters(), lr=config["lr"])
    criterio = nn.CrossEntropyLoss(
        weight=torch.tensor(pesos_classe, dtype=torch.float32, device=device)
    )

    historico = []
    for epoca in range(epocas):
        rede.train()
        lotes = montar_lotes(dados_treino, lote_arquivos, True, seed + epoca)
        perdas = [passar_lote(rede, x, y, criterio, otimizador, tamanho_bloco, True)[0]
                  for x, y in lotes]
        loss_ep = float(np.mean(perdas))
        historico.append(loss_ep)

        if not silencioso:
            msg = f"      [Época {epoca + 1}/{epocas}] Loss: {loss_ep:.4f}"
            if dados_val is not None and (epoca + 1) % 5 == 0:
                y_v, p_v, _ = prever_arquivos(rede, dados_val, criterio, tamanho_bloco)
                msg += f" | F1-macro val: {np.mean(f1_por_classe(y_v, p_v, tamanho_saida)):.3f}"
            print(msg)

    return rede, criterio, historico


@torch.no_grad()
def prever_arquivos(rede, dados, criterio, tamanho_bloco=TAMANHO_BLOCO, lote=4):
    """
    E2: o veredito por arquivo agora é o argmax da MÉDIA das log-probabilidades
    ao longo do tempo. Sem limiar mágico de 0.95, que era inatingível.
    Retorna (y_true, y_pred, confiança_do_arquivo).
    """
    rede.eval()
    y_true, y_pred, confs = [], [], []

    for x, y in montar_lotes(dados, lote, embaralhar=False):
        _, logits = passar_lote(rede, x, y, criterio, None, tamanho_bloco, False)
        logprob_media = F.log_softmax(logits, dim=-1).mean(dim=0)     # (B, C)
        prob = logprob_media.exp()
        prob = prob / prob.sum(dim=1, keepdim=True)

        y_pred += prob.argmax(dim=1).tolist()
        confs += prob.max(dim=1)[0].tolist()
        y_true += y.tolist()

    rede.train()
    return np.array(y_true), np.array(y_pred), np.array(confs)


def f1_por_classe(y_true, y_pred, n_classes):
    f1 = []
    for c in range(n_classes):
        tp = int(((y_true == c) & (y_pred == c)).sum())
        fp = int(((y_true != c) & (y_pred == c)).sum())
        fn = int(((y_true == c) & (y_pred != c)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    return f1


def calcular_fitness(y_true, y_pred, n_classes=TAMANHO_SAIDA):
    """Mesma ponderação que você usava (20% normal / 80% falhas), mas em VALIDAÇÃO."""
    f1 = f1_por_classe(y_true, y_pred, n_classes)
    return f1[0] * 0.20 + (sum(f1[1:]) / (n_classes - 1)) * 0.80


# ==============================================================================
# CÉLULA 5: BUSCA DE HIPERPARÂMETROS (hill-climbing — não é algoritmo genético)
# ==============================================================================
def gerar_variacao(base, intensidade, janela_max=31):
    """
    E: a versão original só perturbava lr e memoria — arquitetura e pesos nunca
    mudavam, e o log mostrava "Janela [21]" em quase toda geração. Agora a
    arquitetura também muta, e a janela é limitada para não exceder o bloco.
    """
    nova = copy.deepcopy(base)

    nova["lr"] = float(np.clip(nova["lr"] * random.uniform(1 - intensidade, 1 + intensidade),
                               1e-4, 1e-2))
    delta_mem = int(nova["memoria"] * intensidade)
    nova["memoria"] = int(np.clip(nova["memoria"] + random.randint(-delta_mem, delta_mem), 10, 150))

    if random.random() < intensidade:
        nova["tamanho_janela"] = int(np.clip(
            nova["tamanho_janela"] + random.choice([-6, -4, -2, 2, 4, 6]), 3, janela_max))
        if nova["tamanho_janela"] % 2 == 0:
            nova["tamanho_janela"] += 1

    if random.random() < intensidade:
        arq = list(nova["arquitetura"])
        i = random.randrange(len(arq))
        arq[i] = int(np.clip(arq[i] * random.choice([0.5, 0.75, 1.5, 2.0]), 8, 128))
        nova["arquitetura"] = arq

    return nova


def buscar_configuracao(config_inicial, dados_treino, dados_val, num_features, pesos_classe,
                        iteracoes=3, variacoes=3, epocas_busca=8):
    print("\n" + "=" * 62)
    print(" BUSCA DE HIPERPARÂMETROS (fitness medido em VALIDAÇÃO)")
    print("=" * 62)

    campea = copy.deepcopy(config_inicial)
    rede, crit, _ = treinar_rede(campea, dados_treino, num_features, epocas_busca, pesos_classe)
    y_v, p_v, _ = prever_arquivos(rede, dados_val, crit)
    melhor = calcular_fitness(y_v, p_v)
    print(f"-> Aptidão inicial (val): {melhor * 100:.2f}%\n")
    del rede
    if device == "cuda":
        torch.cuda.empty_cache()

    for it in range(iteracoes):
        intensidade = max(0.10, 0.40 - it * 0.05)
        print(f"--- Geração {it + 1} (mutação {intensidade * 100:.0f}%) ---")
        for _ in range(variacoes):
            cand = gerar_variacao(campea, intensidade)
            rede, crit, _ = treinar_rede(cand, dados_treino, num_features,
                                         epocas_busca, pesos_classe)
            y_v, p_v, _ = prever_arquivos(rede, dados_val, crit)
            fit = calcular_fitness(y_v, p_v)
            print(f"  janela={cand['tamanho_janela']:3d} mem={cand['memoria']:3d} "
                  f"lr={cand['lr']:.5f} arq={cand['arquitetura']} -> fit {fit * 100:.2f}%")

            if fit > melhor:
                melhor, campea = fit, copy.deepcopy(cand)
                print("  ⭐ nova campeã")

            del rede
            if device == "cuda":
                torch.cuda.empty_cache()

    print(f"\n[✔] Melhor aptidão em validação: {melhor * 100:.2f}%")
    print(f"    Config: {campea}")
    return campea, melhor


# ==============================================================================
# CÉLULA 6: RELATÓRIO FINAL
# ==============================================================================
def relatorio_final(rede, criterio, dados_teste, titulo="TESTE"):
    """E10: matriz de confusão 4x4 no lugar da contagem confusa de 'falsos alarmes'."""
    y_true, y_pred, confs = prever_arquivos(rede, dados_teste, criterio)

    print("\n" + "=" * 62)
    print(f" RELATÓRIO — CONJUNTO DE {titulo} ({len(y_true)} arquivos)")
    print("=" * 62)

    mc = confusion_matrix(y_true, y_pred, labels=list(range(TAMANHO_SAIDA)))
    cabecalho = "real \\ previsto".ljust(20) + "".join(
        NOMES_CLASSES[c][:12].rjust(14) for c in range(TAMANHO_SAIDA))
    print("\n" + cabecalho)
    for c in range(TAMANHO_SAIDA):
        linha = NOMES_CLASSES[c][:18].ljust(20)
        linha += "".join(str(mc[c, j]).rjust(14) for j in range(TAMANHO_SAIDA))
        print(linha)

    print("\n" + classification_report(
        y_true, y_pred,
        labels=list(range(TAMANHO_SAIDA)),
        target_names=[NOMES_CLASSES[c] for c in range(TAMANHO_SAIDA)],
        zero_division=0, digits=3))

    # E1/E2: prova de que o teto do softmax sumiu. No modelo antigo isso jamais
    # passaria de 0.711 — motivo pelo qual o filtro de 0.95 zerava tudo.
    print(f"Confiança por arquivo: min={confs.min():.3f} "
          f"mediana={np.median(confs):.3f} max={confs.max():.3f}")
    if confs.max() < 0.75:
        print("[!] Confiança ainda com teto baixo — verifique se a cabeça linear está ativa.")

    print(f"\nFitness (20/80, mesma fórmula da tese): {calcular_fitness(y_true, y_pred) * 100:.2f}%")
    return y_true, y_pred, confs


@torch.no_grad()
def plotar_reacao(rede, dados, nomes_colunas, indice=0):
    """
    O gráfico original prometia 'reação milissegundo a milissegundo', mas o
    rótulo é CONSTANTE por arquivo — não existe gabarito temporal para comparar.
    Aqui o painel da direita mostra a EVOLUÇÃO DA CRENÇA do modelo, que é a
    afirmação que os dados de fato sustentam.
    """
    rede.eval()
    sinal, rotulo = dados[indice]

    rede.resetar_estados(1, device)
    x = sinal.unsqueeze(0)                                   # (1, T, F)
    feats = rede.forward_cnn(x.permute(0, 2, 1))
    logits = torch.stack([rede.forward_rnn(feats[:, :, t]) for t in range(x.shape[1])], dim=0)
    probs = F.softmax(logits.squeeze(1), dim=-1).cpu().numpy()   # (T, C)

    idx = next((i for i, c in enumerate(nomes_colunas)
                if "voltage" in c and "l1" in c and "delta" not in c), 0)
    serie = sinal[:, idx].cpu().numpy()

    cores = {0: "green", 1: "orange", 2: "purple", 3: "red"}
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))

    axes[0].plot(serie, color="black", lw=0.6)
    axes[0].axvspan(0, len(serie), color=cores[rotulo], alpha=0.25)
    axes[0].set_title(f"Sinal — rótulo do arquivo: {NOMES_CLASSES[rotulo]} (constante)")
    axes[0].set_xlabel("amostra")

    for c in range(TAMANHO_SAIDA):
        axes[1].plot(probs[:, c], color=cores[c], lw=1.2, label=NOMES_CLASSES[c])
    axes[1].axhline(0.711, ls="--", lw=0.8, color="gray")
    axes[1].text(len(serie) * 0.55, 0.72, "teto do modelo ANTIGO (0.711)", fontsize=8, color="gray")
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title("Probabilidade atribuída pelo modelo ao longo do tempo")
    axes[1].set_xlabel("amostra")

    legendas = [mpatches.Patch(color=cores[k], alpha=0.5, label=NOMES_CLASSES[k])
                for k in cores]
    fig.legend(handles=legendas, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.06), fontsize=11)
    plt.tight_layout()
    plt.show()
    rede.train()


# ==============================================================================
# EXECUÇÃO
# ==============================================================================
if __name__ == "__main__":
    fixar_seed()
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    dados_treino, dados_val, dados_teste, NUM_FEATURES, NOMES_COLUNAS = \
        preparar_dados_reais(CAMINHO_ZIP, PASTA_EXTRAIDA)

    # Rode isto antes de qualquer treino.
    diagnosticar_separabilidade(dados_treino, NOMES_COLUNAS)

    # E8: pesos por frequência inversa. O original usava [1, 30, 30, 30] com um
    # dataset 36/18/18/18 — um exagero de 15x que empurrava para sobre-alarme,
    # exatamente o oposto do que o limiar de 0.95 fazia. Os dois se anulavam.
    cont = collections.Counter(r for _, r in dados_treino)
    pesos_classe = [
        len(dados_treino) / (TAMANHO_SAIDA * max(cont.get(c, 0), 1))
        for c in range(TAMANHO_SAIDA)
    ]
    print(f"\nPesos de classe (frequência inversa): "
          f"{[round(p, 2) for p in pesos_classe]}")

    config_inicial = {
        "arquitetura": [32, 64, 32],
        "memoria": 60,
        "lr": 0.001,
        "tamanho_janela": 21,
    }

    config_campea, _ = buscar_configuracao(
        config_inicial, dados_treino, dados_val, NUM_FEATURES, pesos_classe,
        iteracoes=3, variacoes=3, epocas_busca=8,
    )

    print("\n" + "=" * 62)
    print(" TREINO FINAL (treino + validação, 40 épocas)")
    print("=" * 62)
    rede_final, criterio_final, hist = treinar_rede(
        config_campea, dados_treino + dados_val, NUM_FEATURES,
        epocas=40, pesos_classe=pesos_classe, silencioso=False, dados_val=dados_val,
    )

    relatorio_final(rede_final, criterio_final, dados_teste, "TESTE")
    plotar_reacao(rede_final, dados_teste, NOMES_COLUNAS, indice=0)
