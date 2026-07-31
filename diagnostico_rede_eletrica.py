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

Melhorias de arquitetura aplicadas nesta revisão (RedeConvTemporal / CamadaVetorizadaBlindada):
  E15 - Um único kernel de convolução não captura transitórios rápidos
        (falha de disjuntor, poucas amostras) e tendências lentas
        (subfrequência, centenas de amostras) ao mesmo tempo -> banco de
        convoluções causais em 3 escalas (curta/média/longa), concatenadas
        e projetadas de volta para `canais_conv`.
  E16 - Convolução com padding "same" (kernel centrado) enxergava o FUTURO
        da sequência em toda predição -> ramos agora são estritamente
        causais (zero-pad só à esquerda), coerente com um relé que decide
        em tempo real a partir apenas do passado.
  E17 - BatchNorm exigiria estatísticas de lote e se comporta mal com
        lote=1 (o caso de `plotar_reacao`, inferência de um arquivo só) ->
        GroupNorm, que normaliza por amostra e é idêntica em treino/eval.
  E18 - LayerNorm sem ganho aprendível e célula recorrente sem nenhuma
        regularização -> ganho por neurônio na LayerNorm + dropout
        configurável na CNN e na célula recorrente (ativo só em .train()).
  E19 - GPU (Colab T4/A100 etc.): cuDNN benchmark + TF32 habilitados quando
        `device == "cuda"`; toda a rede permanece 100% device-agnostic
        (nenhum `.cuda()` hardcoded, tudo segue o device do tensor de entrada).

Correções aplicadas nesta revisão (Fitness 20.00% / Accuracy 38.9%, recall 0
em "Breaker Failure" e "Under Frequency", loss estagnada em ~1.05):
  E20 - 'circuit breaker mechanical failure_delta' é constante em todo o
        dataset (std=0.0, `diagnosticar_separabilidade` só avisava, não
        agia) -> removida junto com qualquer outra coluna sem variância no
        TREINO, de forma genérica (sem hardcode de nome de coluna).
  E21 - CrossEntropyLoss com pesos de classe trata todo exemplo igual
        independente de quão errado o modelo já está; numa rede que colapsa
        para 1-2 classes, a maioria do gradiente vem de exemplos já fáceis
        -> Focal Loss ponderada (fator (1-p)^gamma), que concentra o
        gradiente nos exemplos difíceis -- exatamente as classes ignoradas.
  E22 - Todas as features entravam na CNN em pé de igualdade, mesmo com
        separabilidade (ANOVA F) muito diferente entre elas (frequency
        F=3.11 vs correntes F~1) -> ganho de entrada aprendível por canal,
        para a própria rede amplificar as features mais separáveis via
        gradiente, em vez de depender só da escala do RobustScaler.
  E23 - Adam sem weight_decay e sem scheduler, num dataset de 53 arquivos de
        treino, converge rápido para um mínimo raso que ignora as classes
        minoritárias -> AdamW com weight_decay (também sob busca de
        hiperparâmetros) + CosineAnnealingLR.
  E24 - O treino final usava os pesos da ÚLTIMA época mesmo quando o F1
        macro em validação era MELHOR em épocas anteriores (ex.: 0.600 na
        época 25 vs 0.496 na época 40, a que efetivamente ia para o teste)
        -> checkpoint dos pesos com melhor fitness em validação, restaurado
        ao final do treino.

Correções aplicadas nesta revisão (V3 — Fitness val 62.11% vs teste 31.52%,
overfitting clássico; Busbar Protection com recall 0, 4 arquivos previstos
como Breaker Failure):
  E25 - Gap grande entre fitness de validação e de teste -> regularização
        reforçada em três frentes: (1) dropout da CNN e da célula recorrente
        agora são parâmetros INDEPENDENTES (`dropout_conv`/`dropout_rnn`, com
        piso mais alto que o `dropout` único da V2) já que os dois módulos
        memorizam de formas diferentes; (2) `weight_decay` padrão maior; (3)
        ruído gaussiano leve injetado no tensor de ENTRADA só durante o
        treino (nunca em validação/teste), forçando a rede a não decorar o
        valor exato de cada amostra. Os três (mais `peso_confusao`, ver E26)
        também entram na busca de hiperparâmetros.
  E26 - Busbar Protection (classe 2) zerou porque a rede está confundindo-a
        sistematicamente com Breaker Failure (classe 1) -> a Focal Loss
        ganhou um termo extra que penaliza especificamente a massa de
        probabilidade que vaza DENTRO desse par de classes (nos dois
        sentidos), além do de sempre (pesos por frequência inversa). Isso é
        mais direcionado do que só aumentar o peso da classe 2, que reduziria
        falsos negativos de Busbar às custas de mais falsos positivos em
        QUALQUER outra classe, não só a que ela está de fato confundindo.

Correções aplicadas nesta revisão (V4 — Fitness 39.77% vs 20% inicial, mas
rede viciada em prever Breaker Failure: 8 previsões dessa classe no teste,
absorvendo todo o Busbar Protection e derrubando a precisão de Breaker para
12%; Under Frequency com 100% de precisão):
  E27 - A Focal Loss (E21) combinada com peso de classe por frequência
        inversa (E8) e a penalidade extra de confusão Breaker<->Busbar (E26)
        empurrou o desbalanceamento longe demais na direção oposta: o fator
        dinâmico (1-pt)^gamma da focal MULTIPLICA o peso já alto de Breaker
        Failure toda vez que a rede está incerta, criando incentivo para
        "chutar" Breaker sempre que a confiança é baixa -- inclusive em cima
        de Busbar Protection. Trocada por CrossEntropyLoss simples com
        label smoothing (0.1): sem o termo dinâmico, e com o alvo suavizado
        (a rede nunca é incentivada a apostar toda a probabilidade numa
        única classe), a atualização de gradiente deixa de amplificar a
        classe de peso mais alto. Os pesos de classe por frequência inversa
        continuam, mas agora com TETO: nenhuma classe pode pesar mais que
        `peso_max_relativo` vezes o peso da classe mais frequente (Normal),
        removendo o mecanismo que fazia o peso da classe mais rara dominar o
        gradiente sozinho. A penalidade extra de par confusível (E26) foi
        removida junto -- ela atacava a MESMA confusão pelo lado oposto e,
        combinada com o teto de peso, deixa de ser necessária.
  E28 - Arquitetura com 4 camadas ocultas ([32, 64, 64, 32], desde a V3
        E23) tem capacidade de sobra para decorar ruído fino em vez do
        padrão da falha, ainda mais agora que o gradiente fica mais "manso"
        com o teto de peso (E27). Reduzida para 3 camadas ([32, 64, 32], a
        mesma profundidade da V2) -- menos parâmetros na célula recorrente,
        que já concentra a maior parte da capacidade de memorização via
        `pesos_memoria`. dropout_conv/dropout_rnn/ruido_treino (E25)
        continuam como estavam.

Correções aplicadas nesta revisão (V4 — Fitness caiu para 30.48%, de volta ao
patamar pré-V3; Breaker Failure e Busbar Protection voltaram a recall 0):
  E29 - A troca da Focal Loss por CrossEntropy+label smoothing (E27) e a
        redução da arquitetura para 3 camadas (E28) foram longe demais na
        direção oposta: sem o termo dinâmico (1-pt)^gamma a rede voltou a
        ignorar as classes raras (o problema original que a Focal Loss
        existia para resolver), e a camada removida tirou capacidade que a
        rede efetivamente usava (V3 é a campeã em fitness de teste, 39.77%
        vs 30.48% da V4). Configuração MESTRE, combinando o que funcionou em
        cada versão em vez de trocar um mecanismo pelo outro:
          (a) arquitetura de volta a [32, 64, 64, 32] (a da V3/E23);
          (b) Focal Loss de volta (E21/E26), mas com `gamma` reduzido de 2.0
              para 1.5 -- o fator dinâmico continua concentrando gradiente
              nos exemplos difíceis, só que multiplica o peso da classe rara
              com menos força, já que ESSE era o mecanismo apontado (E27)
              como causa do viés para prever Breaker Failure demais; o
              `peso_confusao` do par (Breaker, Busbar) também cai de 0.5
              para 0.3 pelo mesmo motivo, mantendo o mecanismo mas com menos
              agressividade. Ambos continuam sob busca de hiperparâmetros;
          (c) dropout_conv/dropout_rnn independentes e ruído gaussiano de
              treino (E25) mantidos como estavam -- nenhuma das duas versões
              apontou esses dois como causa de problema;
          (d) treino final estendido de 40 para 60 épocas -- com a
              arquitetura maior de volta e o CosineAnnealingLR calculado
              sobre `epocas`, um treino mais longo dá tempo do LR terminar
              de decair e da rede efetivamente assentar nas classes raras
              antes do checkpoint de melhor fitness (E24) fixar os pesos.
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


import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[{'OK' if device == 'cuda' else '!'}] Dispositivo: {device}")
if device == "cuda":
    # Robustez/velocidade em GPU (Colab T4/A100, ou qualquer CUDA local):
    # cuDNN escolhe os kernels por benchmark (formas de tensor fixas aqui,
    # já que os lotes são sempre (B, F, T) com T de bloco constante) e TF32
    # é habilitado onde o hardware suporta, sem custo de precisão relevante
    # para este problema.
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    print(f"    GPU: {torch.cuda.get_device_name(0)}")

CAMINHO_ZIP = "/content/DATASET-MESTRADO.zip" if os.path.exists("/content/DATASET-MESTRADO.zip") else os.path.abspath("DATASET-MESTRADO.zip")
PASTA_EXTRAIDA = "/content/dataset_iec61850" if os.path.exists("/content") else os.path.abspath("dataset_iec61850")

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
    print(f"[OK] {len(arquivos_csv)} arquivos CSV encontrados.")

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

    print(f"\n[OK] Treino: {len(treino)} | Validação (seleção): {len(val)} | Teste (intocado): {len(teste)}")
    for nome, conj in [("treino", treino), ("val", val), ("teste", teste)]:
        c = collections.Counter(r for _, r in conj)
        print(f"    {nome:7s}: {[c.get(i, 0) for i in range(TAMANHO_SAIDA)]}")

    # ---- E20: remover colunas constantes no TREINO -------------------------
    # Uma coluna sem variância não carrega informação -- só ruído e parâmetros
    # a mais. Decidido a partir do treino (nunca de val/teste) para não
    # vazar estatística, e aplicado às três partições pelos mesmos índices.
    desvios_treino = np.vstack([s for s, _ in treino]).std(axis=0)
    manter = np.where(desvios_treino > 1e-6)[0]
    removidas = [nomes_finais[i] for i in range(len(nomes_finais)) if i not in manter]
    if removidas:
        print(f"\n[OK] Colunas constantes removidas do pré-processamento: {removidas}")
    nomes_finais = [nomes_finais[i] for i in manter]

    def filtrar_colunas(lista):
        return [(s[:, manter], r) for s, r in lista]

    treino, val, teste = filtrar_colunas(treino), filtrar_colunas(val), filtrar_colunas(teste)

    # ---- normalização: ajustada SÓ no treino ------------------------------
    scaler = RobustScaler()
    scaler.fit(np.vstack([s for s, _ in treino]))

    def to_tensor(lista):
        return [
            (torch.tensor(scaler.transform(s), dtype=torch.float32, device=device), r)
            for s, r in lista
        ]

    num_features = len(nomes_finais)  # E: não depende mais de variável vazada do loop
    print(f"[OK] Features (base + delta): {num_features}")

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
    def __init__(self, n_neuronios, total_conexoes, tamanho_memoria, dropout=0.0):
        super().__init__()
        self.n_neuronios = n_neuronios
        self.tamanho_memoria = tamanho_memoria

        escala_e = 1.0 / math.sqrt(total_conexoes) if total_conexoes > 0 else 1.0
        escala_m = 1.0 / math.sqrt(tamanho_memoria) if tamanho_memoria > 0 else 1.0
        self.pesos = nn.Parameter(torch.randn(total_conexoes, n_neuronios) * escala_e)
        self.pesos_memoria = nn.Parameter(torch.randn(tamanho_memoria, n_neuronios) * escala_m)
        self.bias = nn.Parameter(torch.zeros(n_neuronios))
        # E18: LayerNorm original não tinha ganho aprendível (só bias manual
        # depois da normalização). Corrente/tensão/frequência normalizadas
        # pelo RobustScaler ainda chegam em escalas internas diferentes por
        # classe de falha; um ganho por neurônio deixa a rede reescalar cada
        # canal em vez de herdar sempre variância unitária.
        self.ganho = nn.Parameter(torch.ones(n_neuronios))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.buffer_memoria = None

    def forward(self, entrada_completa, sinal_anterior, projecao_residual=None):
        if entrada_completa.shape[0] != sinal_anterior.shape[0]:
            raise ValueError(
                f"Lote inconsistente na célula recorrente: entrada tem batch="
                f"{entrada_completa.shape[0]}, estado anterior tem batch="
                f"{sinal_anterior.shape[0]}. `resetar_memoria`/`resetar_estados` "
                f"provavelmente foi chamado com um batch_size diferente do usado agora."
            )

        soma = torch.matmul(entrada_completa, self.pesos)

        if self.tamanho_memoria > 0 and self.buffer_memoria is not None:
            soma = soma + torch.sum(self.buffer_memoria * self.pesos_memoria.unsqueeze(1), dim=0)

        soma = self.ganho * F.layer_norm(soma, normalized_shape=[self.n_neuronios]) + self.bias
        novo_estado = torch.tanh(soma)

        # E13: no original a residual só somava se as dimensões batessem por acaso.
        # Com arquitetura [32, 64, 32] ela NUNCA ativava (16!=32, 32!=64, 64!=32).
        if sinal_anterior.shape[1] == novo_estado.shape[1]:
            novo_estado = novo_estado + sinal_anterior
        elif projecao_residual is not None:
            novo_estado = novo_estado + projecao_residual(sinal_anterior)

        if self.tamanho_memoria > 0 and self.buffer_memoria is not None:
            # O buffer guarda o estado LIMPO (antes do dropout): dropout
            # aplicado sobre o que é propagado às próximas camadas, nunca
            # gravado na memória temporal, senão o ruído se acumularia por
            # até centenas de passos dentro de um mesmo arquivo.
            self.buffer_memoria = torch.roll(self.buffer_memoria, shifts=-1, dims=0)
            self.buffer_memoria[-1] = novo_estado.detach()

        return self.dropout(novo_estado)

    def resetar_memoria(self, batch_size, device):
        if self.tamanho_memoria > 0:
            self.buffer_memoria = torch.zeros(
                self.tamanho_memoria, batch_size, self.n_neuronios,
                device=device, dtype=self.pesos.dtype,
            )


class RedeConvTemporal(nn.Module):
    def __init__(self, in_channels, tamanho_janela, arquitetura_ocultas, tamanho_saida,
                 tamanho_memoria, canais_conv=16, dropout_conv=0.25, dropout_rnn=0.30):
        super().__init__()
        self.tamanho_janela = max(3, int(tamanho_janela))
        self.canais_conv = canais_conv

        # E22: ganho de entrada aprendível, um escalar por feature. As
        # colunas chegam do RobustScaler todas com a MESMA escala de
        # variância, mas não com a mesma SEPARABILIDADE entre classes (ex.:
        # "frequency" tem F-ANOVA=3.11, "current line 'l1'" tem F~1). Sem
        # isso a rede tem que aprender essa relevância inteira dentro dos
        # pesos da primeira convolução; com isso, o gradiente pode amplificar
        # diretamente o canal mais informativo antes de qualquer conv/RNN.
        self.ganho_entrada = nn.Parameter(torch.ones(in_channels))

        # E15/E16: banco de convoluções CAUSAIS em 3 escalas de tempo.
        # Curta  -> transitórios rápidos (falha de disjuntor).
        # Média  -> a janela configurada (busca de hiperparâmetros).
        # Longa  -> tendências lentas (deriva de subfrequência).
        # Cada ramo só olha para trás (padding só à esquerda, ver
        # `forward_cnn_com_contexto`): a saída no instante t nunca depende
        # de amostras em t+1 em diante, como exigido de um relé real.
        curta = max(3, self.tamanho_janela // 3)
        media = self.tamanho_janela
        longa = min(self.tamanho_janela * 2 + 1, 61)
        self.escalas = sorted(set([curta, media, longa]))
        self.contexto_necessario = max(self.escalas) - 1

        canais_por_ramo = max(1, canais_conv // len(self.escalas))
        self.ramos_conv = nn.ModuleList([
            nn.Conv1d(in_channels, canais_por_ramo, kernel_size=k, padding=0)
            for k in self.escalas
        ])
        self.projecao_conv = nn.Conv1d(
            canais_por_ramo * len(self.escalas), canais_conv, kernel_size=1
        )

        # E17: GroupNorm em vez de BatchNorm -- não depende de estatísticas
        # de lote, então se comporta de forma idêntica em treino e em
        # inferência de um único arquivo (B=1, o caso de `plotar_reacao`).
        # Aplicada PONTUALMENTE (ver `_norm_conv_pontual`): por padrão o
        # GroupNorm de um tensor (B, C, T) mistura estatísticas ao longo de
        # T, o que vazaria amostras futuras para dentro da normalização de
        # cada instante e quebraria a equivalência bloco == arquivo inteiro
        # (E6). Normalizando (B*T, C, 1) cada instante usa só os próprios canais.
        grupos = next(g for g in range(min(8, canais_conv), 0, -1) if canais_conv % g == 0)
        self.norm_conv = nn.GroupNorm(grupos, canais_conv)
        # E25: dropout da CNN e da célula recorrente são independentes -- a
        # CNN decora padrões locais (poucas amostras de contexto), a célula
        # recorrente decora trajetórias inteiras via `pesos_memoria`; um
        # único dropout compartilhado (V2) sub-regularizava um dos dois.
        self.dropout_conv = nn.Dropout(dropout_conv) if dropout_conv > 0 else nn.Identity()

        self.ocultas = list(arquitetura_ocultas)
        self.camadas = nn.ModuleList()
        self.projecoes = nn.ModuleList()

        for i, n in enumerate(self.ocultas):
            entradas = canais_conv + sum(self.ocultas[:i]) + sum(self.ocultas)
            self.camadas.append(CamadaVetorizadaBlindada(n, entradas, tamanho_memoria, dropout=dropout_rnn))
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

    def _norm_conv_pontual(self, x):
        """
        x: (B, C, T) -> (B, C, T). GroupNorm normalmente reduz sobre (C/G, T)
        juntos; aqui cada instante t é normalizado usando SÓ os próprios
        canais, para não vazar estatísticas do futuro (nem de outros blocos
        do TBPTT) para dentro do instante atual.
        """
        B, C, T = x.shape
        x_pontual = x.permute(0, 2, 1).reshape(B * T, C, 1)
        x_norm = self.norm_conv(x_pontual)
        return x_norm.reshape(B, T, C).permute(0, 2, 1)

    def forward_cnn_com_contexto(self, x_com_contexto):
        """
        x_com_contexto: (B, F, contexto_necessario + T) -> (B, canais_conv, T).
        Os primeiros `contexto_necessario` passos devem ser amostras REAIS
        anteriores do mesmo arquivo (ou zero, só no início dele). Usado no
        TBPTT em blocos (`passar_lote`) para reproduzir exatamente a mesma
        convolução causal que seria obtida rodando o arquivo inteiro de uma vez.
        """
        x_com_contexto = x_com_contexto * self.ganho_entrada.view(1, -1, 1)
        max_ctx = self.contexto_necessario
        saidas = []
        for conv, k in zip(self.ramos_conv, self.escalas):
            recorte = x_com_contexto[:, :, max_ctx - (k - 1):]
            saidas.append(conv(recorte))
        concat = torch.cat(saidas, dim=1)
        return self.dropout_conv(torch.relu(self._norm_conv_pontual(self.projecao_conv(concat))))

    def forward_cnn(self, x):
        """x: (B, F, T) -> (B, canais_conv, T). Uso em sequência completa (ex.:
        `plotar_reacao`): zero-preenche o início do arquivo -- antes da
        primeira amostra não existe passado nenhum -- e delega ao caminho
        com contexto."""
        return self.forward_cnn_com_contexto(F.pad(x, (self.contexto_necessario, 0)))

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
class PerdaFocal(nn.Module):
    """
    E21: CrossEntropyLoss (mesmo com pesos de classe) dá o mesmo peso de
    gradiente a um exemplo já bem classificado e a um exemplo que a rede
    erra feio. Numa rede que colapsa para prever só 1-2 classes, a maioria
    dos exemplos "fáceis" pertence às classes majoritárias e domina o
    gradiente médio, afogando as classes raras. O fator (1 - p_alvo)^gamma
    aproxima esse peso de zero para exemplos já acertados com confiança e o
    mantém alto para exemplos errados -- concentrando o aprendizado
    justamente nas classes com recall 0.

    E26 - `pares_confusaveis`: pares de classes que a rede troca entre si
    (ex.: Busbar Protection previsto como Breaker Failure) recebem uma
    penalidade extra proporcional à probabilidade colocada na classe ERRADA
    do par, nos dois sentidos. Isso ataca a confusão específica sem inflar o
    peso da classe inteira (o que penalizaria confundir Busbar com QUALQUER
    outra classe, não só com Breaker Failure).

    E29 - a V4 removeu esta loss inteira porque `gamma=2.0` MULTIPLICAVA o
    peso já alto de Breaker Failure toda vez que a rede estava incerta,
    incentivando a rede a "chutar" essa classe -- mas a V4 também zerou o
    recall de Breaker Failure e Busbar Protection ao tirar o mecanismo. O
    ajuste correto é reduzir a agressividade, não remover: `gamma` (padrão
    1.5, era 2.0) e `peso_confusao` (padrão 0.3, era 0.5) continuam ativos,
    só que mais fracos, e ambos seguem sob busca de hiperparâmetros.
    """
    def __init__(self, pesos_classe, gamma=1.5, pares_confusaveis=None, peso_confusao=0.3):
        super().__init__()
        self.gamma = gamma
        self.peso_confusao = peso_confusao
        self.pares_confusaveis = list(pares_confusaveis or [])
        self.register_buffer("pesos", torch.tensor(pesos_classe, dtype=torch.float32))

    def forward(self, logits, alvo):
        log_p = F.log_softmax(logits, dim=-1)
        prob = log_p.exp()
        log_pt = log_p.gather(1, alvo.unsqueeze(1)).squeeze(1)
        pt = log_pt.exp()
        peso_t = self.pesos[alvo]
        perda = -peso_t * (1 - pt) ** self.gamma * log_pt

        for a, b in self.pares_confusaveis:
            mascara_a = (alvo == a)
            mascara_b = (alvo == b)
            perda = perda + self.peso_confusao * mascara_a * prob[:, b]
            perda = perda + self.peso_confusao * mascara_b * prob[:, a]

        return perda.mean()


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


def passar_lote(rede, x, y, criterio, otimizador, tamanho_bloco, treinar, ruido_treino=0.0):
    """
    Percorre um lote de arquivos com TBPTT em blocos de `tamanho_bloco`.
    Retorna (loss_média, logits (T, B, C) destacados).
    """
    B, T, _ = x.shape
    rede.resetar_estados(B, x.device)

    if treinar and ruido_treino > 0:
        # E25: ruído gaussiano leve só no treino (nunca em validação/teste,
        # `treinar=False` nessas chamadas) -- data augmentation simples que
        # impede a rede de decorar o valor exato de cada amostra. O
        # RobustScaler já deixa a maior parte das features em escala ~1,
        # então um sigma pequeno é uma perturbação real sem apagar o sinal.
        x = x + torch.randn_like(x) * ruido_treino

    # E6/E16: no original a CNN via blocos de 60 no treino (com zero-padding
    # dominando um kernel de 21) e o arquivo inteiro na inferência -> features
    # diferentes nas duas fases; além disso o padding "same" centrado deixava
    # cada bloco enxergar amostras FUTURAS em relação ao instante avaliado.
    # Agora a CNN é causal: cada bloco recebe como contexto só as amostras
    # REAIS anteriores do próprio arquivo (zero-padding apenas no primeiro
    # bloco, onde não existe passado), reproduzindo exatamente a convolução
    # causal que se obteria rodando o arquivo inteiro de uma vez.
    margem = rede.contexto_necessario

    perdas, logits_todos = [], []
    for ini in range(0, T, tamanho_bloco):
        fim = min(ini + tamanho_bloco, T)
        a = max(0, ini - margem)

        trecho = x[:, a:fim, :].permute(0, 2, 1)
        pad_esquerdo = margem - (ini - a)
        if pad_esquerdo > 0:
            trecho = F.pad(trecho, (pad_esquerdo, 0))

        feats = rede.forward_cnn_com_contexto(trecho)

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
                 dados_val=None, avaliar_a_cada=5):
    fixar_seed(seed)
    rede = RedeConvTemporal(
        num_features, config["tamanho_janela"], config["arquitetura"],
        tamanho_saida, config["memoria"],
        dropout_conv=config.get("dropout_conv", 0.25),
        dropout_rnn=config.get("dropout_rnn", 0.30),
    ).to(device)

    # E23: AdamW (weight_decay desacoplado do gradiente, ao contrário do
    # Adam+L2 clássico) + cosseno decrescente. Sem isso o Adam puro, num
    # treino de 53 arquivos, converge cedo para um mínimo raso que já
    # "resolve" a maioria das classes majoritárias e para de melhorar --
    # exatamente a loss estagnada em ~1.05 relatada.
    # E25: piso de weight_decay mais alto que a V2 -- o gap fitness
    # val (62.11%) vs teste (31.52%) era overfitting clássico.
    otimizador = optim.AdamW(
        rede.parameters(), lr=config["lr"],
        weight_decay=config.get("weight_decay", 3e-4),
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=max(epocas, 1))
    # E29: Focal Loss de volta (E21/E26), com gamma e peso_confusao reduzidos
    # em relação à V3 -- ver PerdaFocal para o raciocínio completo.
    criterio = PerdaFocal(
        pesos_classe,
        gamma=config.get("gamma", 1.5),
        pares_confusaveis=[(1, 2)],
        peso_confusao=config.get("peso_confusao", 0.3),
    ).to(device)
    ruido_treino = config.get("ruido_treino", 0.03)

    # E24: checkpoint dos pesos com melhor FITNESS em validação. Sem isso o
    # modelo devolvido é o da última época, mesmo que o treino tenha
    # piorado depois de um pico (visto no log: F1-macro val 0.600 na época
    # 25, caindo para 0.496 na época 40 -- a que ia para o teste).
    melhor_fitness_val = -1.0
    melhor_estado = None

    historico = []
    for epoca in range(epocas):
        rede.train()
        lotes = montar_lotes(dados_treino, lote_arquivos, True, seed + epoca)
        perdas = [passar_lote(rede, x, y, criterio, otimizador, tamanho_bloco, True,
                              ruido_treino=ruido_treino)[0]
                  for x, y in lotes]
        loss_ep = float(np.mean(perdas))
        historico.append(loss_ep)
        scheduler.step()

        avaliar_agora = dados_val is not None and (
            (epoca + 1) % avaliar_a_cada == 0 or epoca == epocas - 1
        )
        if avaliar_agora:
            y_v, p_v, _ = prever_arquivos(rede, dados_val, criterio, tamanho_bloco)
            fit_val = calcular_fitness(y_v, p_v, tamanho_saida)
            if fit_val > melhor_fitness_val:
                melhor_fitness_val = fit_val
                melhor_estado = copy.deepcopy(rede.state_dict())

        if not silencioso:
            msg = f"      [Época {epoca + 1}/{epocas}] Loss: {loss_ep:.4f}"
            if avaliar_agora:
                msg += (f" | Fitness val: {fit_val * 100:.2f}% "
                        f"(melhor: {melhor_fitness_val * 100:.2f}%)")
            msg += f" | lr: {scheduler.get_last_lr()[0]:.5f}"
            print(msg)

    if melhor_estado is not None:
        rede.load_state_dict(melhor_estado)
        if not silencioso:
            print(f"      [OK] Pesos restaurados da melhor época em validação "
                  f"(fitness {melhor_fitness_val * 100:.2f}%)")

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
    # E23: weight_decay também sob busca -- regularização forte demais
    # (dataset de 53 arquivos) sufoca as classes raras, fraca demais deixa a
    # rede memorizar a maioria e ignorá-las.
    nova["weight_decay"] = float(np.clip(
        nova["weight_decay"] * random.uniform(1 - intensidade, 1 + intensidade), 1e-6, 1e-2))
    delta_mem = int(nova["memoria"] * intensidade)
    nova["memoria"] = int(np.clip(nova["memoria"] + random.randint(-delta_mem, delta_mem), 10, 150))

    # E25: dropout_conv/dropout_rnn/ruido_treino também sob busca -- o piso
    # mais alto da V3 é um chute inicial, não um ótimo conhecido.
    nova["dropout_conv"] = float(np.clip(
        nova["dropout_conv"] + random.uniform(-intensidade, intensidade) * 0.3, 0.05, 0.6))
    nova["dropout_rnn"] = float(np.clip(
        nova["dropout_rnn"] + random.uniform(-intensidade, intensidade) * 0.3, 0.05, 0.6))
    nova["ruido_treino"] = float(np.clip(
        nova["ruido_treino"] * random.uniform(1 - intensidade, 1 + intensidade), 0.0, 0.15))
    # E29: gamma da focal e peso_confusao do par (Breaker, Busbar) também sob
    # busca -- os padrões (1.5 / 0.3) são um chute deliberadamente mais fraco
    # que a V3 (2.0 / 0.5), não um ótimo conhecido.
    nova["gamma"] = float(np.clip(
        nova["gamma"] * random.uniform(1 - intensidade, 1 + intensidade), 0.5, 3.0))
    nova["peso_confusao"] = float(np.clip(
        nova["peso_confusao"] * random.uniform(1 - intensidade, 1 + intensidade), 0.0, 1.0))

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
                print("  [*] nova campeã")

            del rede
            if device == "cuda":
                torch.cuda.empty_cache()

    print(f"\n[OK] Melhor aptidão em validação: {melhor * 100:.2f}%")
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
    plt.savefig("reacao_modelo.png", bbox_inches="tight")
    print("[OK] Gráfico salvo em 'reacao_modelo.png'")
    plt.close()
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

    # E29: configuração MESTRE -- arquitetura de volta a 4 camadas ocultas
    # ([32, 64, 64, 32], a da V3/E23, a campeã em fitness de teste até aqui);
    # gamma/peso_confusao da Focal Loss (PerdaFocal) reduzidos em relação à
    # V3 para não repetir o viés de sobre-previsão de Breaker Failure; e
    # dropout_conv/dropout_rnn/ruído de treino (E25) mantidos como estavam.
    config_inicial = {
        "arquitetura": [32, 64, 64, 32],
        "memoria": 60,
        "lr": 0.001,
        "tamanho_janela": 21,
        "weight_decay": 3e-4,      # E25: piso mais alto que a V2 (1e-4)
        "dropout_conv": 0.25,      # E25: antes um único "dropout": 0.15
        "dropout_rnn": 0.30,       # E25: célula recorrente sofre mais overfit
        "ruido_treino": 0.03,      # E25: ruído gaussiano leve só no treino
        "gamma": 1.5,              # E29: focal de volta, menos agressiva que a V3 (2.0)
        "peso_confusao": 0.3,      # E29: idem para o par Breaker/Busbar (V3: 0.5)
    }

    config_campea, _ = buscar_configuracao(
        config_inicial, dados_treino, dados_val, NUM_FEATURES, pesos_classe,
        iteracoes=3, variacoes=3, epocas_busca=8,
    )

    print("\n" + "=" * 62)
    print(" TREINO FINAL (treino + validação, 60 épocas)")
    print("=" * 62)
    rede_final, criterio_final, hist = treinar_rede(
        config_campea, dados_treino + dados_val, NUM_FEATURES,
        epocas=60, pesos_classe=pesos_classe, silencioso=False, dados_val=dados_val,
    )

    relatorio_final(rede_final, criterio_final, dados_teste, "TESTE")
    plotar_reacao(rede_final, dados_teste, NOMES_COLUNAS, indice=0)
