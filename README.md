# Blocks With Guns — Edição RL

Um jogo determinístico de tiro em arena 1v1 com pygame, em um mapa Perlin de 100x100. Inclui visões superior e poligonal em tela dividida, paredes, árvores sólidas, lama que reduz a velocidade, power-ups, uma zona de calor que encolhe, cinco bots integrados, ambientes Gymnasium, autojogo PettingZoo, pastas rígidas para concurso, visualização pygame e avaliação sem interface. O projeto usa apenas pygame; não há servidor web nem interface de navegador.

## Configuração

É recomendado usar Python 3.13.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Execute o jogo normal:

```bash
.venv/bin/python main.py
```

O menu do pygame oferece renderização Poly ou Topdown e partidas Human vs Algo ou Algo vs Algo com os cinco bots determinísticos.

### Regras atuais do jogo

- Mapa Perlin determinístico de 100x100; paredes bloqueiam movimento, balas e visão.
- Árvores bloqueiam movimento, mas não balas ou visão. Lama reduz a velocidade para 0,5x.
- Jogadores começam com 8 vidas; partidas terminam com 0 vidas ou após 120 segundos.
- Balas viajam a 14 blocos/s e desaparecem depois de 20 blocos.
- O cooldown normal de tiro é 0,65 s. QUICKSHOT o altera para 0,35 s por 5 s.
- Um power-up aparece a cada 1 s e expira após 25 s. Os tipos são LIFE, QUICKSHOT, SPEED (1,5x por 5 s) e SWAP.
- O quadrado seguro central perde 0,25 blocos de semilargura por segundo. Jogadores fora dele perdem uma vida a cada 5 s.
- O primeiro acerto que deixa um jogador exatamente com 3 vidas ativa o teleporte único de recuo, ao menos 25 blocos distante do atirador.

## Início rápido do concurso

As duas entradas de exemplo incluídas mostram a estrutura obrigatória:

```text
algo1/
  algo1.py
  weights/
    README.md
algo2/
  algo2.py
  weights/
    README.md
```

Valide as duas pastas sem executar uma partida:

```bash
.venv/bin/python -m rl.contest --validate-only
```

Execute uma batalha de três episódios entre pastas e salve o primeiro episódio como replay:

```bash
.venv/bin/python -m rl.contest \
  --algo1-folder algo1 --algo2-folder algo2 \
  --episodes 3 --base-seed 100
```

O executor alterna os lados de spawn entre os episódios. Seeds, pastas e ações idênticas produzem resultados idênticos no motor.

### Batalhas visuais e Humano vs Algo personalizado

```bash
# algo1/algo1.py contra algo2/algo2.py
.venv/bin/python -m rl.visualize --player1 custom --player2 custom

# algo1 personalizado contra qualquer bot determinístico integrado
.venv/bin/python -m rl.visualize --player1 custom --player2 markov

# jogador humano 1 contra algo2 personalizado
.venv/bin/python -m rl.visualize --player1 human --player2 custom
```

Use `--algo1-folder` e `--algo2-folder` para selecionar outras pastas de concurso extraídas. Os controles humanos são WASD, mira com o mouse e clique ou E para atirar.

## Algo Test: treine e jogue um exemplo de RL

[`Algo Test/`](Algo%20Test/) é um exemplo pequeno e intencionalmente simples de aprendizado por reforço para o slot 1. Sua tabela Q escolhe entre oito táticas legíveis: perseguir, recuar, strafing à esquerda/direita, buscar power-up, desviar de uma bala, alinhar uma linha de tiro ou retornar ao centro seguro. Auxiliares determinísticos convertem essa escolha de alto nível em pathfinding, mira de 16 bins e uma ação de concurso válida.

O `weights/qtable.npz` incluído foi produzido por `train.py` contra Dijkstra; não é um marcador de posição escrito manualmente. Uma execução de referência de 100 episódios, com episódios de 45 segundos e seed 456, produziu uma vitória no treino no episódio 60. Nas dez seeds reservadas usadas pelo treinador, a política salva obteve 9 empates e 1 derrota, contra 5 empates e 5 derrotas para o prior tático sem treinamento. É uma base defensiva compreensível, não um agente de última geração.

Treine do zero contra o bot Dijkstra de referência:

```bash
.venv/bin/python "Algo Test/train.py" \
  --episodes 100 --opponent dijkstra --max-seconds 45 --seed 456
```

Para uma demonstração rápida do pipeline, use menos episódios e episódios mais curtos:

```bash
.venv/bin/python "Algo Test/train.py" --episodes 5 --max-seconds 10
```

`train.py` salva `Algo Test/weights/qtable.npz`, avalia a política salva e imprime retornos dos episódios e vencedores. Adicione `--resume` para continuar a partir do checkpoint existente.

Valide e assista à política treinada lutar contra Dijkstra no pygame:

```bash
.venv/bin/python -m rl.contest \
  --algo1-folder "Algo Test" --algo2-folder algo2 --validate-only

.venv/bin/python -m rl.visualize \
  --player1 custom --algo1-folder "Algo Test" --player2 dijkstra
```

Teste a mesma política salva contra todos os bots determinísticos:

```bash
.venv/bin/python "Algo Test/train.py" \
  --evaluate-only --opponent all --eval-episodes 10
```

Ou visualize qualquer confronto individual substituindo `dijkstra` por `astar`, `strafe`, `camper` ou `markov`. O treino também pode ter outro bot como alvo com `--opponent BOT_NAME`, ou alternar entre todos os cinco com `--opponent all`.

## Como criar um algoritmo de concurso

Um envio para o slot 1 deve conter `algo1.py` e um diretório `weights/`. Um envio para o slot 2 deve conter `algo2.py` e um diretório `weights/`. O diretório de pesos pode estar vazio para um algoritmo determinístico. Ele pode conter `.npz`, checkpoints de modelo ou quaisquer outros arquivos de dados exigidos pela política.

O script Python deve expor `Agent` ou `make_agent`:

```python
from __future__ import annotations

import os
import numpy as np


class Agent:
    def __init__(self, weights_path=None):
        # Em uma pasta de concurso, weights_path é o diretório weights/ absoluto.
        model_file = os.path.join(weights_path, "policy.npz")
        self.model = np.load(model_file) if os.path.exists(model_file) else None

    def reset(self):
        # Chamado antes de cada episódio. Redefina aqui o estado recorrente.
        pass

    def act(self, obs):
        # Retorne exatamente três inteiros: [move, aim, shoot].
        return [4, 0, 0]
```

Uma factory também é válida:

```python
def make_agent(weights_path):
    return Agent(weights_path)
```

A ação é `MultiDiscrete([9, 16, 2])`:

| índice | significado |
|---|---|
| `move` | `0..8`, uma grade de direção 3x3 gerada por `for y in (-1,0,1), for x in (-1,0,1)`; `4` permanece parado |
| `aim` | `0..15`, bins de ângulo absoluto cobrindo `[0, 2π)`; o bin `0` aponta para a direita |
| `shoot` | `0` não atira, `1` atira |

A validação do concurso é rígida. A ação deve ter exatamente três valores inteiros finitos dentro desses limites. Uma saída inválida encerra a partida com um erro que identifica algoritmo, episódio e tick.

Políticas determinísticas escritas manualmente são permitidas. Veja `algo1/algo1.py` e `algo2/algo2.py` para exemplos completos.

### Observação completa do concurso

Os dois algoritmos recebem o estado inteiro. Nada é ocultado de nenhum jogador. Os valores são arrays `numpy.float32` com formatos fixos, portanto o contrato pode ser usado diretamente por frameworks de redes neurais.

| chave | formato | conteúdo |
|---|---:|---|
| `map` | `(12, 100, 100)` | todas as posições da grade e ocupação de entidades |
| `self` | `(15,)` | estado completo do jogador que está agindo |
| `opponent` | `(15,)` | estado completo do outro jogador |
| `bullets` | `(32, 7)` | todas as balas ativas, preenchido com zeros |
| `bullet_mask` | `(32,)` | `1` para linhas de bala válidas |
| `powerups` | `(32, 7)` | todos os power-ups ativos, preenchido com zeros |
| `powerup_mask` | `(32,)` | `1` para linhas de power-up válidas |
| `game` | `(18,)` | tempo, zona segura e constantes normalizadas do jogo |

`map` usa a ordem `[channel, x, y]`. Seus canais são:

```text
0 paredes               6 power-ups LIFE
1 árvores               7 power-ups QUICKSHOT
2 lama                  8 power-ups SPEED
3 células seguras       9 power-ups SWAP
4 posição própria      10 balas próprias
5 posição oponente     11 balas do oponente
```

`self` e `opponent` usam a mesma ordem de atributos:

```text
x, y, sin(aim), cos(aim), lives, cooldown, quickshot timer,
speed timer, heat timer, retreated, last-hit recency,
last-shot recency, in heat, current speed, line of sight
```

As linhas de bala contêm:

```text
x, y, vx, vy, owned by self, remaining ttl, traveled distance
```

As linhas de power-up contêm:

```text
x, y, remaining ttl, LIFE, QUICKSHOT, SPEED, SWAP
```

O vetor `game` contém tempo decorrido/restante normalizado, tick, semilargura da zona segura, timestep fixo, velocidades de jogador e bala, alcance da bala, os dois cooldowns de tiro, cadência e expiração de power-ups, regras de calor, regras de recuo, redução por lama e multiplicador do power-up de velocidade. A maioria dos valores contínuos é normalizada para `[-1, 1]`; máscaras, canais do mapa e constantes do jogo usam `[0, 1]`. Os nomes canônicos e o codificador estão em `rl/full_observation.py`.

## Treino contra os cinco bots determinísticos

`BlocksWithGunsBotTrainingEnv` é um ambiente Gymnasium padrão que usa a mesma observação completa e a ação rígida de concurso. `opponent="all"` alterna de modo determinístico entre Dijkstra, A*, Strafe, Camper e Markov.

```python
from rl.bot_training_env import BlocksWithGunsBotTrainingEnv

env = BlocksWithGunsBotTrainingEnv(opponent="all")
obs, info = env.reset(seed=0)

done = False
while not done:
    action = my_policy.act(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

Use `render_mode="human"` para uma visualização de treino em Python/pygame ao vivo ou `render_mode="rgb_array"` para obter frames. Um único nome de bot pode substituir `"all"` para treino por currículo.

Recompensas por jogador:

- `+1` acerto causado e `-1` acerto sofrido
- `-1` dano da zona de calor
- `+0.5` coleta de power-up
- `+10/-10` vitória/derrota na partida
- `-0.005` a cada passo

O `BlocksWithGunsEnv` compacto original permanece disponível em `rl/env.py` para o exemplo tabular existente:

```bash
.venv/bin/python -m rl.train_example --episodes 300 --opponent astar
```

## Autojogo com PettingZoo

O jogo é simultâneo, portanto a API multiagente nativa é o `ParallelEnv` do PettingZoo. As duas políticas recebem sua própria perspectiva do estado completo e enviam ações para o mesmo tick do motor a 30 Hz.

```python
from rl.pettingzoo_env import parallel_env

env = parallel_env()
observations, infos = env.reset(seed=42)

while env.agents:
    actions = {
        "player_0": policy_0.act(observations["player_0"]),
        "player_1": policy_1.act(observations["player_1"]),
    }
    observations, rewards, terminations, truncations, infos = env.step(actions)

env.close()
```

Este é o ponto de entrada de autojogo para bibliotecas que se integram ao PettingZoo. Mantenha um conjunto de checkpoints antigos e atribua um a cada jogador periodicamente para evitar treinar apenas contra a versão mais nova de si mesmo.

Frameworks que exigem a interface AEC por turnos podem usar:

```python
from rl.pettingzoo_env import env
aec_env = env()
```

O ambiente paralelo também oferece `render_mode="human"` e `render_mode="rgb_array"`.

## Testando uma implantação de concurso

Execute a suíte completa depois de alterar o motor, app pygame, contrato RL, bots, executor do concurso ou algoritmo de exemplo:

```bash
.venv/bin/python tools/verify.py
```

A suíte verifica semântica do motor, os cinco bots, conformidade Gymnasium, cobertura da observação completa, o teste oficial da API paralela do PettingZoo, reprodutibilidade de autojogo por seed, ações rígidas de concurso, carregamento de pastas de exemplo, uma batalha real entre pastas, o ciclo de treinar/salvar/carregar do Algo Test, visualização pygame e assets de áudio.

Para teste de fumaça dos renderizadores:

```bash
.venv/bin/python tools/smoke_render.py
```

## Segurança e hospedagem do concurso

Scripts de concurso são Python comum e **não são isolados em sandbox**. As verificações de pasta impõem a estrutura esperada, rejeitam links simbólicos, limitam as entradas a 1.000 arquivos e 512 MB, e validam ações, mas não tornam Python hostil seguro. Execute submissões não confiáveis em contêineres ou máquinas virtuais isoladas, com limites de CPU, memória, tempo de parede, sistema de arquivos, processos e rede.

## Estrutura do projeto

```text
main.py                 ponto de entrada do jogo pygame
core/                   motor determinístico do jogo e geração de mapa
bots/                   cinco bots determinísticos integrados
game/                   app pygame e renderizadores
rl/env.py               ambiente Gymnasium compacto original
rl/full_observation.py  contrato de observação completa do concurso
rl/bot_training_env.py  treino com estado completo contra bots integrados
rl/pettingzoo_env.py    autojogo paralelo e conversão AEC
rl/contest.py           validação de pastas e batalhas sem interface
rl/visualize.py         partidas visuais customizadas/integradas/humanas
algo1/, algo2/          entradas de concurso de exemplo e pastas de pesos
Algo Test/              exemplo de tabela Q treinada, train.py e pesos
tools/verify.py         suíte completa de verificação
```
