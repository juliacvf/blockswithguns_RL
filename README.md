# Blocks With Guns — Edição RL

> **Desafio do Insper Academy 2026.2 — Blocks With Guns RL edition.**
> A recomendação do desafio é construir **um único algoritmo** capaz de
> enfrentar os **cinco algoritmos determinísticos** embutidos (Dijkstra, A*,
> Strafe, Camper e Markov) — não um algoritmo por adversário.

Um jogo determinístico de tiro em arena 1v1 feito em pygame, sobre um mapa
Perlin de 100x100 células, estendido para um ambiente de aprendizado por
reforço (RL): visões superior e poligonal 3D em tela dividida, paredes,
árvores sólidas, lama que reduz a velocidade, power-ups, uma zona de calor
que encolhe, cinco bots determinísticos, ambientes Gymnasium, autojogo
PettingZoo, pastas rígidas de concurso, visualização em pygame e avaliação
sem interface gráfica. O projeto usa apenas pygame na interface; não há
servidor web nem interface de navegador.

## Sumário

- [Sobre o desafio](#sobre-o-desafio)
- [Configuração](#configuração)
- [Como jogar](#como-jogar)
- [Regras do jogo](#regras-do-jogo)
- [O mapa e suas garantias](#o-mapa-e-suas-garantias)
- [Os cinco bots determinísticos](#os-cinco-bots-determinísticos)
- [Início rápido do concurso](#início-rápido-do-concurso)
- [Como criar um algoritmo de concurso](#como-criar-um-algoritmo-de-concurso)
- [Observação completa do concurso](#observação-completa-do-concurso)
- [Algo Test: exemplo de RL treinável](#algo-test-exemplo-de-rl-treinável)
- [Treino contra os cinco bots determinísticos](#treino-contra-os-cinco-bots-determinísticos)
- [Autojogo com PettingZoo](#autojogo-com-pettingzoo)
- [Testando uma implantação de concurso](#testando-uma-implantação-de-concurso)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Referências bibliográficas](#referências-bibliográficas)

## Sobre o desafio

Este repositório é a arena do desafio. Seu objetivo é entregar **uma pasta
de concurso** (veja [Como criar um algoritmo de concurso](#como-criar-um-algoritmo-de-concurso))
com **um único algoritmo** — heurístico, treinado ou híbrido — que jogue
bem contra **qualquer um** dos cinco bots determinísticos, em mapas que
você nunca viu. Para isso o projeto oferece:

- um **motor de jogo determinístico** (`core/`): mesma seed + mesmas ações
  → mesmo resultado, sempre;
- um **contrato de observação completa** (nada é escondido de nenhum
  jogador), pronto para frameworks de redes neurais;
- **ambientes de treino** padrão Gymnasium (1v1 contra os bots) e
  PettingZoo (autojogo simultâneo);
- um **exemplo treinável completo** ([`Algo Test/`](Algo%20Test/)) que usa
  Q-learning tabular e vence o Dijkstra em ~70% dos mapas de avaliação;
- ferramentas de **validação, batalha e visualização** das suas entregas.

## Configuração

É recomendado usar Python 3.13.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

As dependências são apenas quatro: `pygame` (jogo e renderização),
`numpy` (grades e observações), `gymnasium` (API de RL) e `pettingzoo`
(autojogo multiagente). Não há etapa de build.

## Como jogar

```bash
.venv/bin/python main.py
```

O menu do pygame oferece renderização **Poly** (3D poligonal em tela
dividida, estilo Quake) ou **Topdown** (vista superior), e partidas
**Human vs Algo** ou **Algo vs Algo** com os cinco bots determinísticos e
as pastas de concurso. Controles humanos: **WASD** move, **mouse** mira,
**clique ou E** atira.

## Regras do jogo

- Mapa Perlin determinístico de 100x100; **paredes** bloqueiam movimento,
  balas e visão. **Árvores** bloqueiam movimento, mas não balas nem visão.
  **Lama** reduz a velocidade para 0,5x.
- Jogadores começam com **8 vidas**; a partida termina quando alguém zera
  as vidas ou após **120 segundos** (desempate por vidas restantes;
  empate se iguais).
- Balas viajam a **14 blocos/s** e desaparecem depois de **20 blocos**.
- O cooldown normal de tiro é **0,65 s**; o power-up QUICKSHOT o reduz para
  **0,35 s por 5 s**.
- Um **power-up** aparece a cada **1 s** e expira após **25 s**. Tipos:
  `LIFE` (+1 vida, até o máximo), `QUICKSHOT`, `SPEED` (1,5x por 5 s) e
  `SWAP` (troca a posição dos dois jogadores).
- A **zona de calor**: o quadrado seguro central perde 0,25 blocos de
  semilargura por segundo; quem ficar fora dele perde **1 vida a cada 5 s**.
- **Recuo**: uma única vez por partida, o primeiro acerto que deixa um
  jogador com **exatamente 3 vidas** o teleporta para uma célula livre
  dentro da zona segura, a pelo menos **25 blocos** do atirador. Acertos
  com 2 ou 1 vida não ativam o recuo.

## O mapa e suas garantias

O mapa é gerado por ruído Perlin com seed fixa, seguido de três cuidados
importantes para a justiça das partidas:

1. **Spawn carving**: um disco livre de raio 4 é aberto ao redor de cada
   spawn, que ficam nos cantos opostos (15,15) e (84,84).
2. **Conectividade garantida**: os dois spawns são sempre mutualmente
   alcançáveis — se as árvores colocadas depois das paredes selarem a rota,
   as árvores sobre um caminho livre-de-paredes entre os spawns são
   removidas. Ninguém nasce preso.
3. **Sem bolsões sem saída para teleportes**: `random_free_cell` (usado
   pelo recuo e pelo surgimento de power-ups) só escolhe células da
   **região conectada aos spawns** (`map.main_region`), então nenhum
   teleporte abandona um jogador num bolsão selado do qual seria
   impossível sair andando.

Paredes, lama e árvores usam fluxos de ruído/aleatoriedade independentes,
e toda a geração é determinística por seed — essencial para reproduzir
treinos e avaliações.

## Os cinco bots determinísticos

Todos recebem uma `View` com o mapa inteiro (como as IAs do jogo original)
e um `random.Random` com seed própria — são perfeitamente reproduzíveis.

| bot | estilo |
|---|---|
| `dijkstra` | Perseguição cautelosa por rotas secas e baratas; foge quando pressionado e atira com mira deliberadamente tremida. Se perceber que ficou preso (quase sem se mover), fixa um ponto de fuga e sai por pathfinding. |
| `astar` | Atacante equilibrado: perseguição A* com custo moderado de lama, mira com antecipação de velocidade e ataques diretos que aceitam cruzar lama quando vale a distância. |
| `strafe` | Orbita o adversário em média distância com strafe perpendicular constante, desviando de obstáculos e invertendo a órbita por um temporizador com seed. |
| `camper` | Defensivo: segura uma posição seca com linha de tiro limpa, kiteia quando pressionado e se reposiciona em direção ao último local onde viu o inimigo. |
| `markov` | O mais difícil: mantém memória datada das posições e tiros inimigos, mistura predição física com predição por memória e desvia de balas proporcionalmente a um nível de ameaça calculado. |

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

Como argumentos de `player1` tem-se {human, custom, dijkstra, astar, strafe, camper, markov} e para `player2` tem-se {custom, dijkstra, astar, strafe, camper, markov}.

Use `--algo1-folder` e `--algo2-folder` para selecionar outras pastas de concurso. Outras opções úteis do visualizador: `--seed` (troca o mapa), `--fps`, `--exit-on-finish` (fecha ao final da partida); ESC ou Q encerram a qualquer momento.

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

## Observação completa do concurso

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

## Algo Test: exemplo de RL treinável

[`Algo Test/`](Algo%20Test/) é um exemplo pequeno e intencionalmente
legível de aprendizado por reforço para o slot 1 — um ponto de partida
para o seu algoritmo do desafio. A arquitetura tem duas camadas:

1. **Decisão (aprendida)**: uma tabela Q escolhe entre oito táticas —
   perseguir, recuar, strafe à esquerda/direita, buscar power-up, desviar
   de bala, segurar a linha de tiro ou correr para a zona segura.
2. **Execução (determinística)**: auxiliares convertem a tática em
   pathfinding A*, mira com antecipação quantizada nos 16 bins legais e
   uma ação de concurso válida.

### Como ele enxerga o jogo

Antes de decidir, a observação completa é comprimida em dez
características — cada uma vira um peso na tabela Q, inicializado com uma
heurística sensata que o Q-learning pode reforçar ou descartar:

- **distância** (em faixas) e **linha de visão** até o oponente;
- **ameaça de balas** considerando posição *e trajetória* de cada bala
  inimiga: só conta se a reta de voo passa perto de você antes de a bala
  esgotar o alcance de 20 blocos — bala passando longe ou voando para
  longe não assusta;
- **mira dos dois jogadores**: se a sua já está alinhada no alvo (vale
  segurar a linha de tiro) e se a do oponente aponta para você (vale sair
  da frente);
- **cenário**: lama sob os pés e margem até a borda da zona de calor;
- **power-ups**: proximidade e tipo (machucado, prefere `LIFE`; para
  pressionar, `SPEED` e `QUICKSHOT` valem mais);
- **placar**: faixa das suas vidas e quem está na frente.

### Como ele aprende

Treino ε-guloso (exploração começando em 0,65 e decaindo até 0,04) com
Q-learning tabular (γ = 0,99; α decaindo por visita, deliberadamente baixo
para não corromper bons priors com alvos ruidosos). A recompensa vem do
ambiente (+1 por acerto, −1 por dano sofrido, −1 por queimadura da zona,
+0,5 por power-up, ±10 no fim da partida) mais um pequeno shaping de
aproximação. **Cada episódio é uma partida completa**: só termina com
nocaute ou no limite de 120 s do próprio motor — não existe corte de tempo
menor. O treino roda sem interface, na velocidade máxima do simulador
(uma partida inteira leva poucos segundos reais).

### Resultados

O `weights/qtable.npz` incluído foi produzido por `train.py` contra o
Dijkstra — não é um placeholder. Execução de referência: 100 episódios,
seed 456.

- **Treino (com exploração ligada): 80 vitórias, 5 empates e 15 derrotas.**
- **Avaliação em seeds reservadas** (20 partidas por seed base, em mapas
  nunca vistos no treino):

| seed base | vitórias | empates | derrotas |
|----------:|--------:|--------:|---------:|
| 10000 | 18 | 0 | 2 |
| 20000 | 12 | 3 | 5 |
| 30000 | 11 | 1 | 8 |
| 40000 | 11 | 0 | 9 |
| 50000 | 17 | 2 | 1 |
| **total (100 partidas)** | **69** | **6** | **25** |

Ou seja, o exemplo **vence o Dijkstra em ~70% dos mapas** e não perde em
75% deles. Transparência: os priors heurísticos sozinhos (tabela Q sem
treino) também são fortes contra o Dijkstra — fechar essa margem com
aprendizado de verdade (mais episódios, currículo contra os cinco bots,
funções de valor aproximadas) é justamente o espírito do desafio.

> Nota de manutenção: estes números são contra o Dijkstra **já corrigido**.
> Ele ganhou detecção de aprisionamento com rota de fuga por pathfinding,
> e o gerador de mapas não permite mais que árvores selem o caminho entre
> os spawns nem que teleportes deixem alguém num bolsão sem saída. Nas
> medições de diagnóstico, as janelas em que o Dijkstra ficava praticamente
> parado caíram de 466 para 51 em 20 partidas.

### Comandos

Treine do zero contra o bot Dijkstra de referência:

```bash
.venv/bin/python "Algo Test/train.py" \
  --episodes 100 --opponent dijkstra --seed 456
```

Para uma demonstração rápida do pipeline, basta usar menos episódios (cada um continua sendo uma partida completa):

```bash
.venv/bin/python "Algo Test/train.py" --episodes 5
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

## Treino contra os cinco bots determinísticos

`BlocksWithGunsBotTrainingEnv` é um ambiente Gymnasium padrão que usa a mesma observação completa e a ação rígida de concurso. `opponent="all"` alterna de modo determinístico entre Dijkstra, A*, Strafe, Camper e Markov — o caminho natural para treinar **o único algoritmo** do desafio contra todos os adversários.

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

A suíte verifica semântica do motor, os cinco bots, conformidade Gymnasium, cobertura da observação completa, o teste oficial da API paralela do PettingZoo, reprodutibilidade de autojogo por seed, ações rígidas de concurso, carregamento de pastas de exemplo, uma batalha real entre pastas, o ciclo de treinar/salvar/carregar do Algo Test, visualização pygame e assets de áudio — 22 verificações no total.

Para teste de fumaça dos renderizadores:

```bash
.venv/bin/python tools/smoke_render.py
```

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

## Referências bibliográficas

- SUTTON, R. S.; BARTO, A. G. **Reinforcement Learning: An Introduction**. 2. ed. Cambridge: MIT Press, 2018. — base do Q-learning e do aprendizado por diferença temporal usados no exemplo.
- WATKINS, C. J. C. H.; DAYAN, P. **Q-learning**. Machine Learning, v. 8, n. 3-4, p. 279–292, 1992. — o algoritmo original da tabela Q.
- DIJKSTRA, E. W. **A note on two problems in connexion with graphs**. Numerische Mathematik, v. 1, p. 269–271, 1959. — caminho de custo mínimo usado pelo bot homônimo e pelo treinador.
- HART, P. E.; NILSSON, N. J.; RAPHAEL, B. **A formal basis for the heuristic determination of minimum cost paths**. IEEE Transactions on Systems Science and Cybernetics, v. 4, n. 2, p. 100–107, 1968. — o algoritmo A*.
- PERLIN, K. **An image synthesizer**. In: SIGGRAPH '85, p. 287–296, 1985. — o ruído Perlin que gera os mapas.
- TOWERS, M. et al. **Gymnasium: a standard interface for reinforcement learning environments**. arXiv:2407.17032, 2024. — a API de ambiente de RL usada no treino contra os bots.
- TERRY, J. et al. **PettingZoo: Gym for multi-agent reinforcement learning**. Advances in Neural Information Processing Systems (NeurIPS), v. 34, 2021. — a API de autojogo simultâneo.
- VON NEUMANN, J.; MORGENSTERN, O. **Theory of Games and Economic Behavior**. Princeton University Press, 1944. — fundamentos de jogos de soma zero como esta arena 1v1.
