# Generalista

Perfil `generalista_v1`, selecionado com `--profile generalista`.

A política estende o Markov e conserva os seus nove comportamentos. A nova
ação `control_range` acrescenta o controlador radial do A* para adversários
observados se aproximando, entre 8 e 22 blocos. O controlador busca distância
de 14 blocos, mantendo deslocamento lateral. Ele não substitui o desvio de
projéteis nem o retorno obrigatório à zona segura.

Não consulta a identidade do adversário. O estado mantém a representação do
Markov, incluindo distância, movimento radial, projéteis, vidas, calor e
histórico de movimento. A política aprende quando escolher cada ação.

## Pesos

- `weights/generalista/initial.npz`: inicialização com os nove valores de ação
  do Markov copiados exatamente, mais o prior da nova ação.
- `weights/generalista/trained_100.npz`: candidato após 100 partidas de treino.
- `weights/generalista/qtable.npz`: candidato escolhido em mapas de validação.

Os checkpoints originais dos especialistas e o modelo padrão anterior são
preservados. O perfil novo deve ser selecionado explicitamente.

## Avaliar

```powershell
.venv\Scripts\python.exe MeuAlgo\train.py --profile generalista --evaluate-only --opponent all --eval-episodes 25 --eval-sides both --eval-seed 810000
```

## Jogar

```powershell
.venv\Scripts\python.exe -m rl.visualize --player1 custom --algo1-script MeuAlgo/algo1.py --algo1-weights MeuAlgo/weights/generalista/qtable.npz --player2 astar
```

## Treinar um novo candidato

```powershell
.venv\Scripts\python.exe MeuAlgo\train.py --profile generalista --opponent all --episodes 100 --seed 610000 --epsilon 0.12 --alpha 0.01 --init-from MeuAlgo/weights/especialistas/markov.npz --output MeuAlgo/weights/generalista/novo_candidato.npz
```

O treino do generalista alterna os lados em pares do mesmo mapa. Com `all`,
blocos de dez partidas distribuem os cinco bots igualmente e usam ambos os
lados para cada bot. Use múltiplos de dez para manter essa distribuição.
`--resume` continua um checkpoint do mesmo perfil. `--init-from` só aceita
um checkpoint Markov em um treino novo e não combina tabelas incompatíveis.

## Verificação

```powershell
.venv\Scripts\python.exe tools\test_generalista.py
```

Foram separados os mapas de treino (610000–610049), validação
(710000–710009) e teste (810000–810024). Os dois candidatos disputam
100 partidas de validação cada. A escolha considera, nessa ordem, mais
vitórias, menos derrotas, menos eventos de calor e menos dano de projéteis.
O escolhido é comparado com Markov e Dijkstra em 250 partidas de teste por
modelo, usando os mesmos mapas, lados e adversários, sem aprendizado.
Resultados estão em `weights/generalista/evaluation.json`.

## Resultado desta execução

Candidato selecionado: `trained`.

| Modelo | Vitórias | Empates | Derrotas | Partidas |
|---|---:|---:|---:|---:|
| generalista | 250 | 0 | 0 | 250 |
| markov | 250 | 0 | 0 | 250 |
| dijkstra | 250 | 0 | 0 | 250 |

Esses resultados medem os cinco bots existentes nos mapas reservados.
Não demonstram, por si só, desempenho contra adversários novos.


## Revisão de defesa — 05/11/2026

A revisão fica dentro de `GeneralistAgent`; os especialistas mantêm suas
implementações e os checkpoints não são alterados. O formato continua
`generalista_v1`, compatível com os pesos existentes.

- `_incoming_flights()` acompanha TTL, alcance e paredes na ordem do motor,
  preservando o primeiro subpasso válido quando o segundo já expira.
- `_bullet_threat()` considera a colisão com o corpo e o movimento do jogador,
  sem exigir que o projétil alcance o centro do personagem.
- `_future_positions()` e `_movement_risk()` estimam trajetórias legais com
  paredes, árvores, lama, expiração de velocidade e risco de calor.
- `_defensive_move()` prioriza evitar colisões e favorece movimento lateral
  sob fogo observado, com escolha de lado persistente. O retorno urgente à
  zona segura continua prioritário. A mira é recalculada depois do movimento
  pela rotina existente, permitindo atirar enquanto se desloca.

O ajuste não depende do nome do bot adversário. Não houve novo treinamento
nem alteração específica da estratégia para sair de impasses.

Teste de geometria e integração com o motor:

```powershell
.venv\Scripts\python.exe tools\test_generalista_defense.py
```

Nos dois casos problemáticos do Camper, a nova versão venceu por 8–0, sem
receber tiros. Na comparação de 500 partidas por versão, os acertos recebidos
caíram de 161 para 45. A versão anterior fez 499 vitórias e 1 empate; a nova
fez 498 vitórias e 2 empates. Nenhuma perdeu partidas nessa comparação.
O dano por calor passou de 5 para 8 eventos no conjunto de regressão; nos
mapas novos, ambas tiveram zero eventos de calor. Resultados completos em
`weights/generalista/defense_evaluation.json`.
