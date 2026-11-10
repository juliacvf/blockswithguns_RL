# Generalista e especialistas

Esta pasta contém o código de desenvolvimento do generalista, o treino e os cinco especialistas preservados. Para enviar o algoritmo ao concurso, use as pastas prontas em [Concurso](../Concurso/README.md).

## Sumário

- [Arquivos principais](#arquivos-principais)
- [Assistir a uma partida](#assistir-a-uma-partida)
- [Avaliar o generalista](#avaliar-o-generalista)
- [Treinar um candidato](#treinar-um-candidato)
- [Verificar o código](#verificar-o-código)
- [Estado atual](#estado-atual)

## Arquivos principais

| Caminho | Finalidade |
|---|---|
| [algo1.py](algo1.py) | Políticas, carregamento dos perfis, navegação, mira e defesa. |
| [train.py](train.py) | Treino e avaliação sem interface gráfica. |
| [weights/generalista/qtable.npz](weights/generalista/qtable.npz) | Checkpoint oficial do generalista, perfil `generalista_v1`. |
| [weights/especialistas/](weights/especialistas/) | Um checkpoint oficial para cada especialista. |
| [weights/README.md](weights/README.md) | Descrição dos pesos oficiais e da entrada legada. |

O generalista herda as táticas do Markov, a navegação e a segurança do Dijkstra/Strafe e o controle de distância do A*. A revisão de defesa considera a colisão com o corpo, os subpassos finais dos projéteis e o movimento lateral sob fogo. A mira é recalculada após o deslocamento, permitindo atirar em movimento quando o disparo é válido. A política não consulta o nome do bot adversário.

Todos os comandos abaixo partem da pasta principal do projeto e usam PowerShell.

## Assistir a uma partida

```powershell
.\.venv\Scripts\python.exe -m rl.visualize --player1 custom --algo1-script MeuAlgo/algo1.py --algo1-weights MeuAlgo/weights/generalista/qtable.npz --player2 camper --seed 1020100 --bot2-seed 1020101
```

Troque `camper` por `dijkstra`, `astar`, `strafe` ou `markov`. `--bot2-seed` fixa a aleatoriedade do adversário para reproduzir uma avaliação. Ao terminar, Enter fecha a janela.

Para assistir a um especialista, substitua o caminho dos pesos por `MeuAlgo/weights/especialistas/NOME.npz`.

## Avaliar o generalista

```powershell
.\.venv\Scripts\python.exe MeuAlgo/train.py --profile generalista --evaluate-only --opponent all --eval-episodes 25 --eval-sides both --eval-seed 1120000
```

Esse comando avalia 25 mapas nos dois lados: 50 partidas por bot e 250 no total. Os pesos permanecem congelados. Use `--opponent camper`, por exemplo, para avaliar somente um adversário.

## Treinar um candidato

Salve experimentos em outro destino para preservar o checkpoint oficial:

```powershell
.\.venv\Scripts\python.exe MeuAlgo/train.py --profile generalista --opponent all --episodes 100 --seed 123 --init-from MeuAlgo/weights/especialistas/markov.npz --output "$env:TEMP\generalista_candidato.npz"
```

O treino alterna os lados em pares do mesmo mapa. Com `all`, cada bloco de dez partidas distribui os cinco bots igualmente. `--init-from` inicia um novo generalista a partir do checkpoint Markov, preservando seus nove valores de ação por estado e adicionando o controle de distância. Não mistura tabelas de especialistas com representações diferentes.

Para continuar um candidato, use `--resume`, o mesmo `--output` e o mesmo `--seed`, sem `--init-from`. Avalie o candidato antes de adotá-lo como oficial. Os padrões do generalista são epsilon inicial 0,12 e alpha inicial 0,01.

## Verificar o código

```powershell
.\.venv\Scripts\python.exe tools/test_generalista.py
.\.venv\Scripts\python.exe tools/test_generalista_defense.py
```

São nove testes de perfis, transferência e navegação e onze testes de colisão, esquiva, tiro em movimento e retorno seguro.

## Estado atual

O checkpoint oficial recebeu 100 partidas de treino equilibradas entre os cinco bots. A correção de defesa alterou o código, mantendo esses pesos.

A proteção de calor também confere movimentos sem projéteis próximos: prevê o fechamento da zona durante os próximos 0,8 s e evita sair da área segura. Rotas de retorno planejadas mantêm os desvios necessários para contornar paredes. Dois cenários que falhavam antes agora passam, incluindo um em que a versão anterior perdia uma vida no próximo tick. A mesma correção está nos dois scripts de concurso.

Na comparação de 9 de novembro de 2026, antes e depois mantiveram 50 vitórias em 50 partidas, sem acertos recebidos nem dano de calor. Foram cinco mapas por bot, seeds 1120000 a 1120004, nos dois lados de spawn. Essa amostra confirma a manutenção dos resultados; a melhoria de calor foi demonstrada nos dois cenários específicos, que falham na versão anterior. A suíte completa passou nas 23 verificações, e os 20 testes específicos do generalista também passaram.

Na comparação de 500 partidas por versão, os acertos recebidos caíram de 161 para 45. A versão anterior teve 499 vitórias e um empate; a corrigida teve 498 vitórias e dois empates, sem derrotas. Os eventos de calor passaram de cinco para oito. A saída antecipada de impasses e o calor residual continuam pendentes.

Os resultados completos e as etapas anteriores ficam no arquivo de acompanhamento do projeto. Este README descreve o uso da versão atual.

## Base para os próximos testes contra especialistas

A candidata anterior aos experimentos de aproximação contra Camper está
preservada em [experiments/generalista_base.py](experiments/generalista_base.py).
Ela usa os pesos `weights/generalista/qtable.npz` e é separada do algoritmo
oficial `algo1.py` e das pastas de concurso.

Os [resultados reproduzíveis por seed e lado](experiments/generalista_base_results.json)
registram 272 vitórias, 37 empates e 41 derrotas em 350 partidas (77,7%).
Foram 70 partidas por especialista: Camper 64 vitórias, Strafe 56, Dijkstra 51,
A* 68 e Markov 33. Essa é a base para o trabalho prioritário contra Markov.
As alterações de espaçamento testadas contra Camper não fazem parte dela.

[Voltar à documentação do desafio](../README.md)
