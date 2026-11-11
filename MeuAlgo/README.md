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
.\.venv\Scripts\python.exe tools/test_powerup_policy.py
.\.venv\Scripts\python.exe tools/test_markov_return.py
.\.venv\Scripts\python.exe tools/test_markov_defense.py
.\.venv\Scripts\python.exe tools/test_contest_equivalence.py
.\.venv\Scripts\python.exe tools/verify.py
```

São 44 testes específicos: nove de perfis e navegação, onze de defesa, quinze de coleta e bônus, seis de retorno e três de regressão contra Markov. A verificação geral executa outras 23 checagens do projeto.

## Estado atual

### Revisão dos empates

A política atual coleta uma vida próxima mesmo estando com vida cheia
quando isso impede a cura de um adversário ferido. A mudança mantém os
limites de distância e retorno seguro e está no código principal e nos dois
slots de concurso. Os pesos permanecem iguais aos do commit `1d53543`.

Nos mesmos 500 confrontos contra especialistas, com 100 por adversário:

| Resultado | Antes | Atual |
|---|---:|---:|
| Vitórias | 424 (84,8%) | 429 (85,8%) |
| Empates | 48 | 45 |
| Derrotas | 28 | 26 |

Oito empates antigos viraram vitórias. O saldo inclui também quatro vitórias
que viraram empates e uma que virou derrota; duas derrotas viraram vitórias
e outra virou empate. A redução líquida foi de três empates, com cinco
vitórias adicionais. As taxas por especialista foram Camper 88%, Strafe 84%,
Dijkstra 79%, A* 98% e Markov 80%.

Em cinco mapas inéditos, nos dois lados, a comparação contra especialistas
passou de 32/50 para 33/50 vitórias, de 15 para 14 empates e manteve três
derrotas. Contra os bots padrão, ambas as versões ganharam 50/50.
Essa amostra nova expôs dificuldades que a bateria principal não mostrava:
Camper 8/10, Strafe 6/10, Dijkstra 7/10, A* 7/10 e Markov 5/10 na versão
atual. A meta de 75% por adversário foi mantida na bateria principal;
não foi mantida em todos os adversários dessa pequena amostra inédita.

As variações de mira foram descartadas porque também criavam novos
empates em partidas anteriormente ganhas. Os resultados completos,
as transições, hashes e limites estão no
[relatório da revisão](experiments/generalista_draw_results.json).
Passaram os 44 testes específicos, as 23 verificações gerais, a validação
dos dois slots e 14 partidas de equivalência com ações idênticas entre
o desenvolvimento e as duas entregas.

### Baterias anteriores à revisão dos empates

A versão do commit `1d53543`, anterior à revisão dos empates, foi validada
nas baterias descritas abaixo. Os pesos foram mantidos. Ela coleta vida, velocidade e
tiro rápido por rotas próximas e seguras, aproveita imediatamente vantagens
temporárias e preserva as correções de esquiva e retorno à zona segura.
Não escolhe comportamentos pelo nome do adversário nem pela seed.

Superou a meta de 75% de vitórias contra **cada especialista treinado**
nas duas baterias abaixo. Cada mapa foi jogado dos dois lados; a segunda
bateria usou 20 mapas novos, com a política congelada antes dos resultados.

| Especialista | Bateria original (70 partidas cada) | Mapas novos (40 partidas cada) |
|---|---|---|
| Camper | 65/70 — 92,9% | 34/40 — 85% |
| Strafe | 60/70 — 85,7% | 33/40 — 82,5% |
| Dijkstra | 59/70 — 84,3% | 35/40 — 87,5% |
| A* | 70/70 — 100% | 40/40 — 100% |
| Markov | 60/70 — 85,7% | 34/40 — 85% |
| Total | 314/350 — 89,7% | 176/200 — 88% |

Contra os **bots padrão**, foram 20 vitórias em 20 partidas contra cada um
(100/100 no total), mantendo o resultado anterior nos mesmos mapas e lados.
Não houve dano de calor nessa bateria. A taxa de vitórias foi preservada;
isso não significa que todos os demais indicadores sejam idênticos.

Passaram 40 testes específicos, as 23 verificações gerais do projeto
(incluindo 32 regressões dos especialistas), a validação oficial dos dois
slots e duas partidas completas no executor de concurso. Nas 12 partidas
de equivalência, os dois scripts de concurso produziram ações idênticas
às do código de desenvolvimento durante todos os 41.700 passos.

O [relatório completo](experiments/generalista_75_each_results.json) guarda
placares, seeds, lados, hashes e a comparação com a candidata anterior
(277/350; Markov 37/70). As taxas descrevem essas amostras e não garantem
o mesmo resultado em todo mapa futuro.

As seções seguintes registram etapas anteriores, preservadas para comparação.

### Histórico anterior à promoção de 10 de novembro

O checkpoint oficial recebeu 100 partidas de treino equilibradas entre os cinco bots. A correção de defesa alterou o código, mantendo esses pesos.

A proteção de calor também confere movimentos sem projéteis próximos: prevê o fechamento da zona durante os próximos 0,8 s e evita sair da área segura. Rotas de retorno planejadas mantêm os desvios necessários para contornar paredes. Dois cenários que falhavam antes agora passam, incluindo um em que a versão anterior perdia uma vida no próximo tick. A mesma correção está nos dois scripts de concurso.

Na comparação de 9 de novembro de 2026, antes e depois mantiveram 50 vitórias em 50 partidas, sem acertos recebidos nem dano de calor. Foram cinco mapas por bot, seeds 1120000 a 1120004, nos dois lados de spawn. Essa amostra confirma a manutenção dos resultados; a melhoria de calor foi demonstrada nos dois cenários específicos, que falham na versão anterior. A suíte completa passou nas 23 verificações, e os 20 testes específicos do generalista também passaram.

Na comparação de 500 partidas por versão, os acertos recebidos caíram de 161 para 45. A versão anterior teve 499 vitórias e um empate; a corrigida teve 498 vitórias e dois empates, sem derrotas. Os eventos de calor passaram de cinco para oito. A saída antecipada de impasses e o calor residual continuam pendentes.

Os resultados completos e as etapas anteriores ficam no arquivo de acompanhamento do projeto. Este README descreve o uso da versão atual.

### Base preservada dos testes contra especialistas

A candidata anterior aos experimentos de aproximação contra Camper está
preservada em [experiments/generalista_base.py](experiments/generalista_base.py).
Ela usa os pesos `weights/generalista/qtable.npz` e é separada do algoritmo
oficial `algo1.py` e das pastas de concurso.

Os [resultados reproduzíveis por seed e lado](experiments/generalista_base_results.json)
registram 272 vitórias, 37 empates e 41 derrotas em 350 partidas (77,7%).
Foram 70 partidas por especialista: Camper 64 vitórias, Strafe 56, Dijkstra 51,
A* 68 e Markov 33. Essa é a base para o trabalho prioritário contra Markov.
As alterações de espaçamento testadas contra Camper não fazem parte dela.

A primeira candidata para Markov está em
[experiments/markov_defense.py](experiments/markov_defense.py), com os mesmos
pesos da base. Em empates de dano previsto, a esquiva prefere um movimento
que não receba o acerto no próximo passo da simulação. Nos 70 confrontos
originais contra Markov, passou de 33/23/14 para 34/23/13 (vitórias/empates/derrotas),
preservando as 33 vitórias e os 23 empates. Os 40 confrontos de verificação
contra os demais especialistas e os 20 confrontos contra Markov em dez mapas
novos mantiveram seus resultados anteriores. A bateria equilibrada de 50
partidas permaneceu em 68%; a bateria completa de 350 não foi repetida.
Os [resultados e limites da avaliação](experiments/markov_defense_results.json)
acompanham a candidata; ela ainda é separada da versão oficial e das pastas
de concurso. Passaram 20 testes existentes e três testes de regressão
baseados em um estado real de partida (`tools/test_markov_defense.py`).

A candidata intermediária de desenvolvimento é
[experiments/markov_return.py](experiments/markov_return.py). Além do desempate
da esquiva, verifica a rota até uma área segura para o fim da partida e sai
de bolsões antes que suas passagens fechem. Após teleporte, recalcula essa
rota imediatamente. A base e a candidata anterior permanecem preservadas.

Na comparação equilibrada de 350 partidas (70 por especialista), a taxa
geral passou de 272/350 (77,7%) para 277/350 (79,1%). Contra Markov passou
de 33/70 (47,1%) para 37/70 (52,9%), preservando suas 33 vitórias anteriores.
Três derrotas e um empate contra Markov viraram vitórias. Os resultados por
especialista foram Camper 65/70, Strafe 55/70, Dijkstra 51/70, A* 69/70 e
Markov 37/70. Strafe perdeu uma vitória em relação à base e continua sendo
uma ressalva. O dano de calor total caiu de 23 para quatro vidas.

Em cinco mapas novos, nos dois lados e contra os cinco especialistas
(50 partidas por versão), a taxa passou de 66% para 70%. Em um controle
separado de 20 partidas contra Markov, manteve seis vitórias, nove empates
e cinco derrotas. A revisão final foi repetida em todos os confrontos com
teleporte; os registros sem teleporte foram reaproveitados após verificar
que a mudança de recálculo não altera esse caminho. Os detalhes estão em
[experiments/markov_return_results.json](experiments/markov_return_results.json).
Passaram 29 testes: 20 existentes, três de esquiva e seis de retorno.
Essa candidata intermediária fica preservada como referência. A versão
oficial é a descrita no início de "Estado atual", também preservada em
[experiments/generalista_candidate.py](experiments/generalista_candidate.py).

[Voltar à documentação do desafio](../README.md)
