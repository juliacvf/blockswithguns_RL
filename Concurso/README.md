# Envios do generalista para concurso

Estas são as pastas prontas para entregar o generalista finalizado em 11 de novembro de 2026. Os dois slots executam a mesma política; o nome do script atende ao slot solicitado pelo concurso.

A fuga antecipada passou de 429 para 431 vitórias, de 45 para 47 empates e de 26 para 22 derrotas nos mesmos 500 confrontos contra especialistas. Na última amostra de cinco mapas de 10/11, passou de 33/50 para 36/50 e manteve 50/50 contra bots. Os pesos originais foram preservados. Os [resultados por adversário e limites](../MeuAlgo/README.md#estado-atual) e o [relatório completo](../MeuAlgo/experiments/generalista_pocket_escape_yesterday_seeds_20261111_results.json) registram as amostras avaliadas.

## Sumário

- [Escolher o envio](#escolher-o-envio)
- [Validar as pastas](#validar-as-pastas)
- [Assistir a uma partida](#assistir-a-uma-partida)
- [Contrato e dependências](#contrato-e-dependências)
- [Atualizar uma entrega](#atualizar-uma-entrega)

## Escolher o envio

```text
Concurso/
  README.md
  slot1/
    algo1.py
    weights/qtable.npz
  slot2/
    algo2.py
    weights/qtable.npz
```

Envie apenas a pasta do slot solicitado. Se o envio for um ZIP, o script e o diretório `weights/` devem ficar diretamente na raiz, sem uma pasta `Concurso/` envolvendo os arquivos.

| Slot | Script obrigatório | Pesos |
|---|---|---|
| 1 | [slot1/algo1.py](slot1/algo1.py) | [slot1/weights/qtable.npz](slot1/weights/qtable.npz) |
| 2 | [slot2/algo2.py](slot2/algo2.py) | [slot2/weights/qtable.npz](slot2/weights/qtable.npz) |

Todos os comandos abaixo partem da pasta principal do projeto e usam PowerShell.

## Validar as pastas

```powershell
.\.venv\Scripts\python.exe -m rl.contest --algo1-folder Concurso/slot1 --algo2-folder Concurso/slot2 --validate-only
```

Para executar uma batalha entre as duas cópias, com validação rígida das ações:

```powershell
.\.venv\Scripts\python.exe -m rl.contest --algo1-folder Concurso/slot1 --algo2-folder Concurso/slot2 --episodes 2 --no-replay
```

## Assistir a uma partida

```powershell
.\.venv\Scripts\python.exe -m rl.visualize --player1 custom --algo1-folder Concurso/slot1 --player2 camper --seed 1020100 --bot2-seed 1020101
```

Ao terminar, Enter fecha a janela. Troque `camper` por outro bot conhecido para assistir a outra combinação.

## Contrato e dependências

- Os scripts expõem `Agent` e `make_agent`; ambas as entradas carregam o mesmo generalista.
- O diretório de pesos recebido pelo concurso contém `qtable.npz`.
- `act(obs)` devolve exatamente três inteiros: movimento de 0 a 8, mira de 0 a 15 e tiro 0 ou 1.
- `reset()` limpa o estado antes de cada episódio.
- A execução precisa de Python 3.10 ou superior e NumPy. Os scripts não importam os módulos `core`, `rl`, `bots` ou `MeuAlgo` do projeto.

Os valores da Q-table são os mesmos do modelo oficial; metadados de treino e caminhos pessoais foram removidos dos arquivos de envio. As constantes e funções auxiliares necessárias estão incluídas nos próprios scripts. A verificação da política atual inclui o validador oficial e 22 partidas completas de equivalência entre os dois slots, o código de desenvolvimento e a variante vencedora arquivada.

## Atualizar uma entrega

O desenvolvimento fica em [MeuAlgo](../MeuAlgo/README.md). Estas pastas são cópias independentes para envio: mudanças futuras no código ou nos pesos de desenvolvimento precisam ser incorporadas aos dois scripts e validadas antes de gerar novos ZIPs.

[Voltar à documentação do desafio](../README.md)
