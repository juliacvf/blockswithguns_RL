# Exemplo de aprendizado por reforço

`Algo Test` é o exemplo treinável fornecido pelo desafio. Uma Q-table escolhe entre oito táticas; regras de mira, movimento e navegação executam a ação escolhida. Este exemplo é separado do [generalista desenvolvido em MeuAlgo](../MeuAlgo/README.md).

## Arquivos

| Arquivo | Finalidade |
|---|---|
| [algo1.py](algo1.py) | Política do exemplo para o slot 1. |
| [train.py](train.py) | Treino e avaliação sem interface gráfica; usa Dijkstra como adversário padrão. |
| [weights/qtable.npz](weights/qtable.npz) | Checkpoint carregado quando a pasta é usada como entrada de concurso. |

Todos os comandos abaixo partem da pasta principal do projeto e usam PowerShell.

## Assistir ao exemplo

```powershell
.\.venv\Scripts\python.exe -m rl.visualize --player1 custom --algo1-folder "Algo Test" --player2 dijkstra
```

## Avaliar os pesos incluídos

```powershell
.\.venv\Scripts\python.exe "Algo Test/train.py" --evaluate-only --opponent all --eval-episodes 10
```

Com `all`, a avaliação executa dez partidas contra cada um dos cinco bots. Cada partida segue até nocaute ou até o limite oficial de 120 segundos, com desempate por vidas.

## Experimentar um treino

```powershell
.\.venv\Scripts\python.exe "Algo Test/train.py" --episodes 100 --opponent dijkstra --output "$env:TEMP\algo_test\qtable.npz"
```

O destino temporário preserva os pesos incluídos no projeto. Use `--opponent all` para alternar entre os cinco bots. O treino salva o checkpoint após cada partida e avalia o resultado ao terminar. `--resume` continua a partir do checkpoint no destino escolhido.

## Validar como entrada de concurso

```powershell
.\.venv\Scripts\python.exe -m rl.contest --algo1-folder "Algo Test" --algo2-folder algo2 --validate-only
```

[Voltar à documentação do desafio](../README.md)
