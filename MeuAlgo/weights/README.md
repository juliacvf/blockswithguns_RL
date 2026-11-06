# Pesos das políticas

| Caminho | Uso |
|---|---|
| [generalista/qtable.npz](generalista/qtable.npz) | Modelo oficial do generalista, perfil `generalista_v1`. É o arquivo usado nos comandos atuais de avaliação e visualização. |
| [especialistas/camper.npz](especialistas/camper.npz) | Especialista Camper; usa o perfil original `general`. |
| [especialistas/strafe.npz](especialistas/strafe.npz) | Especialista Strafe, perfil `strafe_v1`. |
| [especialistas/dijkstra.npz](especialistas/dijkstra.npz) | Especialista Dijkstra, perfil `dijkstra_v1`. |
| [especialistas/astar.npz](especialistas/astar.npz) | Especialista A*, perfil `astar_v1`. |
| [especialistas/markov.npz](especialistas/markov.npz) | Especialista Markov, perfil `markov_v1`. |
| [qtable.npz](qtable.npz) | Modelo legado do perfil `general`. Mantido porque a entrada antiga que recebe esta pasta e o treino sem perfil específico ainda usam esse caminho. |

Os arquivos não são intercambiáveis: cada perfil tem sua própria representação de estado e quantidade de ações. O carregador lê o perfil do checkpoint.

Mantenha somente um checkpoint oficial em cada caminho. Use `--output` para salvar candidatos fora desta pasta. Os envios independentes do concurso têm suas próprias cópias dos pesos em `Concurso/slot1/weights/` e `Concurso/slot2/weights/`.

[Comandos de uso, treino e avaliação](../README.md)
