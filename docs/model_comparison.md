# Porównanie modeli — Sprint 3

Wspólny zbiór testowy: 570 rekordów. F1 dla klasy `Churn=1`.

| Model | F1 | Czas predykcji [ms / zbiór testowy] | Rozmiar [MiB] | Przyrost RSS [MiB] |
| --- | ---: | ---: | ---: | ---: |
| logistic_regression | 0.6087 | 2.709 | 0.0048 | 100.70 |
| medium_60s | 0.8636 | 37.745 | 37.7129 | 120.65 |
| best_960s | 0.8800 | 270.668 | 361.3294 | 121.18 |

Czas predykcji to mediana z 30 wywołań po rozgrzewce na całym zbiorze testowym.
Rozmiar obejmuje wszystkie pliki pakietu baseline lub katalogu AutoGluon.
Przyrost RSS zmierzono po wczytaniu modelu w osobnym procesie,
przed pierwszą predykcją. Pomiar obejmuje biblioteki wczytane w tym kroku.
