# ADR-1: Gdzie żyje przetwarzanie danych

- **Status:** przyjęty
- **Sprint:** S2
- **Autor:** Kostiantyn Feniuk
- **Recenzent:** Łukasz de Sousa

## Kontekst

Mamy trzy rodzaje przetwarzania: input normalization, training-only filtering i model
preprocessing. Teraz wszystko idzie przez Kedro, a w S4 dojdzie API, które dostaje jednego
klienta naraz. Trzeba ustalić, gdzie każde z nich działa, żeby trening i API liczyły to samo
i żeby dane z testu nie wpływały na trening.

## Rozważone opcje

1. Wszystko w pipeline `data_processing` (normalizacja, filtrowanie, skalowanie i one-hot),
   a do modelu idą już gotowe liczby.
2. Każde przetwarzanie w innym miejscu, zależnie od tego, kiedy ma działać.

## Decyzja

Wybieramy opcję 2:

| Przetwarzanie | Gdzie | Kiedy działa |
|---|---|---|
| input normalization | `normalize_input` w `data.py`, node `input_normalization` | trening i później API |
| training-only filtering | `filter_training_rows` w `data.py`, node `training_only_filtering` | tylko trening |
| model preprocessing | sklearn `Pipeline` w `model.py` (imputer, `StandardScaler`, `OneHotEncoder`) | `fit` tylko na train, potem razem z modelem |

Input normalization działa na pojedynczym rekordzie i nie patrzy na inne wiersze ani na
target, więc API może użyć tej samej funkcji. Filtrowanie usuwa wiersze, a w API nie wolno
wyrzucić zapytania klienta, dlatego jest tylko w treningu. Preprocessing modelu uczy się
statystyk (mediana, średnia, kategorie), więc musi siedzieć w sklearn `Pipeline` i uczyć się
tylko na train. Wtedy zapisany model ma te statystyki w środku.

| Wielkość | Wartość | Skąd pochodzi |
|---|---|---|
| wiersze przed / po training-only filtering | 3150 / 2850 (300 duplikatów) | `kedro run`, `data/02_intermediate/processed_data.csv` |
| wiersze, na których uczy się preprocessing | 2280 (train), test 570 | `metrics.json`, `split` |
| F1 LogisticRegression / Dummy | 0.6087 / 0.0 | `metrics.json` |
| cechy w X | 12 ALLOWED, bez `Customer Value` | `project.yaml`, `datasets.yaml` |

## Odrzucone alternatywy

Opcję 1 odrzuciliśmy. Skaler i encoder liczone przed splitem widzą też dane testowe, więc
metryki byłyby za dobre. Do tego API musiałoby osobno wczytać skaler i encoder i pilnować,
żeby były te same co przy treningu. Usuwanie duplikatów w `normalize_input` też odpada,
bo API dostaje jeden rekord i nie ma czego porównywać.

## Konsekwencje

- `normalize_input` musi zostać prosta: bez targetu, bez statystyk zbioru, idempotentna.
  Pilnują tego testy w `tests/test_project.py`.
- Nowe reguły usuwania wierszy idą tylko do `filter_training_rows`.
- W S3 artefaktem jest cały sklearn `Pipeline`, nie sam klasyfikator, bo preprocessing jest
  w środku.
- W S4 API wywołuje `normalize_input`, a potem `predict_proba` na wczytanym `Pipeline`.

## Historia

| Data | Zmiana |
|---|---|
| 12.11.2026 | utworzenie |
