# ADR-2: Wybór modelu produkcyjnego

- Status: Proponowany
- Autor: Alina Shevtsova
- Recenzent: Kostiantyn Feniuk

## Kontekst

Porównaliśmy LogisticRegression oraz dwa warianty AutoGluon na tym samym
zbiorze testowym (570 rekordów). Kryteria wyboru to F1 dla klasy `Churn=1`,
czas predykcji, rozmiar zapisanego modelu i przyrost RSS po wczytaniu.
Pełne wyniki i metodyka są w [porównaniu modeli](../model_comparison.md).

## Rozważane warianty

- LogisticRegression osiągnął F1 0,6087. Jest najszybszy i najmniejszy,
  ale wykrywa odejścia klientów gorzej niż oba warianty AutoGluon.
- AutoGluon `medium_60s` osiągnął F1 0,8636. Predykcja dla 570 rekordów
  zajęła około 39 ms, a zapisany model około 37,71 MiB.
- AutoGluon `best_960s` osiągnął najwyższe F1 (0,8800). Zysk 0,0164
  względem `medium_60s` oznacza około 7 razy dłuższą predykcję i około
  9,6 razy większy katalog modelu. Zmierzony przyrost RSS przy wczytaniu
  był podobny dla obu wariantów AutoGluon.

## Decyzja

Wybieramy `medium_60s`. Osiąga F1 o 0,2549 wyższe niż baseline,
a względem `best_960s` ogranicza czas predykcji i rozmiar modelu.

Wytrenowany model przekażemy w pakiecie z `metadata.json`,
zapisywanym i wczytywanym funkcjami z `model_package.py`.
Pakiet zostanie opublikowany jako W&B Artifact z aliasem `production`.
