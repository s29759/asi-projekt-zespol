# ADR-2: Wybór modelu produkcyjnego

- Status: Proponowany
- Sprint: S3
- Autor: Alina Shevtsova
- Recenzent: Kostiantyn Feniuk

## Kontekst

Porównaliśmy LogisticRegression oraz dwa warianty AutoGluon na tym samym
zbiorze testowym (570 rekordów). Kryteria wyboru to F1 dla klasy `Churn=1`,
czas predykcji, rozmiar zapisanego modelu i przyrost RSS po wczytaniu.
Pełne wyniki i metodykę opisano w [porównaniu modeli](../model_comparison.md).

## Rozważane warianty

Wartości F1 poniżej pochodzą z eksportu wyników eksperymentów W&B.
Po zaokrągleniu do czterech miejsc po przecinku pokrywają się z wynikami
lokalnego porównania. Czas predykcji, rozmiar i RSS zmierzono lokalnie.

- **LogisticRegression** osiągnął F1 równy 0,6087. Był najszybszy
  i najmniejszy, ale uzyskał najniższy wynik F1.
- **AutoGluon `medium_60s`** osiągnął F1 równy 0,8636. Predykcja dla
  570 rekordów zajęła około 38 ms, a zapisany model miał 37,71 MiB.
- **AutoGluon `best_960s`** osiągnął najwyższy F1 (0,8800). Względem
  `medium_60s` zyskał 0,0164 F1, a predykcja trwała około 7,2 razy dłużej
  i katalog modelu zajmował około 9,6 razy więcej miejsca. Zmierzony
  przyrost RSS po wczytaniu był podobny dla obu wariantów AutoGluon.

## Decyzja

Proponujemy `medium_60s` jako model produkcyjny. Osiąga F1 o 0,2549 wyższy
niż baseline. Dodatkowy wzrost F1 w wariancie `best_960s` nie uzasadnia
dłuższego czasu predykcji i większego rozmiaru zapisanego modelu.

## Konsekwencje

Po zaakceptowaniu decyzji przekażemy wytrenowany model jako pakiet
z `metadata.json`. Do zapisu i odczytu pakietu użyjemy wspólnych funkcji
z `model_package.py`. Pakiet opublikujemy jako W&B Artifact z aliasem
`production`, z którego API będzie mogło pobrać model bez ponownego treningu.
