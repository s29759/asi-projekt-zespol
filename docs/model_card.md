# Model Card

> Minimalną Model Card wypełnia zespół w Sprincie 3, a aktualizuje w Sprincie 6.

## Model

- **Typ modelu:** AutoGluon `TabularPredictor`, konfiguracja `medium_60s`
  (`presets: medium`, `time_limit: 60`), najlepszy model `WeightedEnsemble_L2`.
  Wybór opisuje [ADR-2](adr/0002-wybor-modelu-produkcyjnego.md).
- **Wersja:** W&B Artifact `churn-model` z aliasem `production`
  (projekt `s29759-asi-project/asi-projekt`). Pakiet: katalog `autogluon_model/` i `metadata.json`,
  zapis i odczyt przez `src/asi_project/model_package.py`, rozmiar ok. 3 MiB
  (`clone_for_deployment`, tylko modele potrzebne do predykcji).
- **Utworzenie:** 25.11.2026, trening przez node `train_autogluon` (`kedro run`),
  publikacja `python scripts/publish_production.py`, autor Kostiantyn Feniuk.

## Przeznaczenie

- **Problem:** binary classification, czy klient operatora komórkowego odejdzie.
  Klasa pozytywna: `Churn = 1`.
- **Użytkownik i decyzja:** dział utrzymania klienta. Model wskazuje klientów z ryzykiem
  odejścia na koniec 9. miesiąca, żeby zdążyć z ofertą przed końcem 12. miesiąca.
- **Nie do:** automatycznych decyzji o kliencie bez człowieka, innych operatorów i rynków
  bez ponownej walidacji, oceny pojedynczej osoby jako jedynej podstawy działania.

## Dane

- **Dataset:** `iranian_churn` (UCI, CC BY 4.0), opis w `docs/datasets.md`.
  Fingerprint SHA-256: `90d5fb6bd1630cd4de4b4d28fcf8b4cb92a8f6ab7484605b0799d47386f7dbe1`.
- **Przygotowanie:** 3150 wierszy, po training-only filtering 2850 (300 duplikatów usuniętych).
- **Split:** `stratified_random`, `test_size 0.2`, `random_state 42`: train 2280, test 570.
  AutoGluon uczy się tylko na train, test służy wyłącznie do oceny.
- **Cechy:** 12 ALLOWED FEATURES, bez `Customer Value` (EXCLUDED).

## Metryki

Zbiór testowy, 570 rekordów, metryka główna F1 dla `Churn = 1`:

| Model | F1 | Accuracy | Precision | Recall |
|---|---:|---:|---:|---:|
| **AutoGluon `medium_60s` (production)** | **0.8636** | 0.9579 | 0.8736 | 0.8539 |
| LogisticRegression (baseline S1/S2) | 0.6087 | 0.9053 | 0.8571 | 0.4719 |
| DummyClassifier | 0.0000 | 0.8439 | 0.0000 | 0.0000 |

Macierz pomyłek modelu production: TN 470, FP 11, FN 13, TP 76.
Czas predykcji i rozmiar wszystkich wariantów: `docs/model_comparison.md`.

## Ograniczenia i ryzyka

- **Dane:** jeden operator z Iranu, 3150 klientów. Mały zbiór, więc metryki na 570 rekordach
  testowych mają dużą wariancję. Część cech to kategorie zakodowane liczbami.
- **Znane błędy:** model przeocza 13 z 89 klientów, którzy odeszli (recall 0.85),
  i daje 11 fałszywych alarmów na 481 klientów, którzy zostali.
- **Zmiana zachowania klientów:** model nie był sprawdzany na nowszych danych;
  przed użyciem na innym okresie trzeba go zwalidować ponownie.

## Wersje

- **Biblioteki:** Python 3.11.17, scikit-learn 1.7.2, pandas 2.3.3, numpy 2.2.6,
  kedro 1.7.0, wandb 0.30.0, autogluon.tabular 1.6.3 (pełna lista w `metadata.json`).
- **Git SHA treningu:** zapisany w `metadata.json` pakietu (pole `git_sha`).
- **Smoke test:** `python scripts/smoke_production.py` pobiera `production` z W&B,
  wczytuje pakiet w osobnym procesie i robi predykcję dla 5 wierszy próbki.
