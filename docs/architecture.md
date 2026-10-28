# Architektura systemu

Ten dokument opisuje architekturę systemu zespołu. W Sprincie 1 powstaje diagram v0, który
będzie aktualizowany w kolejnych sprintach.

## Architektura docelowa (scaffold)

Tak wygląda system po sześciu sprintach — na poziomie warstw. To mapa odpowiedzialności,
a nie diagram do skopiowania: diagram v0 (niżej) rysujecie sami.

Relacje między warstwami (opis tekstowy, nie diagram):

- training: dane → pipeline Kedro → artefakt modelu
- serving: UI → API → model (artefakt wczytany przez API)
- state: API → persistence
- deployment: API + UI → Cloud Run

| Warstwa | Odpowiedzialność | Sprint | Miejsce w repozytorium |
|---|---|---|---|
| Dane i kontrakt | źródło danych, kontrakt, input normalization, training-only filtering | S1 | `data/01_raw/`, `src/asi_project/datasets.yaml`, `src/asi_project/data.py` |
| Pipeline | odtwarzalne przetwarzanie i trening od surowych danych | S2 | `conf/`, `src/asi_project/pipelines/` |
| Eksperymenty i artefakt | porównanie runów, wybór modelu, pakiet modelu ładowalny poza treningiem | S3 | W&B, pakiet modelu, `docs/model_card.md` |
| API | kontrakt usługi, walidacja na granicy, model wczytywany raz | S4 | `src/asi_project/api/`, `docker/` |
| Persistence | zapis predykcji poza procesem aplikacji (tryby `sql` / `disabled`) | S4 → S5 → S6 | repository w API, Postgres w compose |
| UI | formularz jako klient API | S5 | `ui/`, `docker/`, `docker-compose.yml` |
| Chmura | wdrożenie, sekret, logi, release | S6 | `deploy/`, Cloud Run, Secret Manager, Cloud Logging |

## Diagram v0 (Sprint 1)

Narysujcie, jak — według waszej obecnej wiedzy — będzie wyglądał system na koniec projektu.
Diagram v0 nie musi być poprawny w szczegółach; ma pokazać wasze rozumienie przepływu.
Sama lista warstw z sekcji wyżej nie wystarcza — diagram pokazuje wasz system i wasze decyzje.

Na diagramie zaznaczcie co najmniej:

- skąd pochodzą dane i gdzie żyje kontrakt danych;
- gdzie działa każdy z trzech typów przetwarzania: input normalization, training-only
  filtering, model preprocessing;
- gdzie powstaje model i jak trafia tam, gdzie wykonuje się predykcja;
- kto lub co wysyła żądanie predykcji i co dostaje w odpowiedzi;
- gdzie przechowywany jest stan (np. zapis predykcji);
- które elementy już istnieją po Sprincie 1, a które są planowane.

Diagram umieśćcie w bloku `mermaid` w sekcji poniżej (GitHub renderuje Mermaid w Markdown)
albo jako obraz w `docs/` z linkiem w tej sekcji. Usuńcie znaczniki TODO, gdy diagram będzie
gotowy — sprawdza to `python scripts/check_dod.py`.

<!-- DIAGRAM-V0:START -->
```mermaid
flowchart LR
    contract["Kontrakt danych<br/>datasets.yaml + project.yaml"]

    subgraph TRAIN["Trening"]
        src["UCI Iranian Churn"] -->|download_data.py| raw["Customer Churn.csv<br/>poza Git, SHA-256"]
        raw --> norm1["input normalization"]
        norm1 --> filt["training-only filtering"]
        filt --> split["make_xy + split"]
        split --> fit["Pipeline: model preprocessing<br/>+ LogisticRegression<br/>notebook 02"]
        fit --> metrics["metrics.json"]
    end

    fit --> artifact["artefakt modelu<br/>S3"]
    contract -.-> norm1

    subgraph CLOUD["Cloud Run, S6"]
        user["Użytkownik"] --> ui["UI, S5"]
        ui -->|cechy klienta| api["API, S4<br/>walidacja → input normalization<br/>→ predict_proba"]
        api -->|szansa na odejście| ui
    end

    artifact -->|wczytany raz| api
    contract -.-> api
    api --> db[("Postgres<br/>zapis predykcji")]

    classDef planned stroke-dasharray:5 5
    class artifact,user,ui,api,db planned
```

Dane z UCI (Iranian Churn) pobieramy skryptem i nie trzymamy ich w Gitcie. Kontrakt danych jest w `datasets.yaml` i `project.yaml`.
Przy treningu robimy input normalization i training-only filtering, potem split na train i test.
Preprocessing jest w sklearn Pipeline razem z LogisticRegression, więc uczy się tylko na train.
Po Sprincie 1 mamy trening i `metrics.json`. Przerywane ramki to plan na S3-S6: artefakt modelu, API, UI, Postgres i Cloud Run.
<!-- DIAGRAM-V0:END -->

## Historia zmian architektury

| Sprint | Zmiana | Powiązany ADR |
|---|---|---|
| S1 | diagram v0 | — |
