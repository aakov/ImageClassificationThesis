# Kod i wyniki do pracy magisterskiej

Porównanie skuteczności modeli transformatorowych w klasyfikacji obrazów.

## Zawartość
- Skrypty train_deit*.py, train_swin*.py i deit_initiaor.py: uczenie modeli i uruchamianie serii.
- deit_tfhub.py, swin_tfhub.py: budowa i adaptacja modeli; tfhub_cache.py: obsługa pamięci podręcznej wag.
- experiment_config.py, data_loader.py, data_generator.py, experiment_utils.py: konfiguracja, dane i wspólne operacje.
- evaluate_models.py: ocena zapisanych modeli na zbiorze testowym lub walidacyjnym.
- smoke_test_pipeline.py: sprawdzenie danych i konfiguracji przed uczeniem.
- outputs/experiments/runs/: oryginalne logi, konfiguracje i wyniki testowe 17 głównych przebiegów oraz konfiguracje i logi głowicy dwóch dodatkowych przebiegów porównujących tryby eager/grafowy.
- docs/derived_results_v3.json: wybór przebiegów i wcześniejsze opracowanie wyników.
- analiza_wynikow/generate_*.py: skrypty opracowania wyników i generowania tabel oraz wykresów.
- analiza_wynikow/v5_assets, v6_assets, v7_assets: zestawienia pochodne używane przez kolejne skrypty. To opracowania, a nie bezpośrednie zapisy treningu.

## Odtworzenie analizy bez treningu
Uruchomić z katalogu tego pakietu:

    python odtworz_analize.py

Wymaga NumPy, pandas i Matplotlib. Polecenie nie uruchamia modeli ani nie pobiera wag. Sprawdza źródłowe wyniki i generuje tabele oraz wykresy w folderach analiza_wynikow/v5_assets, v6_assets i v7_assets. Nadpisuje pliki opracowań; nie zmienia logów w outputs/experiments/runs. Wersje v5/v6/v7 oznaczają etapy opracowania pracy, nie trzy odrębne serie eksperymentów.

## Środowisko i uruchamianie modeli
requirements.txt podaje wersje pakietów odczytane z lokalnego środowiska tf dnia 19.09.2026; nie jest archiwalną migawką środowiska każdego treningu. W pracy używano Python 3.10.13, TensorFlow/Keras 2.10, pojedynczej karty RTX 4060 Ti 16 GB, CUDA i cuDNN. Sama instalacja pakietów Python nie konfiguruje sterowników ani bibliotek GPU.

Przykładowe polecenia (z katalogu pakietu, po przygotowaniu środowiska i danych):

    python smoke_test_pipeline.py --experiment stanford_deit
    python train_deit.py --backbone deit_tiny
    python train_deit_oxfordpets.py --backbone deit_tiny
    python train_swin.py --dataset stanford --backbone swin_tiny
    python evaluate_models.py --model SCIEZKA_DO_MODELU.keras --dataset stanford --backbone swin_tiny --split test

Polecenia train_* rozpoczynają trening. Budowa modeli może pobrać wagi z TF Hub. Ponowna ocena wymaga zapisanego modelu oraz obrazów; tych dużych plików nie dołączono.

Dane: Stanford Cars i Oxford-IIIT Pet. Układ ścieżek określają experiment_config.py i data_loader.py. Dla Stanford Cars wymagane są obrazy, cars_meta.mat, cars_train_annos.mat i etykietowane adnotacje testowe cars_test_annos_withlabels.mat. Dla Oxford-IIIT Pet wymagane są obrazy i adnotacje ras. Położenie danych należy dostosować w obu wskazanych modułach. Zapisane split_manifest.csv dokumentują oryginalne podziały, a config.json ustawienia konkretnych przebiegów; ścieżki wewnątrz tych plików odnoszą się do oryginalnego komputera.

## Wyniki źródłowe i korekta czasu
Pliki w runs zachowano bez zmian. evaluation/official_test_metrics.json zawiera końcowe metryki, a official_test_predictions.csv etykiety i przewidywane klasy. Są też metryki klas i macierze pomyłek. Logi head/fine zawierają pomiary epok, a training_history historię uczenia i walidacji.

W opracowaniu przebiegu 20260830_062653_oxfordpets_deit_deit_small_distilled zastąpiono czas szóstej epoki dostrajania 21100,035 s medianą pozostałych epok 35,861 s. Surowe logi w pakiecie nadal zawierają oryginalny pomiar. Pozostałe czasy w tabelach są sumami czasów epok. Kod tej korekty znajduje się w generate_v5_assets.py i generate_v5_performance.py.
