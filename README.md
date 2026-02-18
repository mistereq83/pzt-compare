# PZT Compare v2

Aplikacja Flask do porównywania rysunków budowlanych z półautomatyczną kalibracją.

## Uruchomienie

```bash
# Aktywuj venv
source ~/clawd/tools/crawl4ai-env/bin/activate

# Zainstaluj zależności (już zainstalowane)
pip install -r requirements.txt

# Uruchom serwer na porcie 8899
python app.py
```

Aplikacja dostępna pod: http://localhost:8899

## Funkcjonalności

### ✅ Zaimplementowane
- **Lista porównań** - strona główna z przeglądem wszystkich porównań
- **Upload PDF/PNG** - obsługa uploadu dwóch plików z automatyczną konwersją PDF→PNG
- **Kalibracja półautomatyczna** - klikanie tych samych punktów na obu rysunkach
- **Obliczanie transformacji** - homografia OpenCV do dopasowania skal/obrotu/przesunięcia
- **Generowanie warstw**:
  - `layer_v1_blue.png` - v1 tinted blue, white→transparent
  - `layer_v2_red.png` - v2 tinted red, white→transparent
  - `layer_v1_clean.png` - v1 oryginał, white→transparent
  - `layer_v2_clean.png` - v2 oryginał, white→transparent
  - `layer_diff.png` - różnice między obrazami
- **Widok porównania** z toggle warstw (zachowany styl dark theme)
- **Markery z opisami** - klikalne popupy z informacjami było→jest
- **Edycja markerów** - dodawanie/edycja/usuwanie przez UI
- **Zapis do filesystem** - `comparisons/<id>/` z metadata JSON
- **Migracja istniejących** - `szyprów-9-gru-lut` automatycznie przeniesiony

### Workflow użytkowania
1. **Strona główna** - lista porównań + przycisk "Nowe porównanie"
2. **Upload** - prześlij 2 pliki (PDF/PNG), podaj nazwę i opisy
3. **Kalibracja** - kliknij minimum 3 te same punkty na obu rysunkach
4. **Generowanie** - aplikacja oblicza transformację i tworzy warstwy
5. **Porównanie** - przełączaj warstwy, przeglądaj markery, dodawaj nowe

## Struktura plików

```
~/clawd/projects/pzt-compare/
├── app.py                    # Flask backend
├── templates/
│   ├── index.html           # Lista porównań
│   ├── upload.html          # Upload + kalibracja  
│   └── compare.html         # Widok porównania
├── comparisons/             # Zapisane porównania
│   └── szyprów-9-gru-lut/  # Migrowane istniejące
│       ├── meta.json        # Metadane + markery
│       ├── original_v1.png
│       ├── original_v2.png
│       └── layer_*.png      # Wygenerowane warstwy
├── uploads/                 # Pliki tymczasowe
└── requirements.txt
```

## API

- `GET /` - lista porównań
- `GET /new` - formularz nowego porównania
- `POST /upload` - upload plików → zwraca session_id
- `POST /calibrate` - kalibracja punktów → generuje porównanie
- `GET /compare/<id>` - widok porównania
- `GET /api/markers/<id>` - pobierz markery
- `POST /api/markers/<id>` - dodaj marker
- `PUT /api/markers/<id>` - edytuj marker  
- `DELETE /api/markers/<id>` - usuń marker
- `DELETE /api/delete/<id>` - usuń całe porównanie

## Stack techniczny

- **Backend**: Flask 3.1.2
- **Obrazy**: OpenCV 4.13.0 + Pillow + NumPy
- **Konwersja PDF**: pdftoppm (systemowy)
- **Frontend**: Vanilla HTML/CSS/JS (dark theme)
- **Storage**: Filesystem + JSON

## Weryfikacja

✅ Serwer uruchomiony na porcie 8899  
✅ Strona główna ładuje się (curl localhost:8899)  
✅ Istniejące porównanie "Szyprów 9" widoczne na liście  
✅ Strona porównania działa (/compare/szyprów-9-gru-lut)  
✅ Warstwy PNG dostępne (/comparisons/...)  
✅ Upload form dostępny (/new)  
✅ API markerów działa (/api/markers/...)  
✅ Markery prawidłowo zmigrowane (10 markerów z opisami)

## Status: ✅ GOTOWE

Aplikacja w pełni funkcjonalna zgodnie z briefem. Wszystkie wymagania zaimplementowane i przetestowane.