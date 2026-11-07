# Источники данных (data v2)

Все внешние данные скачаны только с официальных сайтов авторов и лежат в `data/external/`, который добавлен в `.gitignore`.
- Архивы лежат в `data/external/_downloads/`, около 2,3 ГБ.
- Кэш лежит в `data/external/cache/`, около 1,0 ГБ.

Архивы **не распаковываются и ничего из них не исполняется**: `scripts/prepare_external.py` читает JPEG и `.mat` потоком прямо из tar/zip и декодирует их через PIL / `scipy.io.loadmat`. Скрипт запускается так: `.venv/bin/python -I scripts/prepare_external.py`.

Кэш состоит из файлов `<source>_img.npy` (uint8, 128×128 оттенков серого; у Fashion-MNIST 28×28), `<source>_mask.npy` (маски объектов), `<source>_split.npy`, `<source>_class.npy` и `manifest.json`. В manifest записано, к какому сплиту отнесён каждый класс.

| Источник | URL | Лицензия | Скачано | В кэше (train / val / test) | Как разделено |
|---|---|---|---|---|---|
| Caltech-101 (Fei-Fei, Fergus, Perona) | https://data.caltech.edu/records/mzrjq-6wc02 (`caltech-101.zip`, 137 МБ) | CC BY 4.0 | 9 144 изображения, 102 класса; класс BACKGROUND_Google исключён | `c101_photo`: 8 677 (6 817 / 920 / 940); `c101_obj`: 8 673 объекта с масками из официальных контуров (6 813 / 920 / 940) | по классам 71 / 15 / 15 |
| Caltech-256 (Griffin, Holub, Perona) | https://data.caltech.edu/records/nyy15-4j048 (`256_ObjectCategories.tar`, 1,18 ГБ) | CC BY 4.0 | 30 607 изображений, 257 классов; взято не больше 60 на класс | `c256_photo`: 15 420 (10 740 / 2 340 / 2 340); `c256_obj`: 3 667 объектов с автомаской по однотонному фону (2 440 / 546 / 681) | по классам 179 / 39 / 39 |
| Oxford Flowers-102 (Nilsback, Zisserman) | https://www.robots.ox.ac.uk/~vgg/data/flowers/102/ (`102flowers.tgz`, `imagelabels.mat`) | явная лицензия не указана; набор распространяется для исследований | 8 189 | `flowers_photo`: 8 189 (5 477 / 1 495 / 1 217) | по классам 72 / 15 / 15 |
| DTD r1.0.1 (Cimpoi et al.) | https://www.robots.ox.ac.uk/~vgg/data/dtd/ (`dtd-r1.0.1.tar.gz`) | только для исследовательских целей (изображения с Flickr) | 5 640, 47 классов | `dtd_tex`: 5 640 (3 960 / 840 / 840) | по классам 33 / 7 / 7 |
| BSDS500 (Arbeláez et al.) | https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html (`BSR_bsds500.tgz`) | для некоммерческих исследований | 500 | `bsds_photo`: 1 500, по 3 случайных кропа на изображение (600 / 300 / 600) | официальный сплит train / val / test |
| Fashion-MNIST (Zalando Research) | https://github.com/zalandoresearch/fashion-mnist (файлы с официального S3) | MIT | 70 000 | `fmnist_obj`: 70 000 (60 000 / 5 000 / 5 000) | официальный train → train; официальный test пополам на val и test по id |

## Разделение train / val / test

- **Без утечек.** Сплит делается по источнику и до любой аугментации: по классу у Caltech, Flowers и DTD, по официальному сплиту или по id у BSDS500 и Fashion-MNIST. Ни один класс не попадает в два сплита; это проверяет `tests/test_data.py`.
- **Закреплённые классы.** Часть классов принудительно отнесена к train, остальные распределены детерминированно по хэшу (crc32).
  - Аэрокосмические и «ночные» классы: comet, fireworks, galaxy, lightning, mars, saturn, fighter-jet, blimp, hot-air-balloon, radio-telescope, airplanes, а также airplanes и helicopter из Caltech-101.
  - Несколько текстур DTD: smeared, marbled, veined, swirly.
  - Цель — чтобы сеть видела контент, похожий на ожидаемые картинки жюри («ракета в дыму»).
- **Шрифты.** Семейства шрифтов macOS разделены отдельно и тоже без пересечений (см. `doe/data.py`).

## Воспроизведение

```
.venv/bin/python -I scripts/prepare_external.py --workers 2   # кэш из архивов, около 40 с
.venv/bin/python scripts/make_datasets.py                     # val/test, демо, preview
```

Фиксированные наборы `data/val.npy` и `data/test.npy` (v2, по 256 четвёрок, сиды 1001 и 2002) состоят в основном из полутоновых изображений: доля картинок с числом уровней серого больше 32 — около 85% в val и 83% в test. Прежние наборы v1, где были в основном глифы и фигуры, лежат в `data/v1/`.
