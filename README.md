# xmx-mul-mat

Eigenständiges Entwicklungsprojekt für den XMX-Matrixkernel aus llama.cpp. Ziel ist, seine Voraussetzungen auf Arrow Lake-H / Arc 140T zu untersuchen und einen angepassten Pfad entwickeln zu können. Ein llama.cpp-Checkout, Modellgewichte und llama.cpp-Bibliotheken sind nicht nötig.

**Stand.** Auf Arrow Lake-H sind alle sechs Quantisierungsformate mit Zeilenloads und LSC-Gathers GPU-validiert. Die Tests decken beide GLU-Operationen, kurze Gewichtszeilen und SLM mit Gathers ab. Der Kernel teilt den 16-zeiligen B-Operanden in zwei DPAS-Operationen mit execution size 8. Der Treiber meldet für dieses Gerät keine native 2D-Block-I/O-Unterstützung. [Validierung und Build-Optionen](implementation.md#xe-lpg-functional-coverage)

## Dateien

| Datei | Inhalt |
| --- | --- |
| `include/xmx/kernel.hpp` | Eigentlicher Kernel mit Traits für Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, DPAS, 2D-Loads und Launchern |
| `include/xmx/compat.hpp` | Kleine Standalone-Hülle: Konstanten, Fehlermeldungen und Queue-Alias |
| `src/fixtures.cpp` | Deterministische gültige Quant-Blöcke, Reorder-Layouts und CPU-Matrixreferenz |
| `tests/kernel_tests.cpp` | GPU-Korrektheitsvergleich und optionales Kernel-Timing für Q4_K, Q6_K und Q8_0 |
| `tests/dpas_probe.cpp` | Einzelner DPAS-Test mit N=8 oder N=16, mit oder ohne Akkumulator; keine 2D-Loads und kein SLM |
| `tests/cpu_reference.cpp` | CPU-Tests für Half-Konvertierung, Layouts, 45 Referenzfälle und Fehlererkennung |
| `tests/history/` | Archiv des Engine-Resets und der früheren Compile-Probes |
| `UPSTREAM.json` | Herkunft, Commit und Prüfsummen |

## Voraussetzungen

Für CPU-Tests: Linux, CMake >= 3.20 und ein C++17-Compiler. Für die GPU-Programme zusätzlich Intel oneAPI DPC++/C++ (`icpx`) mit ESIMD/XMX-Headern und eine passende SYCL/Level-Zero-Laufzeit. Hier wurde oneAPI 2026.2 verwendet. Eine Paketinstallation oder ein Download gehört nicht zum Build.

## Bauen und normale Tests

```bash
cd ~/workspace/xmx-mul-mat
./build.sh
```

Das Skript lädt bei Bedarf `/opt/intel/oneapi/setvars.sh`, konfiguriert einen Release-Build unter `build/`, baut mit zwei Jobs und führt die Tests ohne GPU-Kernelstart aus. Alternativen können über `ONEAPI_SETVARS`, `--build-dir` und `--jobs` gewählt werden.

```bash
# Ganz ohne oneAPI/SYCL bauen:
./build.sh --cpu-only --build-dir build-cpu

# Optionale AOT-Kompilierung; diese startet ebenfalls keinen GPU-Kernel:
./build.sh --aot 12.74.4 --build-dir build-aot

# Tests des Standardbuilds wiederholen:
./run-tests.sh
```

Ein erfolgreicher Build bestätigt weder die GPU-Ausführung noch die Rechengenauigkeit des Kernels auf der Zielarchitektur.

## GPU-Tests bewusst starten

Zuerst andere LLM-Prozesse beenden. Bei der ersten Fehlermeldung oder einem Treiberreset abbrechen und die Ausgabe sowie `journalctl -k` sichern. Keine Endlosschleife über fehlgeschlagene Varianten starten.

```bash
source /opt/intel/oneapi/setvars.sh --force
export ONEAPI_DEVICE_SELECTOR=level_zero:0

# Zuerst eine einzelne, kleine DPAS-Variante:
./build/xmx_dpas_probe --run-gpu --n 8 --accumulator

# Weitere Varianten jeweils einzeln prüfen:
./build/xmx_dpas_probe --run-gpu --n 16 --accumulator
./build/xmx_dpas_probe --run-gpu --n 16

# Danach ein einzelner vollständiger Kernelfall:
./run-tests.sh --gpu --type q8_0 --rows 16 --cols 1 --k 1024

# Erst nach erfolgreichen Einzelfällen die 198 Fälle ausführen:
./run-tests.sh --gpu --suite
```

Der Launcher akzeptiert Arrow Lake-H und Alchemist für den DPAS-Pfad mit execution size 8. Bei `XMX_ENABLE_TRANSPOSED_2D_LOADS=ON` prüft er zusätzlich die Geometrie und die gemeldete native 2D-Block-I/O-Unterstützung.

Die Suite deckt Q2_K, Q3_K, Q4_K, Q5_K, Q6_K und Q8_0 ab, jeweils mit 16/17/128 Gewichtszeilen und 1/2/4/8/15/16/17/31/32/33/64 Aktivierungsspalten. Die Fälle mit 17 Zeilen oder 15/17/31/33 Spalten prüfen Teiltile. K beträgt standardmäßig 1024. Über `--k` sind Vielfache von 256 bis 32768 möglich. Die Eingaben sind synthetische, bereits quantisierte Daten. Die CPU-Referenz dequantisiert sie und berechnet die Matrixprodukte mit Double-Akkumulation. Verglichen werden relativer L2-Fehler und maximaler absoluter Fehler. NaN/Inf führen zum Fehlschlag.

Mit `--glu swiglu` oder `--glu geglu` prüft der Test die fusionierte Gate- und Up-Projektion mit getrennten Gewichten und gemeinsamen Aktivierungen. `--short-rows` testet die Short-Row-Traits aller Formate. Beide Optionen eignen sich auch für eine Suite. `--type` beschränkt eine Suite auf ein Format. Die Suite bricht beim ersten Fehler ab.

Fusionierte GLU-Kacheln decken bis zu acht Spalten ab. Der Softwarepfad ohne SLM bündelt bis zu 64 Spalten pro Submission. Der Kernel berechnet die Up- und Gate-Projektion nacheinander und nutzt für Q8_0 und Q4_K gepaarte 64-Byte-Aktivierungsloads. GLU-Suiten prüfen zusätzlich je Format einen Aufruf mit 65 Spalten. Der Arrow-Lake-AOT-Test prüft die fusionierten Softwarepfade auf null Scratch. [GLU-Benchmarks und Randfälle](implementation.md#fused-glu-register-spills)

### Regressionsprüfungen

Der CPU-Referenztest prüft zusätzlich gepackte Skalen, die sich je Zeile und K-Block unterscheiden. Der Host-Test `kernel_rejects_zero_k` prüft in CTest, dass der Launcher K=0 vor der Queue-Nutzung ablehnt. Er deckt einen Einspaltenaufruf und einen Aufruf mit neun Spalten und Short-Row-Traits ab.

```bash
./build/xmx_kernel_tests --check-zero-k
```

Die folgenden GPU-Einzelfälle prüfen den Dispatcher mit Short-Row-Traits und die Skalenadressierung über Zeilen-, Spalten- und K-Blockgrenzen. Für den Skalentest kann `--type` auch auf `q6_k` oder `q8_0` gesetzt werden.

```bash
./run-tests.sh --gpu --short-rows --type q8_0 --rows 16 --cols 9 --k 1024
./run-tests.sh --gpu --type q3_k --rows 17 --cols 17 --k 256
./run-tests.sh --gpu --type q3_k --glu swiglu --rows 17 --cols 17 --k 1024 --vary-scales
./run-tests.sh --gpu --vary-scales --type q4_k --rows 17 --cols 17 --k 1024
```

`--require-native-2d` setzt für den Q8_0-Vergleich gültige Geometrie, execution size 8 und native 2D-Block-I/O-Unterstützung voraus. Der Launcher lehnt Lunar Lake vor dem XMX-Kernelstart wegen der abweichenden DPAS-Ausführungsbreite ab.

## Optionales Timing

Nach einem bestandenen Korrektheitstest:

```bash
./run-tests.sh --gpu --type q8_0 --rows 128 --cols 4 --k 2048 --iterations 100
```

Ausgegeben werden JSON-Zeilen mit Fehlermaßen und durchschnittlichen Mikrosekunden pro Kernelstart. Das Timing enthält Queue-Submission und Synchronisation, aber keinen Upload, keine Eingabequantisierung und keinen Modelllauf. Es misst daher keine LLM-t/s und ist kein Vergleich mit dem normalen llama.cpp-Verifikationspfad. Es wird nur nach bestandener Korrektheitsprüfung ausgeführt.

### Profiling für Arrow Lake

Der Offline-Test `arrow_lake_isa` zerlegt das AOT-Binary und zählt Instruktionen pro K=256-Schritt. CTest bindet ihn für `12.74.4` mit LSC-Gathers, ohne SLM oder native 2D-Loads und mit Python 3.9+, `objcopy` und `ocloc` ein. Q8_0 und Q4_K verwenden für gepaarte Aktivierungsblöcke einen gemeinsamen 64-Byte-Load. Die Obergrenzen im unfusionierten 16-Spalten-Pfad liegen bei 80 Aktivierungsloads für Q8_0 und Q4_K, 144 für Q2_K, Q3_K, Q5_K und Q6_K sowie null Scratch-Bytes pro Thread. Q4_K und Q5_K teilen sich je Gruppe und Spalte eine Aktivierungssumme über alle 16 Gewichtszeilen. Q2_K berechnet zwei Summen mit einem DPAS und wendet die Minimumkoeffizienten je Zeile an. Das DPAS-Budget für Q2_K, Q4_K und Q5_K beträgt 384 Instruktionen bei 16 Spalten. Q8_0 lädt alle acht FP16-Skalen je Zeile mit einem gepackten Gather.

```bash
cmake -S . -B build-arl-profile -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=icpx \
    -DXMX_DEVICE_ARCH=12.74.4 -DXMX_ENABLE_GATHER_TRANSPOSE=ON
cmake --build build-arl-profile --target xmx_kernel_tests
ctest --test-dir build-arl-profile -R arrow_lake_isa -V
# Bereits extrahiertes AOT-Image, etwa aus einem Prototyp
python3 tests/arrow_lake_isa.py --zebin path/to/device.bin
```

Für GPU-Vergleiche jeweils denselben Fall mit Baseline und Kandidat ausführen. 16 Spalten prüfen gepaarte Ladezugriffe und die Minimumkorrekturen. Eine Spalte isoliert den Q8_0-Skalenpfad. Die Spaltenzahlen 32, 33, 47 und 48 vergleichen Arbeit an der Tile-Grenze. Der Software-Loadpfad ohne SLM berechnet das Ende mit bis zu zwei Tiles von jeweils 1 bis 8 Spalten. Die Tiles eines Blocks von bis zu 64 Spalten laufen in derselben Submission. Für 33 Spalten ergibt sich die Aufteilung 16 + 16 + 1. Der Offline-Bericht prüft die DPAS-Pfade für diese exakten Tile-Breiten. Der GPU-Test überwacht außerdem die reservierten Ausgabespalten hinter der gültigen Ausgabe.

```bash
./run-tests.sh --build-dir build-arl-profile --gpu --type q4_k --rows 1024 --cols 16 --k 4096 --vary-scales --iterations 64
./run-tests.sh --build-dir build-arl-profile --gpu --type q8_0 --rows 1024 --cols 1 --k 4096 --vary-scales --iterations 64
./run-tests.sh --build-dir build-arl-profile --gpu --type q8_0 --rows 1024 --cols 33 --k 4096 --vary-scales --iterations 64
```

## Weiterentwicklung

Der Arbeitsstand liegt in `include/xmx/kernel.hpp`. Die unveränderte Extraktion bleibt unter `third_party/llama.cpp/xmx-kernel.original.inc` erhalten. Herkunft prüfen:

```bash
python3 scripts/check_provenance.py
# Zusätzlich prüfen, dass der Kernelkörper noch unverändert ist:
python3 scripts/check_provenance.py --strict
```

Die Herkunft und Lizenz stehen in `THIRD_PARTY_NOTICES.md`; aktuelle Buildprüfungen in `VALIDATION.md`.

## Technische Referenzen

[references/](references/README.md) enthaelt lokale Dokumentationssnapshots fuer [Xe2](references/xe2/README.md) und [Xe-LPG+ / Arrow Lake-H](references/xe-lpg-plus/README.md): den kompletten ESIMD-Dokumentationsordner, DPAS-/vISA-Referenzen, installierte SDK-Header und getrennt eingeordnete Plattformquellen. Quellen, feste Git-Revisionen und Pruefsummen sind dokumentiert. Die Referenzen sind keine Build-Abhaengigkeiten.
