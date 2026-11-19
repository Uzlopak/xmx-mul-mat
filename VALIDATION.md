# Validierung

Die aktuellen GPU-Ergebnisse für alle sechs Formate, kurze Gewichtszeilen, die fusionierten GLU-Pfade und die optionale SLM-Reduktion stehen im [Implementierungsbericht](implementation.md#xe-lpg-functional-coverage).

## Historische Buildprüfung zum Bring-up

Auf diesem Linux-System erfolgreich geprüft:

| Variante | Befehl | Ergebnis |
| --- | --- | --- |
| CPU-only | `./build.sh --cpu-only --build-dir build-cpu` | Build und CPU-Referenztest bestanden |
| SYCL/JIT | `./build.sh` | Alle drei Programme gebaut; drei Tests ohne GPU-Kernelstart bestanden |
| SYCL/AOT | `./build.sh --aot 12.74.4 --build-dir build-aot` | Kompilierung für Arrow Lake-H und drei Tests ohne GPU-Kernelstart bestanden |
| Frische Shell | `./run-tests.sh` und `ctest --test-dir build --output-on-failure` | Tests bestanden, SYCL-Runtime wird gefunden |
| Herkunft | `python3 scripts/check_provenance.py` | Originalprüfsumme stimmt; Arbeitskernel als Entwicklungsport erkannt |
| Shell-Syntax | `bash -n build.sh run-tests.sh` | Bestanden |

SYCL-Compiler: Intel oneAPI 2026.2. Der CPU-only-Build verwendet den normalen System-C++-Compiler. Die CPU-Prüfung deckt Half-Konvertierungen, Quant-Layouts und 45 synthetische Referenzfälle ab. Die zwei anderen Standardtests prüfen lediglich die Hilfe der GPU-Programme; sie sind keine GPU-Rechentests.

Die Programme linken keine llama.cpp-/ggml-Bibliotheken und die aktiven Quelldateien benötigen keinen bestehenden llama.cpp-Checkout. Nur die historischen Dateien und Herkunftsangaben enthalten Verweise auf den früheren Versuchsaufbau.

**GPU-Ausführung:** Der execution-size-8-DPAS-Probe bestand. Die native 16er-Variante lieferte ab dem zweiten 8er-Block falsche Werte. Der anschließende kleine Q8_0-Kernelfall endete mit `UR_RESULT_ERROR_DEVICE_LOST`; der xe-Treiber meldete einen CCS-Engine-Reset und erzeugte einen Coredump. Keine weiteren Varianten wurden danach gestartet. Die Details liegen unter `tests/history/xe-lpg-port-attempt-2026-11-18.md`. Ein erfolgreicher SYCL- oder AOT-Build und ein bestandener DPAS-Test sind deshalb kein Nachweis für vollständige Laufzeitkompatibilität, Korrektheit oder Geschwindigkeit.

Der aktuelle Quellstand verwendet für den nächsten Bring-up-Schritt standardmäßig `XMX_ENABLE_SLM=OFF` und `XMX_ENABLE_TRANSPOSED_2D_LOADS=OFF`. Der no-SLM-Pfad wurde nach der Geräteerholung mit demselben kleinen Q8_0-Fall gestartet und löste ebenfalls einen CCS-Reset aus; SLM-/Barrier-Operationen sind damit nicht die alleinige Ursache. Der nächste statische Bring-up-Schritt umgeht zusätzlich transponierte 2D-Gewichts- und Skalenloads. Für explizite Experimente können beim CMake-Konfigurieren `-DXMX_ENABLE_SLM=ON` oder `-DXMX_ENABLE_TRANSPOSED_2D_LOADS=ON` gesetzt werden. Danach wurden keine weiteren GPU-Varianten gestartet.

Der normale llama.cpp-Integrationsbranch, seine Binärdateien und die Modellstarter wurden durch die Projekterstellung nicht verändert.
