# Referenzen zum Verstehen des XMX-Kernels

Diese Sammlung enthaelt lokale Kopien technischer Primaerquellen, Stand 18.11.2026. Die Dateien wurden fuer das Studium des Xe2-Kernels gesammelt. **Der Ordnername bedeutet nicht, dass jede Datei Xe2-spezifisch ist.**

## Ebenen und Lesereihenfolge

| Referenz | Zweck | Architekturbezug |
| --- | --- | --- |
| [ESIMD-Spezifikation](esimd/sycl_ext_intel_esimd.md) | Ausfuehrungsmodell, simd/simd_view, DPAS und SLM | Gemeinsame API fuer mehrere Intel-GPU-Generationen; einzelne Funktionen haben eigene Grenzen |
| [ESIMD-Funktionsreferenz](esimd/sycl_ext_intel_esimd_functions.md) | load_2d, gather/scatter, Blockzugriffe, Cache-Hints, Synchronisation | Gemeinsame API; konkrete Hardwareunterstuetzung separat pruefen |
| [DPAS-Beispiel](esimd/examples/dpas.md) | Operanden und Matrixmultiplikation im C++-Code | Lehrbeispiel, kein Nachweis fuer beliebige Shapes auf jeder GPU |
| [vISA-DPAS](visa/DPAS.md) | Semantik, Systolic Depth, Repeat Count, Operandenpraezision und Packing | Virtuelle Compiler-ISA, nicht das vollstaendige native Xe2-Binaerformat |
| [vISA-LSC](visa/LSC_UNTYPED.md) | Load/Store-Befehle einschliesslich 2D-Blockzugriffen | Compiler-ISA; Plattformgrenzen stehen bei den einzelnen Varianten |
| [vISA-Ausfuehrungsmodell](visa/3_execution_model.md) und [Operanden](visa/5_operands.md) | Register, Kanaele, Operanden und Ausfuehrung | Architekturuebergreifende Grundlagen |
| [Xe2-Modell im Intel Graphics Assembler](iga/ModelXe2.hpp) | Konkrete Xe2-Instruktionseigenschaften im Assembler | Xe2-spezifische Quellcodetabelle, kein erklaerendes Hardwarehandbuch |
| [Installierte DPAS-Header](installed-oneapi-2026.2/esimd/xmx/dpas.hpp) und [Speicher-Header](installed-oneapi-2026.2/esimd/memory.hpp) | Exakte API-Implementierung unseres oneAPI 2026.2 | SDK-Snapshot; neuere Online-Dokumentation kann abweichen |

Der ESIMD-Unterordner ist vollstaendig: alle 15 Dateien des ausgewaehlten Git-Trees, einschliesslich Beispielen und SVG-Abbildungen. Relative Verweise innerhalb dieses Ordners bleiben erhalten. Verweise auf andere Teile des Intel-LLVM-Repositories sind nicht automatisch mitgespiegelt.

## Bezug zum Kernel

Im [Arbeitskernel](../../include/xmx/kernel.hpp):

- `xmx_term` ruft `xmx::dpas<8, RC, int>(B, A)` auf. Die 8 ist die Systolic Depth, RC der Repeat Count, int der Ergebnisdatentyp. Die konkreten Operandenlaengen bestimmen weitere Matrixdimensionen. Das Packing ist Teil des Vertrags.
- `xmx_load_t` und `xmx_mul_mat` verwenden `load_2d`, teilweise transponiert. Breite, Hoehe, Pitch, Koordinaten, Ausrichtung und das resultierende Registerlayout muessen gemeinsam verstanden werden.
- `xmx_grf` und die Launch-Properties waehlen die Registerbelegung.
- `slm_init`, SLM-Loads/Stores und `barrier` kombinieren die Teilresultate mehrerer Threads.
- `simd<T,N>` ist ein expliziter Registervektor. N ist nicht pauschal die SYCL-Subgroup-Groesse.

## Was hier nicht vorliegt

Ein vollstaendiges oeffentliches natives Xe2-ISA-/PRM-Handbuch wurde bei dieser Recherche nicht identifiziert. vISA-Dokumentation, ESIMD-Spezifikation und Assemblermodell sind verschiedene Ebenen und duerfen nicht als dasselbe Handbuch behandelt werden. Intels aktuell gelistete PRMs stehen im [offiziellen Index](https://www.intel.com/content/www/us/en/docs/graphics-for-linux/developer-reference/1-0/overview.html).

Die direkte lokale Kopie dieses Index und der Download des [oneAPI GPU Optimization Guide 2025.2](https://cdrdv2.intel.com/v1/dl/getContent/859422?fileName=oneapi_optimization-guide-gpu_2025.2-771772-859422.pdf) wurden vom Intel-Server mit HTTP 403 abgewiesen. Diese beiden Ressourcen sind deshalb nur verlinkt, nicht als vollstaendige lokale Dokumente vorhanden.

## Herkunft und Integritaet

[sources.json](sources.json) nennt Original-URLs, feste Git-Commits, Dateigroessen und SHA256-Pruefsummen. Die GitHub-Dateien wurden zusaetzlich gegen die Git-Blob-Hashes bzw. erwarteten Groessen geprueft. [esimd-tree.json](esimd-tree.json) enthaelt die vollstaendige Dateiliste des ESIMD-Snapshots. Originale Lizenz- und Copyright-Hinweise bleiben in den kopierten Dateien erhalten. Die Dateien sind Referenzmaterial, keine Build-Abhaengigkeiten dieses Projekts.
