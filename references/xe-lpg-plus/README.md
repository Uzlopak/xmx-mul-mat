# Xe-LPG+ / Arrow Lake-H: technische Referenzen

Lokale Referenzsammlung fuer die Entwicklung eines XMX-Kernelpfads auf unserer Intel Arc 140T. Zielgeraet: Core Ultra 7 265H, `intel_gpu_arl_h`, PCI-ID `8086:7d51`, Revision 03, IP `12.74.4`. **Arrow Lake-H ist nicht Xe2. Nicht alle Arrow-Lake- und Meteor-Lake-Varianten haben dieselben Matrixfunktionen.**

## Gemeinsame API- und Befehlsreferenzen

Diese Dokumente sind bytegleiche Kopien des in der Xe2-Sammlung festgehaltenen Stands. Sie sind architekturuebergreifend; die Unterstuetzung jeder Operation und Matrixform muss fuer das Zielgeraet separat geprueft werden.

- [ESIMD-Spezifikation](esimd/sycl_ext_intel_esimd.md): Ausfuehrungsmodell, Registervektoren, DPAS und SLM.
- [ESIMD-Funktionsreferenz](esimd/sycl_ext_intel_esimd_functions.md): 2D-Loads, Pitch, Ausrichtung, Gather/Scatter und Speicherregeln.
- [DPAS-Beispiel](esimd/examples/dpas.md): C++-Beispiel fuer Matrixoperanden und Packing.
- [vISA-DPAS](visa/DPAS.md), [LSC-Speicherbefehle](visa/LSC_UNTYPED.md), [Ausfuehrungsmodell](visa/3_execution_model.md) und [Operanden](visa/5_operands.md).
- [Installierte DPAS-Header](installed-oneapi-2026.2/esimd/xmx/dpas.hpp) und [Speicher-Header](installed-oneapi-2026.2/esimd/memory.hpp): exakter Snapshot unseres oneAPI 2026.2.

Der komplette ESIMD-Ordner mit 15 Dateien, Beispielen und Abbildungen ist enthalten. vISA ist eine virtuelle Compiler-ISA und nicht mit dem nativen Befehlssatz oder einem vollstaendigen Hardware-PRM gleichzusetzen.

## Arrow-Lake-spezifische Quellen

| Datei | Wofuer sie relevant ist |
| --- | --- |
| [Intels Architekturuebersicht als PDF](intel/arrow-lake-h-architecture.pdf) | Offizielle Arrow-Lake-H-Hardwareuebersicht mit XMX; kein ISA-Handbuch |
| [IGC Platform.hpp](compiler/Platform.hpp) | Feature-Abfragen des Intel-Compilers: `supportDpasInstruction`, `hasExecSize16DPAS`, `getGRFSize`, LSC und weitere Plattformgrenzen |
| [vISA-Plattformtabelle](compiler/platform-table.cpp) | Konkrete Plattformdaten fuer ARL, DG2, MTL, PVC und Xe2 |
| [PlatformInfo.h](compiler/PlatformInfo.h) | Bedeutung der Plattformdaten und Registergroessen |
| [BuildIR.h](compiler/BuildIR.h) | Compiler-interne IR-Schnittstellen; ergaenzende Implementierungsreferenz |
| [Compute-Runtime-Compilerhelfer](runtime/compiler-product-helper-arl.cpp) | Zuordnung von Device-ID und Revision zu ARL-H-Konfigurationen |
| [hw_info_arl.cpp](runtime/hw_info_arl.cpp) und [hw_info_arl.h](runtime/hw_info_arl.h) | Arrow-Lake-Hardwarebeschreibung in der Runtime |
| [hw_cmds_arl.h](runtime/hw_cmds_arl.h) | Verwendete Hardwarefamilie im Runtime-Code |

Diese Dateien sind relevante Implementierungsquellen, aber kein vollstaendiges natives Xe-LPG+-ISA-Handbuch.

## Konkrete Stellen fuer den Kernelport

1. In `compiler/platform-table.cpp` wird `Xe_ARL` mit 32 Byte pro GRF gefuehrt, `Xe2` mit 64 Byte. Dazu passt `getGRFSize()` in `compiler/Platform.hpp`. Vektor-Packing und Registerzugriffe aus dem Xe2-Kernel duerfen deshalb nicht ungeprueft uebernommen werden.
2. `hasExecSize16DPAS()` unterscheidet die native DPAS-Ausfuehrungsbreite nach Hardwarefamilie. `xmx::dpas<8, RC, int>` im Arbeitskernel enthaelt nicht alle Hardwareparameter explizit: Auch die Operandenlaengen und deren Layout bestimmen die Abbildung. Diese Abfrage ist ein Ausgangspunkt fuer die Untersuchung, kein Nachweis fuer eine fertige Anpassung.
3. `supportDpasInstruction()` schliesst Meteor Lake und bestimmte Arrow-Lake-Varianten ausdruecklich aus. Ein allgemeiner ARL-Name allein genuegt nicht als Feature-Nachweis.
4. `runtime/compiler-product-helper-arl.cpp` ordnet `0x7D51` mit Revision `0x3` der Konfiguration `ARL_H_B0` zu. Das passt zu unserem Geraet; die Datei unterscheidet auch andere ARL/MTL-Konfigurationen.
5. `load_2d` und die SLM-/Barrier-Logik bleiben gesondert zu pruefen. Ein erfolgreich kompilierter DPAS-Test ist kein Nachweis fuer den vollstaendigen Matrixkernel.

Die Kerneltests haben bisher nur einen fehlgeschlagenen Lauf des unveraenderten Xe2-Kernels auf der Arc 140T dokumentiert. Es gibt noch keinen bestaetigten Arrow-Lake-H-Port und keinen nachgewiesenen Geschwindigkeitsgewinn.

## Grenzen und Herkunft

Ein vollstaendiges oeffentliches natives Xe-LPG+-ISA-/PRM-Handbuch wurde bei dieser Recherche nicht identifiziert. Die gemeinsame API-Dokumentation, Compilerabfragen, Plattformdaten und Intels Architekturuebersicht werden deshalb getrennt eingeordnet. Die Hardwareunterstuetzung darf nicht pauschal aus einem ESIMD-Funktionsnamen abgeleitet werden.

[sources.json](sources.json) enthaelt feste Repository-Revisionen, Original-URLs, Dateigroessen und SHA256-Pruefsummen. Die Quellen sind Referenzkopien und keine Build-Abhaengigkeiten. Originale Copyright- und Lizenzhinweise bleiben erhalten. Gemeinsame Dokumente liegen bewusst auch hier lokal, damit diese Sammlung ohne Symlinks lesbar bleibt.
