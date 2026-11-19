# Herkunft und Lizenz

Der Kernel stammt aus [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), ursprünglich [PR #29864](https://github.com/ggml-org/llama.cpp/pull/29864), gemergt am 22. Oktober 2026. Er wurde aus dem lokalen Integrationsstand `3bfc80f3da00ea10e8182ce707d4e3c22597a038` extrahiert. Das ist kein neuer, bereits für Arrow Lake-H angepasster Kernel.

- `include/xmx/kernel.hpp`: unveränderter Körper des XMX-Abschnitts aus `ggml/src/ggml-sycl/esimd.hpp`, mit eigenständigen Includes und Namespace-Hülle.
- `third_party/llama.cpp/xmx-kernel.original.inc`: unveränderte Kopie dieses Abschnitts als Vergleichsbasis.
- `third_party/llama.cpp/dequant-reference.inc`: die vier in `UPSTREAM.json` genannten CPU-Referenzfunktionen aus `ggml/src/ggml-quants.c`.

Diese Teile stehen unter der MIT-Lizenz, Copyright (c) 2023-2026 The ggml authors. Der vollständige Lizenztext liegt in `LICENSE`. Die ursprüngliche `esimd.hpp` hat keinen eigenen abweichenden Lizenzkopf; die Repository-Lizenz gilt weiter. Die Standalone-Hülle und Tests werden ebenfalls unter MIT bereitgestellt.

Die notwendigen Blockstrukturen und Konstanten werden lokal bereitgestellt. `dpct::queue_ptr` ist hier lediglich ein Alias für `sycl::queue*`; das Projekt benötigt weder das DPCT-Paket noch llama.cpp-Bibliotheken.

Die Dateien unter `tests/history` dokumentieren die bisherigen lokalen Versuche. Sie werden nicht kompiliert. Insbesondere der historische Testcode benötigt noch die damaligen llama.cpp-Header und ist ausschließlich ein Archiv des Fehlerversuchs.
