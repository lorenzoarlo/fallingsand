// codly
#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#show: codly-init.with()

#codly(
  languages: codly-languages,
  header-repeat: true,
  zebra-fill: luma(248),
  lang-stroke: lang => rgb(0, 0, 0, 0),
  smart-indent: true,
  number-format: none,
)

= Implementazione Parallela CUDA 

Comne visto nella parte di analisi del problema l'algoritmo particolarmente parallelizzabile per leì'esecuzione di ogni frame in quanto i dati iniziali sono presenti in una matrice. L'unica dipendenza fra i dati si trova all'interno di un blocco di 2x2 in cui gli scambi devono essere consistenti.

== Naive

La prima implementazione svolta è stata quella naive in cui si è cercato di mantenere una certa leggibilità del codice ed equivalenza rispetto al codice sequenziale.\
Le variazioni rispetto alla versione sequenziale stanno nella parte di lancio del thread (cambia la funzione next). Il ruolo della funzione difatti è quello di lanciare i kernel, dimensionando griglia e blocchi. Per questa prima versione si è decisa una dimensione di blocchi di 32x32. Dal momento che questa prima versione è del tutto similare alla versione sequenziale, ogni kernel dovrà gestire blocchi di 2x2 elementi della matrice iniziale, quindi la griglia si è decisa tramite la formula:
```cpp
dim3 grid((u_in->width / 2 + block.x - 1) / block.x,
          (u_in->height /2 + block.y - 1) / block.y);
```
Inoltre, per rispettare totalmente la logica di gestione della memoria fatta per il caso sequenziale, ad ogni invocazione della funzione viene copiata in memoria del device la matrice di input e copiata in memoria dell'host la matrice di output. 

== Optimized

Questa versione mantiene la struttura della versione precedentemente descritta. Tuttavia, ciò che è cambiato principalmente è:
- le chiamate a funzione: nella precedente versione si utilizzavano chiamate a funzione senza curarsi di possibili tempistiche dovute all'overhead delle chiamate a funzione. In questa versione abbiamo tolto tutte le chiamate a funzione.
- le istruzioni condizione: nella versione naive si utilizzavano istruzioni di if in continuazione. In questa versione si è cercato di limitarle per aumentare la branch prediction.

