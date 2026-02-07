#import "@preview/zero:0.6.1"

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


= Ottimizzazioni SIMD
Di seguito sono descritte alcune ottimizzazioni che utilizzano istruzioni SIMD (_Single Instruction, Multiple Data_) per migliorare le prestazioni dell'implementazione base.

Per farlo è stata utilizzata la libreria Google Highway @highway_library, che fornisce un'interfaccia portabile per l'utilizzo di istruzioni SIMD su diverse architetture.

L'idea alla base è che sia possibile elaborare più celle contemporaneamente caricando nei registri vettori che contengono i valori di più celle (e quindi di diversi blocchi).

Oltre ad adattare l'algoritmo per processare più blocchi contemporaneamente, è stato fatto un passo in più per rendere _branchless_ l'algoritmo, ovvero evitare di utilizzare istruzioni di salto per gestire gli scambi. In alternativa sono state impiegate le seguenti istruzioni che permettono di eseguire lo scambio
```cpp
template <class Descriptor, class Vector, class Mask>
HWY_INLINE void IfSwap(Descriptor d, Mask condition_mask, Vector &a, Vector &b)
{
    auto new_a = hn::IfThenElse(condition_mask, b, a);
    auto new_b = hn::IfThenElse(condition_mask, a, b);
    a = new_a;
    b = new_b;
}
```
Per determinare ogni condizione, si è utilizzata una combinazione di istruzioni logiche per calcolare le maschere di condizione, poi fornite alla funzione `IfSwap` per scambiare i corretti \"componenti\" del vettore caricato.

Per effettuare le trasformazioni, è stato necessario adattare ogni istruzione `if-else-if` ponendo attenzione a rispettare l'effettivo comportamento dell'algoritmo originale (ad esempio, per calcolare lo scambio che avviene nel ramo `else-if` è necessario negare la condizione del ramo `if` precedente e unirlo con la condizione del ramo `else-if`).

A causa delle dipendenze tra i dati all'interno di ogni \"blocco\", si ha che all'interno di ogni vettore sarebbe necessario effettuare confronti tra componenti adiacenti. Per farlo è quindi necessario effettuare delle permutazioni sui vettori caricati.

Sono state implementate diverse versioni dell'algoritmo, con leggere differenze.


Una versione \"fallimentare\"#footnote[di cui non sono presentati i risultati] è stata quella di utilizzare come istruzione di caricamento la funzione
```cpp
hn::LoadInterleaved2(d, toprow_address + x, toplefts, toprights)
```
che permette di caricare in due vettori distinti tutti i dati in posizione pari (nel primo) e in posizione dispari (nel secondo)   a partire da un indirizzo.

Tuttavia, nel nostro caso non si è rivelata una buona scelta, in quanto nei diversi _sample_ presenterebbe performance peggiori rispetto alla versione base di riferimento.
Il motivo di tale inefficienza è dovuto al fatto che per la struttura dei dati, sono presenti diversi blocchi composti da celle uguali che quindi non sono processati. Tale ottimizzazione è riportata anche in questa versione \"fallimentare\" ma non è sufficiente a compensare l'overhead introdotto dallo smistamento dei dati.

La soluzione#footnote[consultabile nel file `src/simd/simd-manual-interleave.cpp`] è stata quella di utilizzare per prima la funzione di caricamento `hn::LoadU` (che effettua il caricamento anche di dati non allineati, per rendere il codice compatibile con il programma preesistente#footnote[scelta progettuale per evitare di rendere dipendente da Google Highway il codice principale]), per poi verificare se tutti i blocchi caricati sono composti da celle uguali tra loro#footnote[si intende che le celle di ogni blocco devono essere uguali tra loro all'interno del blocco, non è necessario che anche i blocchi siano uguali tra loro], e in caso affermativo non processarli. In caso contrario, si procede con lo smistamento dei dati attraverso le funzioni `hn::ConcatEven` e `hn::ConcatOdd`.

Da notare che in entrambe le versioni ad ogni iterazione si processano esattamente il doppio di celle rispetto a quelle inseribili in un vettore (indicato dalla variabile `lanes` che è calcolata dalla libreria in base alla dimensione dei dati e all'architettura), in quanto i confronti vengono fatti tra i componenti adiacenti che devono essere separati, spostando i dati confrontabili in due vettori distinti.

L'ultima ottimizzazione implementata prevede l'utilizzo di istruzioni di _prefetch_#footnote[consultabile nel file `src/simd/simd-manual-interleave-prefetch.cpp`] per caricare in anticipo i dati che saranno processati nei successivi passi dell'algoritmo.


== Confronto delle prestazioni
Di seguito sono riportati i risultati delle esecuzioni dei diversi algoritmi sui vari _sample_. Tutti i test sono stati eseguiti su un computer MacBook Pro con processore Apple M2 e $8 "GB"$ di RAM e i risultati sono la media di tre esecuzioni.

Oltre alle versioni ottimizzate manualmente, sono stati inseriti anche i risultati delle versioni ottimizzate dal compilatore attraverso l'utilizzo del flag `-O3`.


#{
  show table: zero.format-table(auto, auto, auto)

  figure(
    table(
      columns: 3,
      table.header([*Versione*], [*Numero cicli*], [*Speedup *]),
      [Base], [17633412], [$times 1$],
      [Base con ottimizzazione `-O3`], [4935159], [$times 3.57$],
      [Ottimizzazione SIMD senza prefetch], [2511367], [$times 7.02$],
      [Ottimizzazione SIMD con prefetch], [2520964], [$times 6.99$],
    ),
    caption: [Risultati ottimizzazione SIMD del sample 1 ($400 times 400$) per $1500$ frame],
  )
  figure(
    image("assets/graph-simd-sample-1.png", width: 100%),
    caption: [Grafico dei risultati per le ottimizzazioni SIMD del sample 1],
  )


  figure(
    table(
      columns: 3,
      table.header([*Versione*], [*Numero cicli*], [*Speedup *]),
      [Base], [18760698], [$times 1$],
      [Base con ottimizzazione `-O3`], [5640478], [$times 3.33$],
      [Ottimizzazione SIMD senza prefetch], [2718206], [$times 6.9$],
      [Ottimizzazione SIMD con prefetch], [2858030], [$times 6.56$],
    ),
    caption: [Risultati ottimizzazione SIMD del sample 2 ($400 times 400$) per $1500$ frame],
  )
  figure(
    image("assets/graph-simd-sample-2.png", width: 100%),
    caption: [Grafico dei risultati per le ottimizzazioni SIMD del sample 2],
  )


  figure(
    table(
      columns: 3,
      table.header([*Versione*], [*Numero cicli*], [*Speedup *]),
      [Base], [392756058], [$times 1$],
      [Base con ottimizzazione `-O3`], [111089334], [$times 3.53$],
      [Ottimizzazione SIMD senza prefetch], [53052551], [$times 7.4$],
      [Ottimizzazione SIMD con prefetch], [53404369], [$times 7.35$],
    ),
    caption: [Risultati ottimizzazione SIMD del sample 3 ($1920 times 1080$) per $3000$ frame],
  )

  figure(
    image("assets/graph-simd-sample-3.png", width: 100%),
    caption: [Grafico dei risultati per le ottimizzazioni SIMD del sample 3],
  )

  figure(
    table(
      columns: 3,
      table.header([*Versione*], [*Numero cicli*], [*Speedup *]),
      [Base], [507135947], [$times 1$],
      [Base con ottimizzazione `-O3`], [510808022], [$times 0.99$],
      [Ottimizzazione SIMD senza prefetch], [206792052], [$times 2.45$],
      [Ottimizzazione SIMD con prefetch], [209215278], [$times 2.42$],
    ),
    caption: [Risultati ottimizzazione SIMD del sample 4 ($3840 times 2160$) per $3000$ frame],
  )

  figure(
    image("assets/graph-simd-sample-4.png", width: 100%),
    caption: [Grafico dei risultati per le ottimizzazioni SIMD del sample 4],
  )
}


Come è possibile notare dai risultati, le ottimizzazioni SIMD implementate permettono di ottenere un miglioramento significativo delle prestazioni rispetto alla versione base, con uno speedup di circa $7$ volte (leggermente migliore per il _sample 3_, dovuto probabilmente al maggior numero di frame coinvolti, e di molto peggiore per il _sample 4_), anche superiore rispetto all'ottimizzazione fatta automaticamente dal compilatore.
I risultati sono coerenti per tutti i _sample_ ed è possibile notare che l'ottimizzazione con prefetch peggiora generalmente le prestazioni (probabilmente dovuto all'overhead introdotto dalle istruzioni che non compensa il vantaggio di avere i dati già caricati automaticamente).

Il miglioramento è dovuto alla combinazione del vantaggio ottenuto dalla versione _branchless_ e dalla maggiore quantità di dati processati contemporaneamente offerta da SIMD.
