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

= Ottimizzazione CUDA

Comne visto nella parte di analisi del problema, l'algoritmo risultera essere particolarmente parallelizzabile nell'esecuzione di ogni frame, in quanto i dati iniziali sono presenti in una matrice. L'unica dipendenza fra i dati si trova all'interno di ogni blocco $2 times 2$, in cui gli scambi devono essere consistenti.

== Implementazione naive
La prima implementazione è stata quella _naive_, in cui si è cercato di mantenere una certa leggibilità del codice ed equivalenza rispetto alla versione sequenziale.

Le variazioni rispetto alla versione sequenziale si trovano ovviamente nella parte di \"lancio\" del kernel (cambia la funzione `next`).

Per questa prima versione, a seguito dell'analisi di diverse alternative, è stata scelta una dimensione di blocchi di $32 times 32$.

Dato che questa prima versione è analoga alla versione sequenziale, ogni kernel dovrà gestire blocchi di $2 times 2$ elementi della matrice iniziale: da questo deriva il calcolo della dimensione della griglia:
```cpp
dim3 grid((u_in->width / 2 + block.x - 1) / block.x,
          (u_in->height /2 + block.y - 1) / block.y);
```
Inoltre, per rispettare totalmente la logica di gestione della memoria fatta del caso sequenziale, ad ogni invocazione della funzione è copiata in memoria del _device_ la matrice di input e copiata in memoria dell'host la matrice di output.

== Implementazione ottimizzata (_optimized_)
Questa versione mantiene la struttura della versione precedentemente descritta. Si sono effettuate piccole ottimizzazioni a seguito del profiling. Tra le modifiche effettuate:
- si sono eliminate le chiamate a funzione, in modo da diminuire l'overhead;
- sono state diminuite le strutture di selezione per aumentare la _branch prediction_;
- le istruzioni sono state svolte su variabili valore: nella versione base si effettuava continuamente la deferenziazione dei puntatori per ottenere il valore. In questa ottimizzazione sono state usate variabili locali al kernel per la gestione delle celle. Leggendo all'inizio del kernel i valori, si evitano i continui accessi alla matrice `in`.

== Versioni _branchless_
A seguito delle precedenti implementazioni, si è cercato di migliorare ulteriormente il kernel in modo da renderlo ancora più efficiente.

Osservando la versione naive, è stata evidente la presenza di molte istruzioni di selezione `if`, che aumentano la warp divergence.

La soluzione adottata nelle successive versioni è stata quella di eseguire in ogni caso le istruzioni (attraverso funzioni inline per evitare l'overhead) per effettuare lo scambio dei valori delle variabili solo ad una condizione, evitando strutture condizionali in favore di operazioni bitwise che effettuano lo scambio solo se la condizione risulta vera
```cpp
__device__ inline void ifSwap(bool condition, unsigned char *a, unsigned char *b){
    unsigned char mask = -condition; // 0xFF se true, 0x00 se false
    unsigned char temp = (*a ^ *b) & mask;
    *a = *a ^ temp;
    *b = *b ^ temp;
}
```
=== Ottimizzazione block branchless
Questa ottimizazione riporta esattamente la stessa logica della versione precedente, ma con l'utilizzo di istruzioni bitwise per effettuare lo scambio dei valori invece di strutture condizionali.

=== Branchless single thread

Per questa versione si è utilizzato un approccio più _GPU-friendly_, per cui ogni thread avesse il compito di aggiornare la singola cella ad esso assegnata.

Tuttavia ciò ha portato ad un aumento dei calcoli necessari a mantenere la consistenza all'interno dei blocchi $2 times 2$.

Oltre a questo, in questa ottimizzazione non è possibile aggiornare la grigia $2 times 2$ interamente, portando alla decisione di creare la funzione (sempre _inline_ per evitare overhead) che restituisce in ogni caso il valore futuro assunto dalla cella. Questa \"discriminazione\" è fatta attraverso il seguente codice che ricorda la logica di una lookup table hardware
```cpp
int mask_x = -x;
int not_mask_x = ~mask_x;
int mask_y = -y;
int not_mask_y = ~mask_y;
return (topleft & not_mask_x & not_mask_y) |
       (topright & mask_x & not_mask_y) |
       (bottomleft & not_mask_x & mask_y) |
       (bottomright & mask_x & mask_y);
```
Oltre a ciò, per questa versione è stato ritenuto utile sfruttare la _shared memory_ messa a disposizione in quanto si adattava particolarmente alla natura del problema.
```cpp
 __shared__ unsigned char s_tile[TILE_DIM_Y][TILE_DIM_X];
```
Il vantaggio ottenuto da tale implementazione riguarda il fatto che ogni thread per calcolare il suo stato futuro deve accedere a dati che si trovano nel suo intorno, in modo da diminuire gli accessi in memoria globale.

Questo approccio porta ovviamente ad un aumento di thread necessari, esattamente il quadruplo.

== Confronto delle prestazioni

Dopo la fase di implementazione si è proceduto con la valutazione delle prestazioni ed il profiling tramite NVIDIA Nsight Compute.

A seguito di una prima analisi delle prestazioni si è notato che le prestazioni delle varie versioni non variavano di tanto fra l'una e l'altra, portando alla scelta di rendere le versioni branchless (considerate le più promettenti) _stateful_. In questo modo, la matrice è caricata dall'host solo la prima volta e utilizzare quella aggiornata direttamente dallo stato precedente.

Per essere consistenti con il resto del programma e mantenendo il fatto che ad ogni iterazione il kernel deve analizzare solo il singolo frame, ad ogni iterazione è necessario copiare la matrice risultante sull'host.

#{
  show table: zero.format-table(auto, auto, auto)
  figure(
    table(
      columns: 3,
      table.header([*Versione*], [*Numero cicli*], [*Speedup*]),
      [Base (sequenziale)], [1312600611], [$times 1$],
      [Naive], [987677573], [$times 1.33$],
      [Ottimizzata], [906587877], [$times 1.45$],
      [Branchless a blocchi], [397147967], [$times 3.31$],
      [Branchless single thread], [493391021], [$times 2.66$],
    ),
    caption: [Risultati ottimizzazione CUDA del sample 1 ($400 times 400$) per $1500$ frame],
  )
  figure(
    image("assets/graph-cuda-sample-1.png", width: 100%),
    caption: [Grafico dei risultati per le ottimizzazioni CUDA del sample 1],
  )

  figure(
    table(
      columns: 3,
      table.header([*Versione*], [*Numero cicli*], [*Speedup*]),
      [Base (sequenziale)], [1446900357], [$times 1$],
      [Naive], [996944612], [$times 1.45$],
      [Ottimizzata], [970225327], [$times 1.5$],
      [Branchless a blocchi], [401415193], [$times 3.6$],
      [Branchless single thread], [503544295], [$times 2.87$],
    ),
    caption: [Risultati ottimizzazione CUDA del sample 2 ($400 times 400$) per $1500$ frame],
  )
  figure(
    image("assets/graph-cuda-sample-2.png", width: 100%),
    caption: [Grafico dei risultati per le ottimizzazioni CUDA del sample 2],
  )

  figure(
    table(
      columns: 3,
      table.header([*Versione*], [*Numero cicli*], [*Speedup*]),
      [Base (sequenziale)], [31522709229], [$times 1$],
      [Naive], [13340281229], [$times 2.36$],
      [Ottimizzata], [13100873649], [$times 2.4$],
      [Branchless a blocchi], [4833328935], [$times 6.5$],
      [Branchless single thread], [7440696748], [$times 4.23$],
    ),
    caption: [Risultati ottimizzazione CUDA del sample 3 ($1920 times 1080$) per $3000$ frame],
  )
  figure(
    image("assets/graph-cuda-sample-3.png", width: 100%),
    caption: [Grafico dei risultati per le ottimizzazioni CUDA del sample 3],
  )
}

=== Profiling
Il profiling è stato fatto tramite NVIDIA Nsight Compute.

Prima di tutto, è importante notare che il kernel sviluppato svolge tante operazioni bitwise, shift e calcolo di indirizzi. Le uniche operazioni floating point effettuate riguardano l'operazione floating point per la generazione di numeri random, che tuttavia non influisce sull'utilizzo della bandwidth e non è modificabile per tener fede all'algoritmo originale. Per questo motivo, si è deciso di non effettuare l'analisi del _roofline model_.

Altra osservazione importante riguarda il fatto l'algoritmo è fortemente memory bound e che le ultime ottimizzazioni cercano di ovviare a tale problema utilizzando la shared memory.

Attraverso la funzione `ifSwap` (descritta in precedenza) si svolge lo scambio dei valori solo se la condizione è vera, permettendo di diminuire la _warp divergence_ e a rendere coerente tra i thread la compute capability, riuscendo a mantenere l'_occupancy_ in un range ottimale.

=== Versione naive e ottimizzata

Tale valore è soddisfacente già nella versione naive (intorno al $75\%$): nonostante ciò all'aumentare dei dati il tempo di esecuzione risulta essere peggiore rispetto a quello delle altre versioni ottimizzate in parallelo, seppur più veloce rispetto alla versione sequenziale. Contribuisce a ciò anche l'hit rate delle memorie cache, pari a $54 \%$ per la cache L1 e $61\%$ per la cache L2.

La versione leggermente  ottimizzata mostra prestazioni lievemente migliori ma comunque molto simili alla versione naive.


=== Versioni branchless

Dati più interessanti sono quelli delle implementazioni branchless.


L'implementazione del kernel _block branchless_  risulta essere la più veloce, ottenendo uno speedup di circa 6 volte e mezzo.
Presenta un'_occupancy_ leggermente più alta rispetto alla versione naive (pari circa all'80%), valori di _cache hit_ maggiori pari al $58\%$ per L1 e picchi del $74\%$ (nei diversi frame).

Considerando una media di $25$ thread attivi sui $32$ massimi per warp, si è deciso tenendo in considerazione i risultati del profiling, di diminuire la dimensione dei blocchi da $32 times 32$ a $16 times 16$ per mantenere un numero di thread attivi più alto possibile. Fisiologicamente diminuisce anche l'_occupancy_ arrivando ad un valore pari a $77/78\%$ ma portando anche ad un miglioramento dei tempi di esecuzione. Variando la dimensione del blocco, si ottiene anche un miglioramento del memory throughput, che si stabilizza intorno ai $50 "Gbyte/sec"$ rispetto ai $30/40 "Gbyte/sec"$ precedenti..


L'implementazione del kernel _single branchless_ (in cui ogni thread aggiorna una singola cella) prevedeva inizialmente l'accesso alla memoria globale, portnado ad un tempo di esecuzione spaventosamente alto (caratterizzato da un'occupancy elevata pari ad oltre il $90\%$ e ad un aumento della cache hit). Utilizzando la shared memory ha portato ad un miglioramento significativo, anche se comunque inferiore rispetto alla versione _block branchless_.
Anche in questo caso, si è deciso di diminuire la dimensione dei blocchi da $32 times 32$ a $16 times 16$ ottenendo miglioramenti simili.

