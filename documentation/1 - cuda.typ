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

== Ottimizzazioni ulteriori

A seguito delle precedenti implementazioni si è cercato di migliorare ulteriormente il kernel in modo da renderlo ancora più efficiente.\
Osservando il kenel della versione naive infatti ci siamo accorti nella presenza di molte istruzioni di selezione `if`, ciò comporta ad aumentare la warp divergence. La soluzione adottata nelle due versioni seguenti far eseguire un blocco di codice (inline per non avere overhead) che effettua lo swap dei valori delle variabili solo sotto una determinata condizione (non si è quindi cambiata la logica del kernel bensì solo la struttura), ma ciò non è valutato con if o istruzioni condizionali, bensì tramite operazioni bitwise che permettono di eseguire lo scambio solo se la condizione risulta vera.
```cpp
__device__ inline void ifSwap(bool condition, unsigned char *a, unsigned char *b)
{
    unsigned char mask = -condition; // 0xFF se true, 0x00 se false
    unsigned char temp = (*a ^ *b) & mask;
    *a = *a ^ temp;
    *b = *b ^ temp;
}
```

=== Branchless single thread

Per questa versione si è utilizzato un approccio più "GPU-friendly" in quanto si è ragionato che ogni thread avesse il compito di aggiornare una singola cella. Tuttavia i calcoli si sono resi più complessi dato che comunque era necessario avere una suddivione dell'immagine in blocchi 2x2 consistenti alla versione originale per avere un confronto equo e su dati identici.\
In questo caso, inoltre, non si poteva aggiornare la grigia 2x2 interamente, infatti bisognava calcolarsi la posizione di ciascun thread all'interno del quadrato, aumentando quindi i calcoli necessari.\
Data la precendente considerazione, si è deciso che la funzione (sempre inline per evitare overhead) che calcolava lo stato futuro dovesse restituire un valore che viene calcolato secondo la seguente operazione:
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
Si utlizzano sempre le istruzioni bitwise in modo da limitare il pìù possibile la warp divergence.

=== Branchless a blocchi

Questa versione, invece, si può ricondurre in tutto e per tutto alla versione base: ogni thread lavora su un blocco 2x2 di elementi, quindi a livello di logica pura non è cambiato niente.\
Infatti questa versione è la semplice ottimizzazione di quanto si fa nel kernel naive, in cui ogni blocco di 4 elementi viene valutato e questi ultimi, in caso fosse necessario vengono scambiati tramite le regole di movimento.\
A differenza della verione precedente la funzione che si occupa di calcolare lo stato futuro andrà a sostituire 4 elementi della matrice finale invece che uno solo.\
Ciò che è cambiato quindi è soltato il modo di valutare le espressioni e fare gli scambi fra i valori degli elementi: si passa da una versione in cui si facevano tutte istruzioni di selezione ad una versione in cui è tutto fatto tramite bitwise operations per diminuire la warp divergence.