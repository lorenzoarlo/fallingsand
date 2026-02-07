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

=== Profiling

Il profiling è stato fatto tramite NVIDIA Nsight Compute.\
È importante notare che il kernel sviluppato sfrutta tante operazioni bitwise, shift e calcolo di indirizzi, l'unica operazione floating point che viene sfruttata dal kernel è la generazione del random, che però non influisce granchè (rispetto all'utilizzo della bandwith). Perciò si è deciso di non effettuare l'analisi del roofline model.\
Tutte le versioni del kernel sono fortemente memory bound, ma nelle ultime ottimizzazioni ciò viene mascherato tramite l'utilizzo di shared memory ed operazioni bitwise per il confronto.\
Difatti la funzione `ifSwap` descritta in precendenza permette di eseguire in ogni caso le operazioni che la compongono però, se si verifica la condizione fa lo switch dei valori, se non si verifica i valori rimangono originali, esattamente come un if. Così facendo diminuisce la warp divergence ed aumenta anche leggermente la compute capability che contribuisce a mantenere l'occupancy in un range ottimale.\
Già dalla versione naive del kernel l'occupancy è buona (intorno al 75%) tuttavia, da quanto appare dai risultati ottenuti e raffigurati nel paragrafo precedente, su grandi moli di dati il tempo di esecuzione è il peggiore fra i vari kernel, seppur più veloce rispetto al sequenziale. Complice di questo è anche l'hit rate delle memorie cache che risulta essere:
- per la cache L1 del 54%
- per la cache L2 del 61%
\
La versione leggermente più ottimizzata ha prestazioni lievemente migliori ma comunque molto simili alla versione naive.\
I dati interessanti sono quelli delle ultime due implementazioni.\
L'implementazione del kernel branchless a blocchi risulta la più veloce, andando circa 6 volte e mezzo più veloce della versione base. Presenta un'occupancy leggermente più alta rispetto alla versione naive (circa dell'80%), un cache hit cresce per la L1 circa intorno al 58% mentre per la L2 arriva a picchi di circa il 74%. Mantenendo gli warp attivi per sm circa intorno ai 25 sui 32 teoricamente raggiungibili. Inizialmente la dimensione dei blocchi era di 32x32 ed il tempo di esecuzione era leggermente superiore alla versione presentata, facendo profiling tramite Nsight Compute, si è notato che abbassando la dimensione del blocco a 16x16 il tempo di esecuzione diminuiva. A diminuire tuttavia era anche l'occupancy che però rimane sempre intorno ai 77-78%, ma dato che l'obiettivo è ridurre il tempo di esecuzione del kernel, una variazione negativa del 2-3% dell'occupancy non è un problema. Inoltre variando la dimensione del blocco (32x32 a 16x16) si è passati da avere un memory throughput che variava fra i 30 e i 40 Gbyte/sec a stabilirsi intorno ai 50 Gbyte/sec.\
\
L'implementazione del kernel branchless in cui una cella corrispondeva ad un thread inizialmente non aveva il sistema a shared memory, bensì caricava i dati direttamente dalla memoria globale. Dopo aver generato i report di ncu ed averli analizzati si è notato che il tempo di esecuzione era spaventosamente alto, nonostante l'occupancy avesse un valore alto (oltre il 90%, ciò non necessariamente è un bene) e le percentuali di cache hit fossero maggiori rispetto ai casi precedenti. Quindi si è cercato di ridurre il più possibile gli accessi in memoria, che rappresentavano il vero collo di bottiglia del, introducendo l'utilizzo della shared memory. Oltre a ciò si è cercato di valutare la variazione della dimensione del blocco, che inizialmente era 32x32 portandola a 16x16 equivalentemente al caso precedente. La combinazione delle varie ottimizzazioni ha portato ad un significativo miglioramento del tempo di esecuzione nell'ordine del centinaio (lievemente superiore) di microsecondi.