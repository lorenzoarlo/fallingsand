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



== Struttura del codice

Il programma realizzato è scritto in C e le operazioni che esegue ad ogni esecuzione sono:
- lettura dello stato iniziale dal file di input in formato `.sand`, particolare formato binario che codifica lo stato della matrice;
- esecuzione dell'algoritmo di simulazione per il numero di iterazioni richiesto e salvataggio di ogni stato intermedio in un file `.sand` indicato;
- eventuale *testing di corretta generazione* rispetto ad un file di riferimento `.sand` fornito;
- *eventuale generazione delle immagini* `.png` di ogni stato intermedio salvato nella cartella di output indicata.

=== Rappresentazione dei dati
Ogni stato dell'automa cellulare è rappresentato da una struct `Universe` così definita:
```cpp
struct Universe {
    unsigned char *cells; // Vettore monodimensionale che rappresenta la matrice (row-major order)
    int width; // Larghezza della matrice
    int height; // Altezza della matrice
};
```
Al fine di semplificare la lettura dei dati, sono anche definite le seguenti costanti simboliche:
```cpp
#define P_EMPTY 0
#define P_WATER 1
#define P_SAND 2
#define P_WALL 3
```
il cui valore rappresenta anche la \"densità\" della particella.


=== Formato `.sand`
Il formato `.sand` è un formato binario che codifica lo stato della matrice di simulazione.
Dato che ogni particella è rappresentata da un `unsigned char` ma sono utilizzati solo $4$ valori, ogni valore della matrice può essere memorizzato in soli $2$ bit permettendo una maggiore compressione dei dati.
Ogni file `.sand` è strutturato come segue:
- i primi $4$ byte indicano il magic number `SAND` (in ASCII);
- i successivi $4$ byte rappresentano la larghezza della matrice;
- i successivi $4$ byte rappresentano l'altezza della matrice;
- i successivi $4$ byte indicano il numero di frame salvati;
- i successivi $"width" times "height"$ valori (ognuno rappresentato da $2$ bit) rappresentano il vettore `cells` della struct `Universe` per ogni frame salvato, in ordine di generazione.

=== Algoritmo di simulazione
L'algoritmo di simulazione ha preso ispirazione dall'implementazione consultabile al seguente #link("https://gelamisalami.github.io/GPU-Falling-Sand-CA/")[link] @GPUFallingSandCA, ma il risultato grafico non differisce particolarmente da altre implementazioni.

Particolarità di questa implementazione riguarda il fatto che ad ogni iterazione lo stato della matrice dipende solamente dallo stato precedente e non dallo stato in corso di aggiornamento.

Per farlo, utilizza come base il *pattern di Margolus*, studiato nella teoria degli automi cellulari e utile per ottenere la conservazione della massa e la reversibilità (non utile per il nostro caso).

Tale pattern prevede la divisione della matrice in blocchi $2 times 2$ per cui si applicano le regole di aggiornamento che delineeranno l'evoluzione. Nelle iterazioni successive i blocchi saranno applicati con particolari offset, in modo da \"applicare le regole\" a tutte le celle della matrice.

#figure(
  image("assets/pattern-margolus.png", width: 70%),
  caption: [Pattern di Margolus]
)

Le regole dedotte dall'implementazione @GPUFallingSandCA e utilizzate permettono di modificare ogni blocco in maniera \"sequenziale\" senza la necessità di tenere conto delle modifiche effettuate al blocco stesso, permettendo l'applicazione di ogni regola basandosi solamente sullo stato precedente.
In particolare, le regole riguardano lo scambio di posizione delle particelle (in modo da garantire la conservazione della massa) sono le seguenti:
#enum.item()[
  se una particella di  _sabbia_ si trova nella parte alta del blocco ed è \"in caduta\" (quindi si trovano sopra a particelle meno dense) è possibile (con una certa probabilità definita) che si muova orizzontalmente (scambiandosi di posizione con la particella ad essa adiacente nel blocco);
]
#enum.item()[
  ipotizzando che la particella di _sabbia_ non si sia mossa orizzontalmente, questa verifica che la particella sottostante sia meno densa ed in questo caso è possibile (con una certa probabilità) che si muova verticalmente (scambiandosi di posizione con la particella sottostante);
  Nel caso la particella sottostante fosse pià densa, la particella di _sabbia_ potrebbe comunque spostarsi diagonalmente (se al suo fianco e nella sua destinazione fossero presenti particelle meno dense);
]
#enum.item()[
  considerando ora la presenza di particelle _acqua_, queste si comportano in maniera simile alla _sabbia_, ma considerando il fatto che sia possibile (con una certa probabilità) il movimento diagonale (sempre rispettando le regole di densità) ed il \"galleggiamento\" nonostante la possibilità di movimento verticale e/o diagonale
]
#enum.item()[
  considerando invece il caso in cui delle particelle d'acqua si trovino sopra a particelle più dense (e non siano già cadute), queste potrebbero comunque spostarsi orizzontalmente con una certa probabilità;
]
#enum.item()[
  sempre considerando il caso di particelle d'acqua, è necessario considerare l'eventuale presenza di particelle più dense al di sotto del blocco ed eventualmente (con una certa probabilità) spostarsi orizzontalmente (sempre rispettando le regole di densità);
]

É da notare che proprio per la struttura dell'algoritmo, qualsiasi *cambiamento ad un blocco composto da quattro celle identiche* non varierebbe la combinazione, rendendo inutile l'applicazione delle regole.

Per garantire il determinismo, quando è necessario verificare la probabilità di un evento, è utilizzato un generatore di numeri pseudo-casuali basati sulla posizione della particella e dalla generazione (il numero del frame).

Tale logica per essere applicata correttamente nel programma, deve essere implementata in una funzione che rispetta l'intestazione
```cpp
void next(Universe *in, Universe *out, int generation);
```
Non sono presenti vincoli sull'integrità dei dati in input, quindi è teoricamente possibile restituire come output in `out` esattamente il puntatore ad `in` con il contenuto modificato.

Sarà sul tempo di esecuzione di tale funzione che saranno effettuate le misurazioni delle performance delle diverse ottimizzazioni.


== Analisi del problema
Come già definito, il cuore delle diverse ottimizzazioni è rappresentato dalla funzione `next`, che deve essere eseguita per il numero di iterazioni richiesto.
In ingresso è ricevuto un puntatore ad una struct `Universe`, contenente un vettore monodimensionale che rappresenta la matrice in _row-major order_.
Come specificato nella definizione dell'algoritmo, non sono presenti dipendenze tra i dati di blocchi diversi ed è solamente necessario garantire la coerenza degli scambi all'interno dello stesso blocco. Ciò rende l'algoritmo *facilmente parallelizzabile* (_embarrassingly parallel_) e ottimizzabile per architetture vettoriali.
