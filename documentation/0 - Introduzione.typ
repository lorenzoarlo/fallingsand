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



= Introduzione
Esistono diverse implementazioni dell'automa cellulare#footnote[modello matematico e computazionale usato per descrivere l'evoluzione di sistemi complessi discreti attraverso semplici regole] _falling sand_ che prevedono diversi insiemi di regole.
Al fine di delineare l'algoritmo più adatto ai nostri scopi, sono state analizzate e confrontate diverse implementazioni, fino a trovare quella considerata migliore per i nostri obiettivi.

Il principale vincolo posto è stato quello di prevedere un'implementazione *completamente deterministica e replicabile in differenti versioni*, in modo da poter ottenere gli stessi risultati in output a partire dallo stesso stato iniziale.

L'obiettivo di ogni implementazione è la generazione dello stato $n + 1$ una volta fornito lo stato $n$

In particolare sono stati utilizzati i seguenti stati di input:
#list.item()[
  il _sample-1_ è una matrice $400 times 400$ contenente una massa di _sabbia_
  #figure(
    rect(
      image("assets/sample-1.png", width: 200pt),
    ),
    caption: [_sample-1_],
  )
  eseguito per $1500$ \"generazioni\";
]
#list.item()[
  il _sample-2_ è una matrice $400 times 400$ contenente una massa di _sabbia_ e una massa di _acqua_
  #figure(
    rect(
      image("assets/sample-2.png", width: 200pt),
    ),
    caption: [_sample-2_],
  )
  eseguito per $1500$ \"generazioni\";
]
#list.item()[
  il _sample-3_ è una matrice $1920 times 1080$ contenente masse di _sabbia_ e _acqua_ più numerose
  #figure(
    rect(
      image("assets/sample-3.png", height: 200pt),
    ),
    caption: [_sample-3_],
  )
  eseguito per $3000$ iterazioni.
]

Si ipotizza che ogni matrice sia contenuta in un rettangolo delimitato da _muri_ (per gestire eventuali accessi fuori dai limiti) in modo da conservare le particelle ad ogni iterazione.


#include "0a - Struttura del codice.typ"

#include "0b - Implementazione base.typ"
