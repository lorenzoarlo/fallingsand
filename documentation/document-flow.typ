= Implementazione base
L'automa cellulare _falling sand_ è stato implementato nel corso del tempo in diversi modi e con diversi sistemi di regole.
Al fine di delineare l'algoritmo più adatto ai nostri scopi, sono stati analizzate e confrontate diverse implementazioni, fino a raggiungere quella considerata migliore.

Il primo vincolo è stato quello di mantenere un'implementazione *completamente deterministica*, in modo da poter riprodurre sempre gli stessi risultati a partire dallo stesso stato iniziale.
In secondo luogo, è stato scelto un sistema di regole per cui lo stato $n + 1$ della matrice dipenda solo dallo stato $n$, in modo da diminuire il numero di dipendenze tra i dati.

L'obiettivo di ogni implementazione era generare un numero prefissato di frame ($1500$) la simulazione di due tipi di 
