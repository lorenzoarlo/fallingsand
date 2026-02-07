= Conclusioni
In conclusione si può affermare che le ottimizzazioni effettuate hanno permesso di ottenere un miglioramento significativo delle prestazioni utilizzando sia ottimizzazioni vettoriali, sia attraverso operazioni di calcolo parallelo.

In particolare l'ottimizzazione SIMD ha permesso di ottenere un miglioramento di circa $7$ volte rispetto alla versione base, anche superiore rispetto all'ottimizzazione fatta automaticamente dal compilatore, con risultati coerenti per tutti i _sample_.
Le ottimizzazioni di calcolo parallelo hanno permesso di ottenere un miglioramento inferiore per i sample più piccoli (a causa dell'overhead introdotto dal caricamento dei dati sul device) che diventa più significativo per il sample di dimensione maggiore ($1920 times 1080$) per un maggiore numero di frame, con uno speedup di oltre $6$ volte per la versione _block branchless_.
Il principale collo di bottiglia individuato (nella versione parallela) risulta essere legato all'accesso in memoria dei dati che causa un overhead significativo.
