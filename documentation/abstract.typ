= Abstract <abstract>

Le simulazioni di tipo \"_falling sand_\" rappresentano una classe di automi cellulari#footnote[modello matematico e computazionale usato per descrivere l'evoluzione di sistemi complessi discreti attraverso semplici regole] in cui ogni elemento di una matrice *simula una particella soggetta a particolari regole* (in questo caso la gravità). Esistono diverse implementazioni, che variano per complessità, per tipi di particelle coinvolti e per regole di interazione tra di esse.

Questa relazione riguarda il progetto di ottimizzazione SIMD#footnote[appogiandosi sulla libreria Highway] e SIMT#footnote[utilizzando CUDA] di una simulazione che prevede l'esistenza di due particelle (oltre a quelle statiche di _vuoto_ e _muro_) cioè la _sabbia_ e l'_acqua_, svolta per il progetto del corso di \"*Sistemi di elaborazione accelerata*\".

#v(10pt) _Lorenzo Arlotti_, _Pierluca Pevere_
