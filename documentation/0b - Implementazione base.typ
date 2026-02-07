== Implementazione base
L'implementazione sequenziale base (consultabile in #ref(<appendice-implementazione-sequenziale>)) è stata realizzata in C ed è stata scritta in modo da essere il più leggibile possibile, utilizzando `if` e svolgendo ogni operazione in modo esplicito ed il meno complicato possibile.
Nonostante questo, sono state adottate ottimizzazioni base per rendere l'implementazione più efficiente possibile senza compromettere la leggibilità del codice.
Tra le ottimizzazioni adottate, si è evitato di processare blocchi completamente uguali e si sono utilizzate funzioni _inline_ per evitare l'overhead di chiamata.

