== Implementazione base
L'implementazione sequenziale base#footnote[visibile in ] è stata realizzata in C ed è stata scritta in modo da essere il più leggibile possibile, utilizzando `if` e svolgendo ogni operazione in modo esplicito ed il meno complicato possibile.
Nonostante questo, sono state adottate ottimizzazioni base per rendere l'implementazione più efficiente possibile senza compromettere la leggibilità del codice.
Tra le ottimizzazioni adottate,
evitare di processare blocchi completamente uguali, l'utilizzo di funzioni _inline_ per evitare l'overhead di chiamata.

Nonostante non si sia prestata particolare attenzione all'efficienza, tale versione risulta essere comunque particolarmente efficiente, soprattutto per configurazioni non troppo dinamiche e con molte celle vuote o ravvicinate.


