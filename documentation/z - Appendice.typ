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


= Appendice <appendice>

== Codice implementazione sequenziale di riferimento <appendice-implementazione-sequenziale>

Il codice dell'implementazione sequenziale di riferimento è consultabile nel file `src/logic.c` (e quindi `src/utility/utility-functions.h` in cui sono state fattorizzate alcune funzioni di utilità) a @FallingSandProject.
Di seguito è comunque riportato un estratto del codice ritenuto particolarmente interessante

#codly(number-format: numbering.with("1"))
#raw(block: true, read("./assets/code/implementazione-sequenziale.c"), lang: "cpp")
#codly(number-format: none)


