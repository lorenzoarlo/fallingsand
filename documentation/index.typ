#import "functions.typ": title
#set document(title: title, author: "Lorenzo Arlotti, Pierluca Pevere")


// Default style
#set page(margin: 2.5cm, paper: "a4")
#set par(leading: 0.75em, spacing: 0.75em, justify: true)
#set text(font: "New Computer Modern", size: 12pt, lang: "it", weight: "medium")
// #show raw: set text(font: "New Computer Modern Mono")

#show ref: underline
#show link: underline

#show figure: set block(inset: ("top": 6pt, "bottom": 6pt))

#set list(indent: 8pt, marker: ("-", "◦", "▹"), spacing: 0.75em)

#show math.equation: set block(above: 1.5em)


#set table(inset: (8pt, 8pt))

#include "title-page.typ"


#show outline.entry.where(
  level: 1,
): set block(above: 1em)

#outline(title: "Indice")

#pagebreak()

// Set page numbering

#set par(justify: true)



#set page(
  numbering: "1",
  number-align: center,
)



#counter(page).update(1)

#show heading: set block(above: 1em, below: 1em)

#show link: underline
#show ref: underline


// level: 1

#show heading.where(level: 1): set heading(numbering: "1")
#show heading.where(level: 1): set text(size: 22pt)

// level: 2
#show heading.where(level: 2): set heading(numbering: "1.")
#show heading.where(level: 2): set text(size: 18pt)


// level: 3
#show heading.where(level: 3): set heading(numbering: "1.")
#show heading.where(level: 3): set text(size: 16pt)


// level: 4
#show heading.where(level: 4): set heading(numbering: "1.")
#show heading.where(level: 4): set text(size: 13pt, fill: rgb("#0021c7"))

// level: 5
#show heading.where(level: 5): set heading(outlined: false)
#show heading.where(level: 5): it => [#it.body #h(2em)]
#show heading.where(level: 5): set text(size: 13pt, fill: rgb("#004cc7"))

// level: 6
#show heading.where(level: 6): set heading(outlined: false)
#show heading.where(level: 6): it => [#it.body #h(1em)]
#show heading.where(level: 6): set text(size: 12pt, fill: rgb("#007ec7"))

#set page(
  header: [#align(right)[#emph[#title]]],
  footer: context table(
    stroke: none,
    columns: (33.3%, 33.4%, 33.3%),
    [],
    [#align(center)[
      #counter(page).display()
    ]],
    [
      #align(right)[versione #version]
    ],
  ),
)

#include "document-flow.typ"
