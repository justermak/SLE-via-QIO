#import math

#let project(title: "", author: "", body) = {
  set document(author: author, title: title)
  set page(numbering: "1", number-align: center, margin: 24pt)
  set par(justify: true, linebreaks: "optimized")
  set text(font: "New Computer Modern", lang: "en", size: 14pt)

  align(center)[
    #pad(
      top: 0.3em,
      bottom: 0.3em,
      text(weight: 700, 1.75em, title)
    )
    #pad(
      top: 0.3em,
      bottom: 0.3em,
      strong(author)
    )
  ]

  show heading: it => {
    if it.level > 3 {
      parbreak()
      text(style: "normal", weight: "regular", it.body)
    } else {
      it
    }
  }

  set par(justify: true)
  set text(hyphenate: false)

  body
}

#let sp = h(4pt)
#let spa = h(8pt)
#let spac = h(16pt)
#let space = h(32pt)
