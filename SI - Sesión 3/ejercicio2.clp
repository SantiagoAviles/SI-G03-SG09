(deffacts datos-prueba
   (lista 7 4 2 9 3)
)

(defrule cargar-datos
   =>
   (load-facts datos-prueba)
)

(defrule sumar-elementos
   (lista $?valores)
   =>
   (bind ?suma 0)
   (foreach ?n ?valores
      (bind ?suma (+ ?suma ?n))
   )
   (assert (suma-total ?suma))
)

(defrule mostrar-suma
   (suma-total ?r)
   =>
   (printout t "La suma total es: " ?r crlf)
)
