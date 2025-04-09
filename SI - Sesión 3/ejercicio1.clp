(deffacts datos-prueba
   (lista 8 3 5 1 9 4 2)
)

(defrule cargar-datos
   =>
   (load-facts datos-prueba)
)

(defrule hallar-minimo
   (lista $?valores)
   =>
   (bind ?min (nth$ 1 ?valores))
   (foreach ?n ?valores
      (if (< ?n ?min) then
         (bind ?min ?n)
      )
   )
   (assert (minimo ?min))
)

(defrule mostrar-minimo
   (minimo ?m)
   =>
   (printout t "El valor mínimo en la lista es: " ?m crlf)
)
