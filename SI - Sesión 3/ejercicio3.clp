(deffacts datos
   (cadena1 B C A D E E B C E)
   (cadena2 E E B C A F E)
)

(defrule unir-cadenas
   ?f1 <- (cadena1 $?l1)
   ?f2 <- (cadena2 $?l2)
   =>
   (bind ?union (create$))
   (foreach ?elem ?l1
      (if (and (member$ ?elem ?l2)
               (not (member$ ?elem ?union)))
          then
          (bind ?union (create$ ?union ?elem))
      )
   )
   (assert (union ?union))
)

(defrule mostrar-union
   (union $?resultado)
   =>
   (printout t "Resultado de la unión: " $?resultado crlf)
)
