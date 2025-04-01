% Base de conocimientos de carreras y sus requisitos
carrera(ingenieria_sistemas) :-
    tiene_habilidad(matematicas),
    tiene_habilidad(resolucion_problemas),
    tiene_interes(tecnologia),
    tiene_personalidad(analitico),
    prefiere_condicion(trabajo_equipo) o prefiere_condicion(autonomia).

carrera(ingenieria_industrial) :-
    tiene_habilidad(matematicas),
    tiene_habilidad(liderazgo),
    tiene_interes(tecnologia) o tiene_interes(negocios),
    tiene_personalidad(organizado).

carrera(medicina) :-
    tiene_interes(salud),
    tiene_habilidad(biologia),
    tiene_personalidad(empatico),
    prefiere_condicion(interaccion_social),
    tiene_rasgo(meticulosidad).

carrera(psicologia) :-
    tiene_interes(salud) o tiene_interes(educacion),
    tiene_personalidad(empatico),
    tiene_habilidad(escucha_activa),
    prefiere_condicion(interaccion_social).

carrera(derecho) :-
    tiene_habilidad(argumentacion),
    tiene_rasgo(sentido_justicia),
    tiene_interes(servicio_social),
    tiene_personalidad(extrovertido) o tiene_personalidad(pensamiento_critico).

carrera(arquitectura) :-
    tiene_habilidad(creatividad),
    tiene_interes(diseño),
    tiene_habilidad(matematicas),
    prefiere_condicion(trabajo_proyectos).

carrera(diseno_grafico) :-
    tiene_habilidad(creatividad),
    tiene_interes(diseño),
    tiene_habilidad(artes),
    prefiere_condicion(trabajo_independiente).

carrera(administracion_empresas) :-
    tiene_habilidad(liderazgo),
    tiene_interes(negocios),
    tiene_personalidad(extrovertido),
    prefiere_condicion(trabajo_equipo).

carrera(marketing) :-
    tiene_habilidad(creatividad),
    tiene_interes(negocios),
    tiene_personalidad(extrovertido),
    prefiere_condicion(trabajo_dinamico).

carrera(economia) :-
    tiene_habilidad(matematicas),
    tiene_habilidad(analisis_datos),
    tiene_interes(negocios) o tiene_interes(investigacion),
    tiene_personalidad(analitico).

carrera(contabilidad) :-
    tiene_habilidad(matematicas),
    tiene_rasgo(meticulosidad),
    tiene_interes(negocios),
    tiene_personalidad(detallista).

carrera(ingenieria_civil) :-
    tiene_habilidad(matematicas),
    tiene_habilidad(resolucion_problemas),
    tiene_interes(construccion),
    prefiere_condicion(trabajo_exterior).

carrera(biologia) :-
    tiene_interes(investigacion),
    tiene_habilidad(biologia),
    tiene_personalidad(curioso),
    prefiere_condicion(laboratorio).

carrera(fisica) :-
    tiene_habilidad(matematicas),
    tiene_interes(investigacion),
    tiene_personalidad(analitico),
    prefiere_condicion(laboratorio).

carrera(quimica) :-
    tiene_habilidad(quimica),
    tiene_interes(investigacion),
    tiene_rasgo(meticulosidad),
    prefiere_condicion(laboratorio).

carrera(educacion) :-
    tiene_interes(educacion),
    tiene_habilidad(comunicacion),
    tiene_personalidad(paciente),
    prefiere_condicion(interaccion_social).

carrera(turismo) :-
    tiene_interes(cultura),
    tiene_personalidad(extrovertido),
    tiene_habilidad(idiomas),
    prefiere_condicion(trabajo_dinamico).

carrera(relaciones_internacionales) :-
    tiene_habilidad(idiomas),
    tiene_interes(cultura),
    tiene_personalidad(extrovertido),
    prefiere_condicion(trabajo_internacional).

% Operador "o" para condiciones alternativas
:- op(500, xfy, o).

A o B :- A.
A o B :- B.

% Sistema de preguntas dinámicas
:- dynamic si/1, no/1.

preguntar(Pregunta) :-
    (si(Pregunta) -> true ;
     (no(Pregunta) -> fail ;
      mostrar_pregunta(Pregunta))).

mostrar_pregunta(Pregunta) :-
    format('~w? (s/n): ', [Pregunta]),
    read(Respuesta),
    (Respuesta = s -> assert(si(Pregunta)) ;
     assert(no(Pregunta)), fail).

% Reglas para determinar habilidades, intereses, etc. basadas en preguntas
tiene_habilidad(matematicas) :-
    preguntar('¿Tienes habilidad para las matemáticas').

tiene_habilidad(resolucion_problemas) :-
    preguntar('¿Eres bueno resolviendo problemas complejos').

tiene_habilidad(creatividad) :-
    preguntar('¿Consideras que eres una persona creativa').

tiene_habilidad(liderazgo) :-
    preguntar('¿Tienes habilidades de liderazgo').

tiene_habilidad(argumentacion) :-
    preguntar('¿Eres bueno argumentando y debatiendo').

tiene_habilidad(biologia) :-
    preguntar('¿Tienes facilidad para la biología').

tiene_habilidad(quimica) :-
    preguntar('¿Tienes facilidad para la química').

tiene_habilidad(comunicacion) :-
    preguntar('¿Tienes buenas habilidades de comunicación').

tiene_habilidad(idiomas) :-
    preguntar('¿Tienes facilidad para aprender idiomas').

tiene_habilidad(artes) :-
    preguntar('¿Tienes habilidades artísticas').

tiene_habilidad(escucha_activa) :-
    preguntar('¿Eres bueno escuchando y comprendiendo a otros').

tiene_habilidad(analisis_datos) :-
    preguntar('¿Tienes habilidad para analizar datos e información').

tiene_interes(tecnologia) :-
    preguntar('¿Te interesa la tecnología').

tiene_interes(salud) :-
    preguntar('¿Te interesa el área de salud').

tiene_interes(negocios) :-
    preguntar('¿Te interesan los negocios y las finanzas').

tiene_interes(diseño) :-
    preguntar('¿Te interesa el diseño').

tiene_interes(investigacion) :-
    preguntar('¿Te interesa la investigación').

tiene_interes(educacion) :-
    preguntar('¿Te interesa la educación').

tiene_interes(construccion) :-
    preguntar('¿Te interesa la construcción y obras').

tiene_interes(cultura) :-
    preguntar('¿Te interesan diferentes culturas').

tiene_interes(servicio_social) :-
    preguntar('¿Te interesa servir a la sociedad').

tiene_personalidad(analitico) :-
    preguntar('¿Consideras que eres una persona analítica').

tiene_personalidad(empatico) :-
    preguntar('¿Eres una persona empática').

tiene_personalidad(extrovertido) :-
    preguntar('¿Eres una persona extrovertida').

tiene_personalidad(organizado) :-
    preguntar('¿Eres una persona organizada').

tiene_personalidad(pensamiento_critico) :-
    preguntar('¿Tienes pensamiento crítico').

tiene_personalidad(curioso) :-
    preguntar('¿Eres una persona curiosa').

tiene_personalidad(paciente) :-
    preguntar('¿Eres una persona paciente').

tiene_personalidad(detallista) :-
    preguntar('¿Eres una persona detallista').

tiene_rasgo(meticulosidad) :-
    preguntar('¿Eres meticuloso en tus actividades').

tiene_rasgo(sentido_justicia) :-
    preguntar('¿Tienes un fuerte sentido de la justicia').

prefiere_condicion(trabajo_equipo) :-
    preguntar('¿Prefieres trabajar en equipo').

prefiere_condicion(autonomia) :-
    preguntar('¿Prefieres trabajar con autonomía').

prefiere_condicion(interaccion_social) :-
    preguntar('¿Prefieres trabajos con mucha interacción social').

prefiere_condicion(trabajo_proyectos) :-
    preguntar('¿Prefieres trabajar por proyectos').

prefiere_condicion(trabajo_independiente) :-
    preguntar('¿Prefieres trabajar de manera independiente').

prefiere_condicion(trabajo_dinamico) :-
    preguntar('¿Prefieres un trabajo dinámico y variado').

prefiere_condicion(laboratorio) :-
    preguntar('¿Te gustaría trabajar en un laboratorio').

prefiere_condicion(trabajo_exterior) :-
    preguntar('¿Prefieres trabajar al aire libre').

prefiere_condicion(trabajo_internacional) :-
    preguntar('¿Te gustaría trabajar en un ámbito internacional').

% Función principal para recomendar carreras
recomendar_carrera :-
    retractall(si(_)),
    retractall(no(_)),
    findall(C, carrera(C), Carreras),
    mostrar_recomendaciones(Carreras).

mostrar_recomendaciones([]) :-
    write('No se encontraron carreras que coincidan con tu perfil.'), nl.

mostrar_recomendaciones([C|Cs]) :-
    format('Carrera recomendada: ~w~n', [C]),
    mostrar_recomendaciones(Cs).

