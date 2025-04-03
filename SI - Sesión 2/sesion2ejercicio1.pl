% Definición de relaciones de parentesco
padre(hans, santiago).
padre(hans, carolina).
padre(hant, hans).
padre(hant, paola).
padre(jacinto, lily).
padre(jacinto, ausberto).
padre(jacinto, elsa).
padre(ausberto, marlee).
padre(ausberto, cynthia).
padre(srcampos, elizabeth).
padre(herbert, alexandra).
padre(herbert, luis).

madre(lily, santiago).
madre(lily, carolina).
madre(elizabeth, hans).
madre(elizabeth, paola).
madre(carmen, lily).
madre(carmen, ausberto).
madre(carmen, elsa).
madre(elsa, alexandra).
madre(elsa, luis).
madre(guillermina, elizabeth).

% Relaciones derivadas
hermano(A, B) :- padre(P, A), padre(P, B), madre(M, A), madre(M, B), A \= B.
abuelo(A, B) :- padre(A, X), padre(X, B).
abuelo(A, B) :- padre(A, X), madre(X, B).
abuela(A, B) :- madre(A, X), madre(X, B).
abuela(A, B) :- madre(A, X), padre(X, B).
tio(A, B) :- hermano(A, X), padre(X, B).
tia(A, B) :- hermana(A, X), madre(X, B).
hijo(A, B) :- padre(B, A).
hijo(A, B) :- madre(B, A).
hermana(A, B) :- hermano(A, B), mujer(A).
mujer(carolina).
mujer(lily).
mujer(elizabeth).
mujer(paola).
mujer(elsa).
mujer(marlee).
mujer(cynthia).
mujer(alexandra).
mujer(guillermina).
mujer(carmen).

% Consultas de ejemplo
% ?- abuelo(X, santiago).
% ?- padre(X, carolina).
% ?- hermana(X, hans).
% ?- tio(X, santiago).
