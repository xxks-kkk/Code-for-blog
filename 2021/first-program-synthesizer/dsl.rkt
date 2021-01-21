#lang rosette/safe

(require rosette/lib/destruct) ; allows us to use destruct

; This script shows how to define a simple arithmetic DSL

(struct plus (left right) #:transparent)
(struct mul (left right) #:transparent)
(struct square (arg) #:transparent)

; destruct is another way of Rosette's pattern matching (another is match)
; doc: https://docs.racket-lang.org/rosette-guide/sec_utility-libs.html?q=destruct#%28form._%28%28lib._rosette%2Flib%2Fdestruct..rkt%29._destruct%29%29
(define (interpret p)
  (destruct p
    [(plus a b)  (+ (interpret a) (interpret b))]
    [(mul a b)   (* (interpret a) (interpret b))]
    [(square a)  (expt (interpret a) 2)]
    [_ p]))

(define prog (plus 7 (mul 10 (square 11))))
(interpret prog)

; interpret works with symbolic as well
(define-symbolic y integer?)
(interpret (plus 5 (mul 7 y)))

(solve
 (assert (= (interpret (plus 5 (mul 7 y))) 61)))