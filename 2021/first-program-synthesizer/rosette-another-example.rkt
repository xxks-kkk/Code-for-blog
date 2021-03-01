#lang rosette

; An example to show something surprise when use rosette

(define-symbolic x y integer?)

(define (range x)
  (or (= x 0) (= x 1)))

(define-symbolic c integer?)
(synthesize
 #:forall (list x y)
 #:guarantee (begin
               ;(assert (range y)) ;this leads to unsat b/c for not all x equals to 0 or 1
               ;(assert (range x))
               (assert (= (* c (+ x y)) (+ (+ (+ x y) x) y)))
               ))