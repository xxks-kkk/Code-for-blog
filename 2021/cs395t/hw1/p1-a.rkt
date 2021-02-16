#lang rosette/safe

(require rackunit)

; Solve 4-queen problem

; represents 4 queens
(define-symbolic x y z w integer?)

; check whether x y are on the same diagonal
(define (is-diag-2 x y)
  (or (equal? (- (quotient x 4) (remainder x 4))
              (- (quotient y 4) (remainder y 4)))
      (equal? (+ (quotient x 4) (remainder x 4))
              (+ (quotient y 4) (remainder y 4)))))

(check-equal? (is-diag-2 3 7) false)
(check-equal? (is-diag-2 2 7) true)
(check-equal? (is-diag-2 4 11) false)

; check whether x y are compatible on the board
(define (compatible-2 x y)
  (cond [(= (quotient x 4) (quotient y 4)) false]
        [(= (remainder x 4) (remainder y 4)) false]
        [(is-diag-2 x y) false]
        [#t true]))

; solve the problem
(define (queens-solve-4 inputs)
  (solve
   (begin
     (assert (andmap (lambda (y) (<= y 15)) (vector->list inputs)))
     (assert (andmap (lambda (y) (>= y 0)) (vector->list inputs)))
     (assert (compatible-2 (vector-ref inputs 0) (vector-ref inputs 1)))
     (assert (compatible-2 (vector-ref inputs 0) (vector-ref inputs 2)))
     (assert (compatible-2 (vector-ref inputs 0) (vector-ref inputs 3)))
     (assert (compatible-2 (vector-ref inputs 1) (vector-ref inputs 2)))
     (assert (compatible-2 (vector-ref inputs 1) (vector-ref inputs 3)))
     (assert (compatible-2 (vector-ref inputs 2) (vector-ref inputs 3)))
     )))

(queens-solve-4 (vector 2 x y z))
(queens-solve-4 (vector x y z w))
(queens-solve-4 (vector 0 9 z w))