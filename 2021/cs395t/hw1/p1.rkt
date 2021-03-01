#lang rosette/safe

(require rackunit)
(require rosette/query/debug rosette/lib/render)
(require rosette/solver/smt/z3)

; some skretch for the record. Maybe useful in the future.

;(define (compatible x y)
;  (cond [(= (quotient x 4) (quotient y 4)) false]
;        [(= (remainder x 4) (remainder y 4)) false]
;        [#t (for ([k '(1 2 3)])
;              (cond [(= (+ x (* 3 k)) y) false]
;                    [(= (- x (* 3 k)) y) false]
;                    [(= (+ x (* 5 k)) y) false]
;                    [(= (- x (* 5 k)) y) false]
;                    [#t true]))]))

(define (compatible-helper-1 x y i step_size)
  (cond [(<= i 0) true]
        [(= (- x i) y) false]
        [(= (+ x i) y) false]
        [#t (compatible-helper-1 x y (- i step_size) step_size)]))

(check-equal? (compatible-helper-1 2 5 9 3) false)
(check-equal? (compatible-helper-1 4 13 9 3) false)
(check-equal? (compatible-helper-1 16 6 15 5) false)
(check-equal? (compatible-helper-1 8 2 9 3) false)

(define (is-diag x y)
  (or (equal? (abs (- (quotient (- x 1) 4) (remainder (- x 1) 4)))
              (abs (- (quotient (- y 1) 4 ) (remainder (- y 1) 4))))
      (equal? (+ (quotient (- x 1) 4) (remainder (- x 1) 4))
              (+ (quotient (- y 1) 4) (remainder (- y 1) 4)))))

(check-equal? (is-diag 11 16) true)

(define (compatible x y)
  (cond [(= (quotient (- x 1) 4) (quotient (- y 1) 4)) false]
        [(= (remainder x 4) (remainder y 4)) false]
        [(is-diag x y) false]
;        [(not (compatible-helper-1 x y 15 5)) false]
;        [(not (compatible-helper-1 x y 9 3)) false]
        [#t true]))

(check-equal? (compatible 2 7) false)
(check-equal? (compatible 3 15) false)
(check-equal? (compatible 8 15) true)
(check-equal? (compatible 2 4) false)
(check-equal? (compatible 8 2) true)
(check-equal? (compatible 9 15) true)


(define (fit-conditions x inputs)
  (and (andmap (lambda (y) (<= y 16)) inputs)
       (andmap (lambda (y) (>= y 1)) inputs)
       (<= x 16)
       (>= x 1)
       (andmap (lambda (y) (compatible x y)) inputs)))
; add constraint length of inputs == 4?

;(define res (evaluate y (sol)))

(define (get-one-sol inputs)
  (display inputs)
  (define-symbolic k integer?)
  (evaluate k
            (solve
             (assert (fit-conditions k inputs)))))

(define (find-solution inputs)
  (if (>= (length inputs) 4)
      inputs
      (find-solution (cons (get-one-sol inputs) inputs))))

; another idea (using bitvector)
(define board
  (bv 0 (bitvector 16)))

; 3rd idea
(define-symbolic x y z w integer?)
;(define vs (vector x y z w))

(define (is-diag-2 x y)
  (or (equal? (abs (- (quotient x 4) (remainder x 4)))
              (abs (- (quotient y 4 ) (remainder y 4))))
      (equal? (+ (quotient x 4) (remainder x 4))
              (+ (quotient y 4) (remainder y 4)))))

(check-equal? (is-diag-2 3 7) false)
(check-equal? (is-diag-2 2 7) true)

(define (compatible-2 x y)
  (cond [(= (quotient x 4) (quotient y 4)) false]
        [(= (remainder x 4) (remainder y 4)) false]
        [(is-diag-2 x y) false]
        [#t true]))

(define (fit-conditions-2 vs)
  (and (andmap (lambda (y) (<= y 15)) (vector->list vs))
       (andmap (lambda (y) (>= y 0)) (vector->list vs))
       (compatible-2 (vector-ref vs 0) (vector-ref vs 1))
       (compatible-2 (vector-ref vs 0) (vector-ref vs 2))
       (compatible-2 (vector-ref vs 0) (vector-ref vs 3))
       (compatible-2 (vector-ref vs 1) (vector-ref vs 2))
       (compatible-2 (vector-ref vs 1) (vector-ref vs 3))
       (compatible-2 (vector-ref vs 2) (vector-ref vs 3))
       ))

(define (get-all-sol inputs)
  (solve
   (assert (fit-conditions-2 inputs))))

(define (get-all-sol-2 inputs)
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

(get-all-sol-2 (vector 2 x y z))

;(define-symbolic* ...) can be used to define a list of symbolic variable