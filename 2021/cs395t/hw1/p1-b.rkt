#lang rosette

(require rackunit)

; Solve n-queen problem

; represents queens
(define-symbolic x y z w t integer?)

; check whether x y are on the same diagonal for nxn matrix
(define (is-diag-2 n x y)
  (or (equal? (- (quotient x n) (remainder x n))
              (- (quotient y n) (remainder y n)))
      (equal? (+ (quotient x n) (remainder x n))
              (+ (quotient y n) (remainder y n)))))

(check-equal? (is-diag-2 4 3 7) false)
(check-equal? (is-diag-2 4 2 7) true)
(check-equal? (is-diag-2 4 4 11) false)

; check whether x y are compatible on the board
(define (compatible-2 n x y)
  (cond [(= (quotient x n) (quotient y n)) false]
        [(= (remainder x n) (remainder y n)) false]
        [(is-diag-2 n x y) false]
        [#t true]))

(define (generate-compatible-checks-helper inputs cand n k lower_bound)
  (if (= (length cand) k)
      (assert (compatible-2 n (vector-ref inputs (car cand)) (vector-ref inputs (car (cdr cand)))))
      (for/list ([i (in-range lower_bound n)])
        (generate-compatible-checks-helper inputs (cons i cand) n k (add1 i)))))

; generate all possible compatible assert
(define (generate-compatible-checks n inputs)
  (generate-compatible-checks-helper inputs '() n 2 0))

; solve the problem
(define (queens-solve n inputs)
  (solve
   (begin
     (assert (andmap (lambda (y) (<= y (sub1 (* n n)))) (vector->list inputs)))
     (assert (andmap (lambda (y) (>= y 0)) (vector->list inputs)))
     (generate-compatible-checks n inputs)
     )))

(queens-solve 4 (vector 2 x y z))
(queens-solve 4 (vector x y z w))
(queens-solve 4 (vector 0 9 z w))
(queens-solve 3 (vector x y z))
(queens-solve 2 (vector x y))
(queens-solve 5 (vector x y z w t))
; Create xs as a list of 6 integer symbols
; See https://homes.cs.washington.edu/~emina/media/cav19-tutorial/part1.html
(define-symbolic* xs integer? [6])
(queens-solve 6 (list->vector xs))