#lang rosette

(require rackunit)

(define (activation inputs weights)
  (define dot
    (apply + (map * inputs weights)))
  (if (> dot 0)
      dot
      0))

; output = 1*4 + 2*5 + 3*6
(check-equal? (activation (list 1 2 3) (list 4 5 6)) 32)

(define (layer inputs weight-matrix)
  (for/list ([weights (in-list weight-matrix)])
    (activation inputs weights)))

; There are three inputs
; Since weight-matrix contains two lists --> three input units connected
; with two output units
; See Andrew Ng's ML on Coursera (Lec8: Neural Networks: Representation)
(check-equal? (layer (list 3 2 1) (list (list 4 5 6) (list 7 8 9))) (list 28 46))

(define (nn inputs weight-matrix1 weight-matrix2)
  (layer (layer inputs weight-matrix1) weight-matrix2))

(check-equal? (nn (list 1 2 3) (list (list 4 5 6) (list 7 8 9)) (list (list 10 11) (list 12 13) (list 14 15))) (list 870 1034 1198))

;(define-symbolic* inputs real? [2])
;(define-symbolic k integer?)

(define (create-weight-matrix m n)
  (define (new-row)
    (define-symbolic* row integer? [n])
    row)
  (define (create-weight-matrix-helper m n res)
    (if (= m 0)
        res
        (create-weight-matrix-helper (- m 1) n (cons (new-row) res))))
  (create-weight-matrix-helper m n '()))

(check-equal? (length (create-weight-matrix 3 4)) 3)

(define (create-k-list k)
  (define (new-m)
    (define-symbolic* m integer?)
    m)
  (define (create-k-list-helper k res)
    (if (= k 0)
        res
        (create-k-list-helper (- k 1) (cons (new-m) res))))
  (create-k-list-helper k '()))

; k-list specifies a list of neourons for each layer
; e.g., (list 3 2) means 2 layers with first layer containing 3 neurons
; and second layer containing 2 neurons
(define (sketch inputs k-list)
  (define dims
    (cons (length inputs) (append k-list '(1))))
  (define (sketch-helper inputs k-list)
    (if (= (length k-list) 1)
        inputs
        (sketch-helper (layer inputs (create-weight-matrix (car (cdr k-list)) (car k-list))) (cdr k-list)))) 
  (sketch-helper inputs dims))

;(define (xor a b)
;  (define a-bits
;    (+ (exact-floor (log a 2)) 1))
;  (define b-bits
;    (+ (exact-floor (log b 2)) 1))
;  (define bits (max a-bits b-bits))
;  (define a-bvec
;    (integer->bitvector a (bitvector bits)))
;  (define b-bvec
;    (integer->bitvector b (bitvector bits)))
;  (bitvector->integer (bvxor a-bvec b-bvec)))

(define (xor a b)
  (define bits 10)
  (define a-bvec
    (integer->bitvector a (bitvector bits)))
  (define b-bvec
    (integer->bitvector b (bitvector bits)))
  (bitvector->integer (bvxor a-bvec b-bvec)))

(define-symbolic x y integer?)

(synthesize
 #:forall (list x y)
 #:guarantee (assert (= (car (sketch (list x y) (list 2))) (xor x y))))