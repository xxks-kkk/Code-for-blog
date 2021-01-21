#lang rosette/safe

; This script demonstrates some features of Rossette

(require rosette/lib/angelic  ; provides `choose*`
         rosette/lib/match    ; provides `match`
         )
; Tell Rosette we really do want to use integers
(current-bitwidth #f)

(define (fact x)
  (if (<= x 0)
      1
      (* x (fact (- x 1)))))

; transparent is to print struct values in the REPL rather
; than hiding them. Convenient for debugging.
(struct btree-int (val left right) #:transparent)
(struct btree-leaf (val) #:transparent)

; match seems to be rosette-specific
(define (sum-tree x)
  (match x
    [(btree-int v l r) (+ v (sum-tree l) (sum-tree r))]
    [(btree-leaf v) v]
    [_ "Fail"]))

(sum-tree (btree-int 10 (btree-leaf 10) (btree-leaf 1)))