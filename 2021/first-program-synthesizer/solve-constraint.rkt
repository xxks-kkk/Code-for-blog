#lang rosette/safe

; This script shows how to use Rosette to find a model
; (an assignment of values to all the symbolic variables)
; for the given constraint

; Compute the absolute value of `x`
(define (absv x)
  (if (< x 0) (- x) x))

; Define a symbolic variable called y of type integer.
(define-symbolic y integer?)

; Solve a constraint saying |y| = 5
; returns a mode with y = -5
(solve
 (assert (= (abs y) 5)))

; returns unsat: there is no possible y that has a negative absolute value.
(solve
 (assert (< (abs y) 0)))