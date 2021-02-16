#lang rosette/safe

(require rackunit)

(require rosette/lib/angelic    ; provides `choose*`
         rosette/lib/destruct)  ; provides `destruct`

(struct notexp (a)   #:transparent)
(struct andexp (a b) #:transparent)
(struct orexp  (a b) #:transparent)
(struct iffexp (a b) #:transparent)
(struct xorexp (a b) #:transparent)

(define (interpret p)
  (destruct p
            [(notexp a) (not (interpret a))]
            [(andexp a b) (and (interpret a) (interpret b))]
            [(orexp a b) (or (interpret a) (interpret b))]
            [(iffexp a b) (and (or (not (interpret a)) (interpret b))
                               (or (not (interpret b)) (interpret a)))]
            [(xorexp a b) (and (or (interpret a) (interpret b)) (not (and (interpret a) (interpret b))))]
            [_ p]))

(check-equal? (interpret (iffexp true false)) false)
(check-equal? (interpret (iffexp true true)) true)
(check-equal? (interpret (iffexp false false)) true)
(check-equal? (interpret (xorexp true true)) false)
(check-equal? (interpret (xorexp false true)) true)

(define-symbolic x y z w boolean?)

(define (ver impl spec)
  (define (sol) (synthesize
               #:forall (list x)
               #:guarantee (assert (eq? (interpret impl) (interpret spec)))))
  (if (sat? (sol))
      true
      (begin
        (define (unsat-sol) (solve
               (assert (not (eq? (interpret impl) (interpret spec))))))
        (complete-solution (unsat-sol) (cons spec '())))))

(check-equal? (ver (orexp false x) x) true)
(check-equal? (ver x x) true)
(check-equal? (evaluate x (ver (notexp x) x)) false)
(check-equal? (ver (orexp y x) x) true)
(check-equal? (ver (implies x (not y)) (orexp (notexp x) (notexp y))) true)

(define (??expr terminals)
  (define a (apply choose* terminals))
  (define b (apply choose* terminals))
  (choose* (notexp a)
           (andexp a b)
           (orexp a b)
           (iffexp a b)
           (xorexp a b)
           a))

(define (syn sketch spec)
  (define M
    (synthesize
     #:forall (list x)
     #:guarantee (assert (eq? (interpret sketch) (interpret spec)))))
  (if (sat? M)
      (evaluate sketch M)
      false))

(check-equal? (syn (??expr (list x y false)) x) x)
(check-equal? (syn (??expr (list x y)) (notexp x)) (xorexp #t x))
(check-equal? (syn (??expr (list x y)) true) (iffexp #t #t))