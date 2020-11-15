
#lang racket

(provide (all-defined-out)) ;; so we can put tests in a second file

;; put your code below

(define (sequence low high stride)
  (cond [(> low high) null]
        [#t (cons low (sequence (+ low stride) high stride))]))

(define (string-append-map xs suffix)
  (map (lambda (str) (string-append str suffix)) xs))

(define (list-nth-mod xs n)
  (cond [(< n 0) (error "list-nth-mod: negative number")]
        [(null? xs) (error "list-nth-mod: empty list")]
        [#t (car (list-tail xs (remainder n (length xs))))]))

(define (stream-for-n-steps s n)
  (cond [(= n 0) null]
        [#t (let ([res (s)])
              (cons (car res) (stream-for-n-steps (cdr res) (- n 1))))]))

; Question: why the body is (f 1) instead of (lambda () (f 1))?
(define (funny-number-stream)
  (define (f x)
      (cons
       (if (= (remainder x 5) 0)
           (* -1 x)
           x)
       (lambda () (f (+ x 1)))))
  (f 1))

(define dan-then-dog  
   (letrec ([f (lambda (x) (cons x (lambda () (f (if (equal? x "dog.jpg") "dan.jpg" "dog.jpg")))))])
    (lambda () (f "dan.jpg"))))

(define (stream-add-zero s)
  (define tmp (s))
  (lambda () (cons (cons 0 (car tmp)) (stream-add-zero (cdr tmp)))))

(define (cycle-lists xs ys)
  (define (f n) (cons (cons (list-nth-mod xs n) (list-nth-mod ys n)) (lambda () (f (+ n 1)))))
  (lambda () (f 0)))

(define (vector-assoc v vec)
  (define (f n) (if (>= n (vector-length vec))
                    #f
                    (let ([current (vector-ref vec n)]
                          [next (lambda (x) (f (+ x 1)))])
                      (if (pair? current)
                          (if (equal? (car current) v) current (next n))
                          (next n)))))
  (f 0))

(define (cached-assoc xs n)
  (letrec ([memo (make-vector n #f)]
           [pos 0]
           [f (lambda (v)
                (let ([ans (vector-assoc v memo)])
                (if ans
                    ans
                    (let ([new-ans (assoc v xs)])
                      (if new-ans
                          (begin (vector-set! memo pos new-ans)
                                 (set! pos (remainder (add1 pos) n))
                                 new-ans)
                          #f)))))])
    f))