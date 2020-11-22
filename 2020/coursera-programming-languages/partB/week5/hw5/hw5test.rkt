#lang racket
;; Programming Languages Homework 5 Simple Test
;; Save this file to the same directory as your homework file
;; These are basic tests. Passing these tests does not guarantee that your code will pass the actual homework grader

;; Be sure to put your homework file in the same folder as this test file.
;; Uncomment the line below and, if necessary, change the filename
(require "hw5.rkt")

(require rackunit)

(define tests
  (test-suite
   "Sample tests for Assignment 5"
   
   ;; check racketlist to mupllist with normal list
   (check-equal? (racketlist->mupllist '()) (aunit) "racketlist->mupllist test1")
   (check-equal? (racketlist->mupllist (list (int 3))) (apair (int 3) (aunit)) "racketlist->mupllist test2")
   (check-equal? (racketlist->mupllist (list (int 3) (int 4))) (apair (int 3) (apair (int 4) (aunit))) "racketlist->mupllist test3")
   
   ;; check mupllist to racketlist with normal list
   (check-equal? (mupllist->racketlist (aunit)) '() "racketlist->mupllist test1")
   (check-equal? (mupllist->racketlist (apair (int 3) (aunit))) (list (int 3)) "racketlist->mupllist test2")
   (check-equal? (mupllist->racketlist (apair (int 3) (apair (int 4) (aunit)))) (list (int 3) (int 4)) "racketlist->mupllist test3")
   (check-equal? (mupllist->racketlist (apair (int 3) (apair (int 4) (apair (int 5) (aunit))))) (list (int 3) (int 4) (int 5))
                 "racketlist->mupllist test4")

   (check-equal? (eval-exp (int 1)) (int 1) "int test")
   (check-equal? (eval-exp (add (int 5) (int 10))) (int 15) "should return int 15 when adding 5 and 10")
   
   ;; tests if ifgreater returns (int 2)
   (check-equal? (eval-exp (ifgreater (int 3) (int 4) (int 3) (int 2))) (int 2) "ifgreater test")
   (check-equal? (eval-exp (ifgreater (add (int 1) (int 2)) (int 3) (int 1) (int 5))) (int 5)
                 "Should return 5 because is not strictly greater")
   
   ;; mlet test
   (check-equal? (eval-exp (mlet "x" (int 1) (add (int 5) (var "x")))) (int 6) "mlet test1")
   (check-equal? (eval-exp (mlet "x" (int 1) (add (var "x") (var "x")))) (int 2) "mlet test2")
   
   ;; call test
   (check-equal? (eval-exp (call (closure '() (fun #f "x" (add (var "x") (int 7)))) (int 1))) (int 8) "call test")
   (check-equal? (eval-exp (call (fun #f "x" (int 7)) (int 1))) (int 7) "Should return 7")
   (check-equal? (eval-exp (call (fun #f "x" (add (var "x") (int 7))) (int 1))) (int 8) "Should return 8")
   (check-equal? (eval-exp (call (fun "count" "x"
                                      (ifgreater (var "x") (int 5)
                                                 (int 2)
                                                 (call (var "count") (add (var "x") (int 1)))))
                                      (int 1))) (int 2) "Recursive call")

   ;; pair test
   (check-equal? (eval-exp (apair (add (int 1) (int 2)) (int 3))) (apair (int 3) (int 3)) "Should return a new pair")

   ;; fst test
   (check-equal? (eval-exp (fst (apair (int 1) (int 2)))) (int 1) "fst test")
   
   ;; snd test
   (check-equal? (eval-exp (snd (apair (int 1) (int 2)))) (int 2) "snd test")
   
   ;; isaunit test
   (check-equal? (eval-exp (isaunit (closure '() (fun #f "x" (aunit))))) (int 0) "isaunit test")
   (check-equal? (eval-exp (isaunit (aunit))) (int 1) "isaunit test")
   
   ;; ifaunit test
   (check-equal? (eval-exp (ifaunit (int 1) (int 2) (int 3))) (int 3) "ifaunit test1")
   (check-equal? (eval-exp (ifaunit (aunit) (add (int 2) (int 3)) (int 3))) (int 5) "ifaunit test2")
   
   ;; mlet* test
   (check-equal? (eval-exp (mlet* (list (cons "x" (int 10))) (var "x"))) (int 10) "mlet* test")
   (check-equal? (eval-exp (mlet* (list (cons "x" (int 10)) (cons "y" (int 1))) (add (var "x") (var "y")))) (int 11)
                 "testing with two vars")
   
   ;; ifeq test
   (check-equal? (eval-exp (ifeq (int 1) (int 2) (int 3) (int 4))) (int 4) "ifeq test1")
   (check-equal? (eval-exp (ifeq (int 2) (int 2) (int 3) (int 4))) (int 3) "ifeq test2")
   (check-equal? (eval-exp (ifeq (int 3) (int 2) (int 3) (int 4))) (int 4) "ifeq test3")
   (check-equal? (eval-exp (ifeq (int 2) (int 3) (int 3) (int 4))) (int 4) "ifeq test4")
   (check-equal? (eval-exp (ifeq (add (int 3) (int 1)) (add (int 2) (int 2)) (add (int 3) (int 2)) (int 4))) (int 5) "ifeq test5")
   
   ;; mupl-map test
   (check-equal? (eval-exp (call (call mupl-map (fun #f "x" (add (var "x") (int 7)))) (apair (int 1) (aunit)))) 
                 (apair (int 8) (aunit)) "mupl-map test")
   
   ;; problems 1, 2, and 4 combined test
   (check-equal? (mupllist->racketlist
   (eval-exp (call (call mupl-mapAddN (int 7))
                   (racketlist->mupllist 
                    (list (int 3) (int 4) (int 9)))))) (list (int 10) (int 11) (int 16)) "combined test")
   
   ))

(require rackunit/text-ui)
;; runs the test
(run-tests tests)
