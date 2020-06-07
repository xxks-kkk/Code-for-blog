(* Homework3 Simple Test*)
(* These are basic test cases. Passing these tests does not guarantee that your code will pass the actual homework grader *)
(* To run the test, add a new line to the top of this file: use "homeworkname.sml"; *)
(* All the tests should evaluate to true. For example, the REPL should say: val test1 = true : bool *)

val test1_only_capitals = only_capitals ["A","B","C"] = ["A","B","C"]
val test2_only_capitals = only_capitals ["a", "B", "c", "D"] = ["B", "D"]
val test3_only_capitals = only_capitals [] = []                                              
                                              
val test1_longest_string1 = longest_string1 ["A","bc","C"] = "bc"
val test2_longest_string1 = longest_string1 [] = ""
val test3_longest_string1 = longest_string1 ["A", "bc", "de", "D"] = "bc"

val test1_longest_string2 = longest_string2 ["A","bc","C"] = "bc"
val test2_longest_string2 = longest_string2 ["A", "bc", "de", "D"] = "de"
                                                                         
val test1_longest_string3 = longest_string3 ["A","bc","C"] = "bc"
val test2_longest_string3 = longest_string3 ["A", "bc", "de", "D"] = "bc"
val test3_longest_string3 = longest_string3 [] = ""

val test1_longest_string4 = longest_string4 ["A","bc","C"] = "bc"
val test3_longest_string4 = longest_string4 [] = ""                                                                
val test2_longest_string4 = longest_string4 ["A", "bc", "de", "D"] = "de"

val test1_longest_capitalized = longest_capitalized ["A","bc","C"] = "A"
val test2_longest_capitalized = longest_capitalized [] = ""
                                                             
val test1_rev_string = rev_string "abc" = "cba"
val test2_rev_string = rev_string "" = ""                                              

val test1_first_answer = first_answer (fn x => if x > 3 then SOME x else NONE) [1,2,3,4,5] = 4
val test2_first_answer = ((first_answer (fn x => if x > 6 then SOME x else NONE) [1,2,3,4,5]; false) handle NoAnswer => true)

val test1_all_answers = all_answers (fn x => if x = 1 then SOME [x] else NONE) [2,3,4,5,6,7] = NONE
val test2_all_answers = all_answers (fn x => if x > 1 then SOME [x] else NONE) [2,3,4,5,6,7] = SOME [7,6,5,4,3,2]                                                                                                   
val test1_count_wildcards = count_wildcards Wildcard = 1
val test2_count_wildcards = count_wildcards (TupleP [Wildcard, Wildcard, Wildcard, Variable "s"]) = 3 

val test1_count_wild_and_variable_lengths = count_wild_and_variable_lengths (Variable("a")) = 1
val test2_count_wild_and_variable_lengths = count_wild_and_variable_lengths (TupleP [Wildcard,
                                                                                     Wildcard,
                                                                                     Variable("a"),
                                                                                     Variable("abc")]) = 6
                                                                                                  
val test1_count_some_var = count_some_var ("x", Variable("x")) = 1
val test2_count_some_var = count_some_var ("x", TupleP [Variable("x"), Variable("x")]) = 2

(* test internal helper functions of check_pat *)
(* val test1_check_unique = check_unique ["x", "x"] = false *)
(* val test2_check_unique = check_unique ["x", "y"] = true *)

(* val test1_get_str_list = get_str_list (Variable("x")) = ["x"] *)
(* val test2_get_str_list = get_str_list (TupleP [Variable("x"), Variable("y")]) = ["y", "x"] *)
(* val test3_get_str_list = get_str_list (ConstructorP("name", TupleP [ TupleP [Variable("x")]])) = ["x"] *)

val test1_check_pat = check_pat (Variable("x")) = true
val test2_check_pat = check_pat (TupleP [Variable("x"), Variable("y")]) = true
val test3_check_pat = check_pat (TupleP [Variable("x"), Variable("x")]) = false
val test4_check_pat = check_pat (TupleP [Variable("x"), TupleP [Variable("x")]]) = false
val test5_check_pat = check_pat (TupleP [Variable("x"), TupleP [Variable("y")]]) = true
                                         
val test1_match = match (Const(1), UnitP) = NONE
val test2_match = match (Const(1), ConstP(12)) = NONE
val test3_match = match (Const(12), ConstP(12)) = SOME []                                                    

val test1_first_match = first_match Unit [UnitP] = SOME []


