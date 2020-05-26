(* Homework2 Simple Test *)
(* These are basic test cases. Passing these tests does not guarantee that your code will pass the actual homework grader *)
(* To run the test, add a new line to the top of this file: use "homeworkname.sml"; *)
(* All the tests should evaluate to true. For example, the REPL should say: val test1 = true : bool *)

val test1_all_except_option = all_except_option ("string", ["string"]) = SOME []
val test2_all_except_option = all_except_option ("string", []) = NONE
val test3_all_except_option = all_except_option ("a", ["a", "b"]) = SOME ["b"]
val test4_all_except_option = all_except_option ("a", ["b", "c", "a", "d"]) = SOME ["b", "c", "d"]
val test5_all_except_option = all_except_option ("a", ["b", "c", "d"]) = NONE                                                                                   
                                                            
val test1_get_substitutions1 = get_substitutions1 ([["foo"],["there"]], "foo") = []
val test2_get_substitutions1 = get_substitutions1 ([["Fred", "Fredrick"], ["Elizabeth", "Betty"], ["Freddie", "Fred", "F"]], "Fred") = ["Fredrick", "Freddie", "F"]
val test3_get_substitutions1 = get_substitutions1 ([["Fred", "Fredrick"], ["Jeff", "Jeffrey"], ["Geoff", "Jeff", "Jeffrey"]], "Jeff") = ["Jeffrey", "Geoff", "Jeffrey"]

val test1_get_substitutions2 = get_substitutions1 ([["foo"],["there"]], "foo") = []
val test2_get_substitutions2 = get_substitutions1 ([["Fred", "Fredrick"], ["Elizabeth", "Betty"], ["Freddie", "Fred", "F"]], "Fred") = ["Fredrick", "Freddie", "F"]
val test3_get_substitutions2 = get_substitutions1 ([["Fred", "Fredrick"], ["Jeff", "Jeffrey"], ["Geoff", "Jeff", "Jeffrey"]], "Jeff") = ["Jeffrey", "Geoff", "Jeffrey"]

val test1_similar_names = similar_names ([["Fred","Fredrick"],["Elizabeth","Betty"],["Freddie","Fred","F"]], {first="Fred", middle="W", last="Smith"}) =
	    [{first="Fred", last="Smith", middle="W"}, {first="Fredrick", last="Smith", middle="W"},
	     {first="Freddie", last="Smith", middle="W"}, {first="F", last="Smith", middle="W"}]

val test1_card_color = card_color (Clubs, Num 2) = Black
val test2_card_color = card_color (Hearts, Num 9) = Red

val test1_card_value = card_value (Clubs, Num 2) = 2
val test2_card_value = card_value (Spades, Ace) = 11
val test3_card_value = card_value (Diamonds, Queen) = 10

val test1_remove_card = remove_card ([(Hearts, Ace)], (Hearts, Ace), IllegalMove) = []
val test2_remove_card = remove_card ([(Hearts, Ace), (Hearts, Num 9)], (Hearts, Ace), IllegalMove) = [(Hearts, Num 9)]
val test3_remove_card = remove_card ([(Spades, Ace), (Hearts, Num 9), (Clubs, Num 8), (Hearts, Ace), (Clubs, Num 8)], (Clubs, Num 8), IllegalMove) =
                        [(Spades, Ace), (Hearts, Num 9), (Hearts, Ace), (Clubs, Num 8)]
                                                                                                                                                  
val test1_all_same_color = all_same_color [(Hearts, Ace), (Hearts, Ace)] = true
val test2_all_same_color = all_same_color [(Hearts, Num 8), (Spades, Ace)] = false

val test1_sum_cards = sum_cards [(Clubs, Num 2),(Clubs, Num 2)] = 4
val test2_sum_cards = sum_cards [(Clubs, Num 4),(Spades, Ace),(Diamonds, King)] = 25

val test1_score = score ([(Hearts, Num 2),(Clubs, Num 4)],10) = 4

val test1_officiate = officiate ([(Hearts, Num 2),(Clubs, Num 4)],[Draw], 15) = 6

val test2_officiate = officiate ([(Clubs,Ace),(Spades,Ace),(Clubs,Ace),(Spades,Ace)],
                        [Draw,Draw,Draw,Draw,Draw],
                        42) = 3
val test3_officiate = ((officiate([(Clubs,Jack),(Spades,Num(8))],
                         [Draw,Discard(Hearts,Jack)],
                         42);false) handle IllegalMove => true)
             
             

