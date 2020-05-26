(* Dan Grossman, Coursera PL, HW2 Provided Code *)

(* if you use this function to compare two strings (returns true if the same
   string), then you avoid several of the functions in problem 1 having
   polymorphic types that may be confusing *)
fun same_string(s1 : string, s2 : string) =
    s1 = s2

(* put your solutions for problem 1 here *)
(* 1.(a) *)
fun all_except_option(s1, sl1) =
    case (s1, sl1) of
        (_,[]) => NONE
      | (x, hd::tl) => if same_string(x,hd) then SOME tl
                       else case all_except_option(x,tl) of
                                NONE => NONE
                              | SOME p => SOME (hd::p)
(* 1.(b) *)
fun get_substitutions1(sll, s) =
    case (sll, s) of
        ([], _) => []
      | (hd::tl, s) => case all_except_option(s, hd) of
                           NONE => get_substitutions1(tl, s)
                         | SOME l => l @ get_substitutions1(tl, s)

(* 1.(c) *)
fun get_substitutions2(sll, s) =
    let fun helper(sll, s, acc) =
            case (sll, s, acc) of
                ([], _, acc) => acc
              | (hd::tl, s, acc) => case all_except_option(s, hd) of
                                        NONE => helper(tl, s, acc)
                                      | SOME l => helper(tl, s, l @ acc)
    in
        helper(sll, s, [])
    end

(* 1.(d) *)
fun similar_names(sll, fullname) =
    let fun helper(firstnames, fullname, acc) =
            case (firstnames, fullname, acc) of
                ([], fullname, acc) => [fullname] @ acc
              | (hd::tl, {first=x, middle=y, last=z}, acc) => helper(tl, {first=x, middle=y, last=z}, acc @ [{first=hd, middle=y, last=z}])
    in
        case (sll, fullname) of
            ([], _) => [fullname]
          | (sll,  {first=x, middle=y, last=z}) => helper(get_substitutions1(sll, x), {first=x, middle=y, last=z}, [])
    end
        
(* you may assume that Num is always used with values 2, 3, ..., 10
   though it will not really come up *)
datatype suit = Clubs | Diamonds | Hearts | Spades
datatype rank = Jack | Queen | King | Ace | Num of int 
type card = suit * rank

datatype color = Red | Black
datatype move = Discard of card | Draw 

exception IllegalMove

(* put your solutions for problem 2 here *)
(* 2.(a) *)
fun card_color(s, r) =
    case (s, r) of
        (Spades, _) => Black
      | (Clubs, _) => Black
      | (Diamonds, _) => Red
      | (Hearts, _) => Red

(* 2.(b) *)             
fun card_value(s, r) =
    case (s, r) of
        (_, Num x) => x
      | (_, Ace) => 11
      | (_, _) => 10

(* 2.(c) *)        
fun remove_card(cs, c, e) =
    case (cs, c, e) of
        ([], _, e) => raise e
      | (hd::tl, c, e) => if hd = c then tl
                          else hd::remove_card(tl, c, e)
                                              
(* 2.(d) *)
fun all_same_color(cs) =
    let fun helper(cardColor, cs) =
            case (cardColor, cs) of
                (cardColor, []) => true
              | (cardColor, (s,r)::tl) => if cardColor <> card_color(s,r) then false
                                          else helper(cardColor, tl)
    in
        case (cs) of
            ([]) => true
          | ((s,r)::tl) => helper(card_color(s,r), tl)
    end

(* 2.(e) *)
fun sum_cards(cs) =
    let fun helper(cs, acc) =
            case (cs, acc) of
                ([], acc) => acc
              | ((s,r)::tl, acc) => helper(tl, card_value(s,r) + acc)
    in
        helper(cs, 0)
    end

(* 2.(f) *)
fun score(cs, goal) =
    let val p = sum_cards(cs)
        val prelim = if (p > goal) then (p - goal) * 3
                 else (goal - p)

    in
        if all_same_color(cs) then prelim div 2
        else prelim
    end
            
(* 2.(g) *)
fun officiate(cs, ms, goal) =
    let fun helper(cs, ms, goal, hcs) =
            case (cs, ms, goal, hcs) of
                (_, [], goal, hcs) => score(hcs, goal)
              | (cs, Discard c::mtl, goal, _) => helper(cs, mtl, goal, remove_card(hcs, c, IllegalMove))
              | ([], Draw::mtl, goal, hcs) => score(hcs, goal)
              | (chd::ctl, Draw::mtl, goal, hcs) => if sum_cards(chd::hcs) > goal then score(chd::hcs, goal)
                                                    else helper(ctl, mtl, goal, chd::hcs)
    in
        helper(cs, ms, goal, [])
    end
