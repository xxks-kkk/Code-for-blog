(* Coursera Programming Languages, Homework 3, Provided Code *)

exception NoAnswer

datatype pattern = Wildcard
		 | Variable of string
		 | UnitP
		 | ConstP of int
		 | TupleP of pattern list
		 | ConstructorP of string * pattern

datatype valu = Const of int
	      | Unit
	      | Tuple of valu list
	      | Constructor of string * valu

fun g f1 f2 p =
    let 
	val r = g f1 f2 
    in
	case p of
	    Wildcard          => f1 ()
	  | Variable x        => f2 x
	  | TupleP ps         => List.foldl (fn (p,i) => (r p) + i) 0 ps
	  | ConstructorP(_,p) => r p
	  | _                 => 0
    end

(**** for the challenge problem only ****)

datatype typ = Anything
	     | UnitT
	     | IntT
	     | TupleT of typ list
	     | Datatype of string

(**** you can put all your code here ****)
(*1*)
val only_capitals = List.filter (fn x => Char.isUpper (String.sub (x,0)));
(*2*)
val longest_string1 = List.foldl (fn (x, max_str) => if String.size x > String.size max_str then x else max_str)
                                 ""
(*3*)
val longest_string2 = List.foldl (fn (x, max_str) => if String.size x >= String.size max_str then x else max_str)
                                 ""
(*4*)
fun longest_string_helper f = List.foldl
   (fn (x, max_str) => if f(String.size x, String.size max_str) then x else max_str) ""
                                         
val longest_string3 = longest_string_helper (fn (x,y) => x > y)
val longest_string4 = longest_string_helper (fn (x,y) => x >= y)
(*5*)
val longest_capitalized = longest_string1 o only_capitals
(*6*)
val rev_string = implode o List.rev o explode
(*7*)
fun first_answer f xs =
    case (f,xs) of
        (_, []) => raise NoAnswer
     |  (f, hd::tl) => case f hd of
                            NONE => first_answer f tl
                         | SOME v => v 
(*8*)
fun all_answers f xs =
    let fun helper (xs,acc) =
            case (f,xs,acc) of
                (f,[],[]) => SOME []
              | (f,[],acc) => SOME acc
              | (f,hd::tl,acc) => case f hd of
                                      NONE => NONE
                                    | SOME v => helper (tl,v@acc)
    in
        helper (xs,[])
    end
(*9.(a)*)
val count_wildcards = g (fn x => 1) (fn p => 0)
(*9.(b)*)
val count_wild_and_variable_lengths = g (fn x => 1) (fn x => String.size x)
(*9.(c)*)
fun count_some_var (s,p) = g (fn x => 0) (fn p => if p = s then 1 else 0) p
(*10*)
fun check_pat p =
    let fun get_str_list p =
            let fun helper (p, acc) =
                case (p, acc) of
                    (Variable x, acc) => x::acc
                  | (TupleP ps, acc) => (List.foldl helper acc ps)
                  | (ConstructorP(_,p),acc) => helper(p,acc)
                  | _ => acc
            in
                helper(p, [])
            end
        fun check_unique xs =
            case xs of
                [] => true
              | hd::tl => case List.exists (fn x => x = hd) tl of
                              true => false
                            | false => check_unique tl
    in
        (check_unique o get_str_list) p
    end
(*11*)
fun match (v,p) =
    case (v,p) of 
        (_, Wildcard) => SOME []
      | (v, Variable s) => SOME [(s,v)]
      | (Unit, UnitP) => SOME []
      | (Const x, ConstP y) => if x = y then SOME [] else NONE
      | (Tuple vs, TupleP ps) => if length vs = length ps then
                                     all_answers (fn (x,y) => match(x,y)) (ListPair.zip(vs, ps))
                                 else NONE
      | (Constructor(s2,v),ConstructorP(s1,p)) => if s1=s2 then match(v,p) else NONE 
      | _ => NONE
(*12*)
fun first_match v ps =
    (case first_answer (fn p => match (v,p)) ps of v => SOME v) handle NoAnswer => NONE
