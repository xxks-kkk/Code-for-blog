fun is_older (date1 : int * int * int, date2 : int * int * int) =
    let
        val y1 = #1 date1
        val m1 = #2 date1
        val d1 = #3 date1
        val y2 = #1 date2
        val m2 = #2 date2
        val d2 = #3 date2
    in
        y1 < y2 orelse (y1=y2 andalso m1 < m2)
        orelse (y1=y2 andalso m1=m2 andalso d1 < d2)
    end 

fun number_in_month (dates : (int * int * int) list, month : int) =
    if null dates
    then 0
    else if #2 (hd dates) = month
    then 1 + number_in_month(tl dates, month)
    else number_in_month(tl dates, month) 

fun number_in_months(dates : (int * int * int) list, months : int list) =
    if null months
    then 0
    else number_in_month(dates, hd months) + number_in_months(dates, tl months)

fun dates_in_month (dates : (int * int * int) list, month : int) =
    if null dates
    then []
    else if #2 (hd dates) = month
    then (hd dates)::dates_in_month(tl dates, month)
    else dates_in_month(tl dates, month)

fun dates_in_months(dates : (int * int * int) list, months : int list) =
    if null months
    then []
    else dates_in_month(dates, hd months) @ dates_in_months(dates, tl months)

fun get_nth (lst : string list, n : int) =
    if n=1
    then hd lst
    else get_nth(tl lst, n-1)

fun date_to_string (date : int * int * int) =
    let
        val names = ["January", "February", "March", "April", "May", "June",
                     "July", "August", "September", "October", "November", "December"]
    in
        get_nth(names,#2 date) ^ " " ^ Int.toString(#3 date)
        ^ ", " ^ Int.toString(#1 date)
    end

fun number_before_reaching_sum (sum : int, lst : int list) =
    if sum <= hd lst
    then 0
    else 1 + number_before_reaching_sum(sum - hd lst, tl lst)

fun what_month (day_of_year : int) =
    let
        val month_lengths = [31,28,31,30,31,30,31,31,30,31,30,31]
    in
        1 + number_before_reaching_sum(day_of_year, month_lengths)
    end

fun month_range (day1 : int, day2 : int) =
    if day1 > day2
    then []
    else what_month day1 :: month_range(day1 + 1, day2)

fun oldest (dates : (int * int * int) list) =
    if null dates
    then NONE
    else let
        val ans = oldest(tl dates)
    in
        if isSome ans andalso is_older(valOf ans, hd dates)
        then ans
        else SOME (hd dates)
    end

(* arguably better alternate solution avoiding isSome / valOf *)
fun oldest (dates : (int * int * int) list) =
    if null dates
    then NONE
    else let
        fun f dates =
            if null (tl dates)
            then hd dates
            else let
                val ans = f (tl dates)
            in
                if is_older(ans, hd dates)
                then ans
                else hd dates
            end
    in
        SOME(f dates)
    end


(* quadratic algorithm rather than sorting which is nlog n *)
fun mem(x : int, xs : int list) =
    not (null xs) andalso (x = hd xs orelse mem(x, tl xs))

fun remove_duplicates(xs : int list) =
    if null xs
    then []
    else
        let
            val tl_ans = remove_duplicates (tl xs)
        in
            if mem(hd xs, tl_ans)
            then tl_ans
            else (hd xs)::tl_ans
        end

fun number_in_months_challenge(dates : (int * int * int) list, months : int list) =
    number_in_months(dates, remove_duplicates months)

fun dates_in_months_challenge (dates : (int * int * int) list, months : int list) =
    dates_in_months(dates, remove_duplicates months)

fun reasonable_date (date : int * int * int) =
    let
        fun get_nth (lst : int list, n : int) =
            if n=1
            then hd lst
            else get_nth(tl lst, n-1)
        val year  = #1 date
        val month = #2 date
        val day   = #3 date
        val leap  = year mod 400 = 0 orelse (year mod 4 = 0 andalso year mod 100 <> 0)
        val feb_len = if leap then 29 else 28
        val lengths = [31,feb_len,31,30,31,30,31,31,30,31,30,31]
    in
        year > 0 andalso month >= 1 andalso month <= 12
        andalso day >= 1 andalso day <= get_nth(lengths,month)
    end
