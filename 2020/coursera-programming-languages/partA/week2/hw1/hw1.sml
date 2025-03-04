(* takes two dates and evaluates to true or false.  It evaluates to true if
   the first argument is a date that comes before the second argument.
   (If the two dates are the same, the result is false.) *)
fun is_older(date1 : int*int*int, date2 : int*int*int) =
    let
        val date1_y = #1 date1
        val date2_y = #1 date2
        val date1_m = #2 date1
        val date2_m = #2 date2
        val date1_d = #3 date1
        val date2_d = #3 date2
    in
        (date1_y < date2_y)
        orelse ((date1_y = date2_y) andalso (date1_m < date2_m))
        orelse ((date1_y = date2_y) andalso (date1_m = date2_m) andalso (date1_d < date2_d))
    end

(* takes a list of dates and a month (i.e., an int) and returns
   how many dates in the list are in the given month *)
fun number_in_month(dates : (int*int*int) list, month : int) =
    let fun number_in_month_helper(dates: (int*int*int) list, month: int, acc : int) =
            if null dates
            then acc
            else
                if month = #2 (hd dates)
                then number_in_month_helper(tl dates, month, acc + 1)
                else number_in_month_helper(tl dates, month, acc)
    in
        number_in_month_helper(dates, month, 0)
    end

(* takes a list of dates and a list of months (i.e., an int list) and returns
   the number of dates in the list of dates that are in any of the months in the list of months.
   Assume the list of months has no number repeated. *)
fun number_in_months(dates: (int*int*int) list, months: int list) =
    let fun number_in_months_helper(dates: (int*int*int) list, months: int list, acc : int) =
            if null months
            then acc
            else number_in_months_helper(dates, tl months, number_in_month(dates, hd months) + acc)
    in
        number_in_months_helper(dates, months, 0)
    end

(* takes a list of dates and a month (i.e., an int) and returns a list holding the dates from the
   argument list of dates that are in the month.  The returned list should
   contain dates in the order they were originally given. *)
fun dates_in_month(dates: (int*int*int) list, month: int) =
    let fun dates_in_month_helper(dates: (int*int*int) list, month: int, acc: (int*int*int) list) =
            if null dates
            then acc
            else
                if #2 (hd dates) = month
                then dates_in_month_helper(tl dates, month, hd dates :: acc)
                else dates_in_month_helper(tl dates, month, acc)
        val l = dates_in_month_helper(dates, month, [])
    in
        rev l                                     
    end

(* takes a list of dates and a list of months (i.e., an int list) and
   returns a list holding the dates from the argument list of dates that
   are in any of the months in the list of months. Assume the list of
   months has no number repeated. *)
fun dates_in_months(dates: (int*int*int) list, months: int list) =
    let fun dates_in_months_helper(dates: (int*int*int) list, months: int list, acc: (int*int*int) list) =
            if null months
            then acc
            else dates_in_months_helper(dates, tl months, acc @ dates_in_month(dates, hd months))
    in
        dates_in_months_helper(dates, months, [])
    end
        
(* takes a list of strings and an int n and returns the n th element of
the list where the head of the list is 1st.  Do not worry about the
case where the list has too few elements:your function may
applyhdortlto the empty list in this case, which is okay *)
fun get_nth(xs: string list, n: int) =
    let fun get_nth_helper(xs: string list, n: int, cur: int) =
            if cur = n
            then hd xs
            else get_nth_helper(tl xs, n, cur + 1)
    in
        get_nth_helper(xs, n, 1)
    end

(* takes a date and returns a string of the form January 20, 2013 (for example).
   Use the operator ^ for concatenating strings and the library function Int.toString for
   converting an int to astring. For producing the month part, do not use a bunch of conditionals.
   Instead, use a list holding 12 strings and your answer to the previous problem. For consistency,
   put a comma following the day and use capitalized English month names: January, February, March,
   April, May, June, July, August, September, October, November, December. *)        
fun date_to_string(date: int*int*int) =
    let
        val months = ["January", "February", "March", "April", "May", "June", "July",
                      "August", "September", "October", "November", "December"]
    in
        get_nth(months, #2 date) ^ " " ^ Int.toString(#3 date) ^ ", " ^ Int.toString(#1 date)
    end

(* takes an int called sum, which you can assume is positive, and an int
   list, which you can assume contains all positive numbers, and returns
   an int. You should return an int n such that the first n elements of the
   list add to less than sum, but the first n+ 1 elements of the list add
   to sum or more. Assume the entire list sums to more than the passed
   in value; it is okay for an exception to occur if this is not the case. *)
fun number_before_reaching_sum(sum : int, xs : int list) =
    let fun number_before_reaching_sum_helper(sum : int, xs : int list, cur : int, idx : int) =
            if null xs orelse cur + hd xs >= sum
            then idx
            else number_before_reaching_sum_helper(sum, tl xs, cur + hd xs, idx + 1)
    in
        number_before_reaching_sum_helper(sum, xs, 0, 0)
    end

(* takes a day of year (i.e., an int between 1 and 365) and returns what month that day is in
   (1 for January, 2 for February, etc.). Use a list holding 12 integers and your answer to
   the previous problem. *)   
fun what_month(day: int) =
    let val days_in_month = [31,28,31,30,31,30,31,31,30,31,30,31]
    in
        1 + number_before_reaching_sum(day, days_in_month)
    end

(* takes two days of the year day1 and day2 and returns an int list
   [m1,m2,...,mn] where m1 is the month of day1, m2 is the month of
   day1+1, ..., and mn is the month of day day2. Note the result will
   have length day2 - day1 + 1 or length 0 if day1>day2 *)
fun month_range(days: int*int) =
    if #1 days > #2 days then []
    else if #1 days = #2 days
    then what_month(#1 days)::[]
    else what_month(#1 days)::month_range((#1 days + 1, #2 days))

(* takes a list of dates and evaluates to an (int*int*int) option.
   It evaluates to NONE if the list has no dates and SOME d if
   the date d is the oldest date in the list *)                                        
fun oldest(dates: (int*int*int) list) =
    if null dates then NONE
    else if null (tl dates) then SOME (hd dates)
    else
        let
            val date1 = hd dates
            val date2 = oldest(tl dates)
        in
            if isSome date2 andalso is_older(date1, valOf date2)
            then SOME date1
            else date2
        end

(* check whether a given int is in the int list *)
fun check_exist(x: int, xs: int list) =
    not (null xs) andalso (x = hd xs orelse check_exist(x, tl xs))

(* the idea is to construct a list with distinct elements
   from the xs, which potentially contains duplicates *)                              
fun remove_duplicates(xs: int list) =
    if null xs then []
    else
        let
            val tl_ans = remove_duplicates(tl xs)
        in
            if check_exist(hd xs, tl_ans)
            then tl_ans
            else (hd xs)::tl_ans
        end
    
(* takes a list of dates and a list of months (i.e., an int list) and returns
   the number of dates in the list dates that are in any of the months in the list
   of months. Note having a month in the second argument multiple times has
   no more effect than having it once *)
fun number_in_months_challenge(dates: (int*int*int) list, months: int list) = 
    number_in_months(dates, remove_duplicates months)

(* takes a list of dates and a list of months (an int list) and
   returns a list holding the date from the argument list of dates
   that are in any of the months in the list of months. Note having
   a month in the second argument multiple times has no more effect than
   having it once *)
fun dates_in_months_challenge(dates: (int*int*int) list, months: int list) = 
    dates_in_months(dates, remove_duplicates months)

(* takes a date and determines if it describes a real date in the common era.
   A "real date" has a positive year (year 0 did not exist), a month
   between 1 and 12, and a day appropriate for the month. Solutions should
   properly handle leap years. Leap years are years that are either divisible
   by 400 or divisible by 4 but not divisible by 100. Do not worry about
   days possibly lost in the conversion to the Gregorian calendar in the Late 1500s.*)                   
fun reasonable_date(date: int*int*int) =
    let
        val year = #1 date
        val month = #2 date
        val day = #3 date
        fun is_leap_year(year: int) =
            year mod 400 = 0 orelse (year mod 4 = 0 andalso year mod 100 <> 0)
    in
        if (year <= 0) orelse
           (month < 1 orelse month > 12) orelse
           (is_leap_year(year) andalso month = 2 andalso day > 29) orelse
           (not (is_leap_year(year)) andalso month = 2 andalso day > 28)
        then false
        else true
    end
            
               
