(* takes two dates and evaluates to true or false.  It evaluates to true if
   the first argument is a date that comes before the second argument.
   (If the two dates are the same, the result is false.) *)
fun is_older(date1 : int*int*int, date2 : int*int*int) =
    #1 date1 < #1 date2 orelse #2 date1 < #2 date2 orelse #3 date1 < #3 date2

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
        
        

                            
                      
        
              

              
