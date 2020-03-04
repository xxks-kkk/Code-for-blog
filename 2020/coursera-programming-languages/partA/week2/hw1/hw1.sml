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

                    
                
        
            
        
        

                            
                      
        
              

              
