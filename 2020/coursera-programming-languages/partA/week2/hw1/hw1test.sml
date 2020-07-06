val test_is_older1 = is_older ((2011,11,10),(2011,11,10)) = false
val test_is_older2 = is_older ((2010,11,10),(2011,11,10)) = true
val test_is_older3 = is_older ((2011,10,10),(2011,11,10)) = true
val test_is_older4 = is_older ((2011,11,9),(2011,11,10)) = true
val test_is_older5 = is_older ((2011,10,9),(2012,3,4)) = true
val test_is_older6 = is_older ((5,4,4),(4,5,4)) = false
val test_is_older7 = is_older ((2020,11,30),(2020,12,1))= true
                                                      
val test_number_in_month1 = number_in_month([(2011,1,12),(2010,1,31),(1992,2,3)],2) = 1
val test_number_in_month2 = number_in_month([(2011,1,12),(2010,1,31),(1992,2,3)],1) = 2

val test_number_in_months1 = number_in_months([(2011,1,12),(2011,2,11),(2013,3,11)],[1,2,3]) = 3

val test_dates_in_month1 = dates_in_month([(2011,1,12),(2011,3,10),(2012,2,14),(1992,3,1)], 3) =
                          [(2011,3,10),(1992,3,1)]
                                                                                                  
val test_dates_in_months1 = dates_in_months([(2011,1,12), (2011,3,10), (2012,2,14), (1992,3,1)],[1,2,3]) =
                            [(2011,1,12),(2012,2,14), (2011,3,10), (1992,3,1)]

val test_get_nth1 = get_nth(["abc","cde","fg"],2) = "cde"

val test_date_to_string1 = date_to_string((2012,1,12)) = "January 12, 2012"

val test_number_before_reaching_sum1 = number_before_reaching_sum(3,[1,2,3]) = 1
val test_number_before_reaching_sum2 = number_before_reaching_sum(20,[1,2,3,4]) = 4

val test_what_month1 = what_month(21) = 1
val test_what_month2 = what_month(64) = 3

val test_month_range1 = month_range((21,22)) = [1,1]                                            
val test_month_range2 = month_range((5,3)) = []
                                                   
val test_oldest1 = oldest([(2020,8,1),(2020,10,20),(2020,12,31),(2020,1,1)]) = SOME (2020,1,1)

val test_check_exist1 = check_exist(1, [1,2,3]) = true
val test_check_exist2 = check_exist(2, [1,1,2,3,1]) = true
val test_check_exist3 = check_exist(3, [1,2,1]) = false

val test_remove_duplicates = remove_duplicates([1,2,3,1]) = [2,3,1]

val test_number_in_months_challenge1 = number_in_months_challenge(
        [(2011,1,12),(2011,2,11),(2013,3,11)],[1,2,3,1,1,2]) = 3
                                                                                    
val test_dates_in_months_challenge1 = dates_in_months_challenge(
        [(2011,1,12), (2011,3,10), (2012,2,14), (1992,3,1)], [1,2,3,3,2,1,2]) =
        [(2011,3,10),(1992,3,1),(2011,1,12),(2012,2,14)]                                      

val test_reasonable_date1 = reasonable_date((2020,2,29)) = true
val test_reasonable_date2 = reasonable_date((2010,2,29)) = false
val test_reasonable_date3 = reasonable_date((2010,2,28)) = true
val test_reasonable_date4 = reasonable_date((2008,3,15)) = true
