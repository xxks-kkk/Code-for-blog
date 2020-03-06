val test_is_older1 = is_older ((2011,11,10),(2011,11,10)) = false
val test_is_older2 = is_older ((2010,11,10),(2011,11,10)) = true
val test_is_older3 = is_older ((2011,10,10),(2011,11,10)) = true
val test_is_older4 = is_older ((2011,11,9),(2011,11,10)) = true
val test_is_older5 = is_older ((2011,10,9),(2012,3,4)) = true

val test_number_in_month1 = number_in_month([(2011,1,12),(2010,1,31),(1992,2,3)],2) = 1
val test_number_in_month2 = number_in_month([(2011,1,12),(2010,1,31),(1992,2,3)],1) = 2

val test_number_in_months1 = number_in_months([(2011,1,12),(2011,2,11),(2013,3,11)],[1,2,3]) = 3

val test_dates_in_month1 = dates_in_month([(2011,1,12),(2011,3,10),(2012,2,14),(1992,3,1)], 3) =
                          [(2011,3,10),(1992,3,1)]
                                                                                                  
val test_dates_in_months1 = dates_in_months([(2011,1,12), (2011,3,10), (2012,2,14), (1992,3,1)],[1,2,3]) =
                            [(2011,1,12),(2012,2,14), (2011,3,10), (1992,3,1)]

val test_get_nth1 = get_nth(["abc","cde","fg"],2) = "cde"
