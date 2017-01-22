#include <stdio.h>
#include <stdlib.h>

/*
 * Goal: we want to find out the number of recursive calls
 * made by the Fibonacci routine to answer the following 
 * questions:
 * 
 * MAW 3.24 If the recursive routine in Section 2.4 used to 
 * compute Fibonacci numbers is run for N = 50, is stack space
 * likely to run out? Why or why not?
 */

unsigned long count; // count the number of recursive calls

unsigned long
Fib(int N)
{
  count++;
  if (N <= 1)
    return 1;
  else
    return Fib(N-1) + Fib(N-2);
}

int
main(int agrc, char* argv[])
{
  unsigned long i, c, d;
  int numRepeat = atoi(argv[1]);
  printf("i    \t \tFib(i)\t \tnumCalls\n");
  for(i = 0; i < numRepeat; i++)
  {
    count = 0;
    d = Fib(i);
    c = count;
    printf("i = %lu\t \t%lu\t \t%lu\n", i, d, c);
  }
  return 0;
}
