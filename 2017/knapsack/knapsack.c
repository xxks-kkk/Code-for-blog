#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/* 
 * [Problem]:
 * In our Materials Marketplace we have many different companies all looking 
 * for the same material. However, each company is looking for a specific 
 * quantity of the material at a price they set themselves. 
 * A company approaches our team with a large amount of that material, 
 * but not enough to complete every request for it. 
 * Given the total amount of the material they have, 
 * the company asks us to find out what companies they should sell to in order 
 * to maximize their profits. This scenario happens frequently so we need to 
 * be able to compute the answer relatively quickly and with minimal processing power.
 * 
 * Company| A  | B  | C  | D  | E  | F  | G  | H  | I  | J  |
 * -------+----+----+----+----+----+----+----+----+----+----+
 * Amount | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 10 |
 * -------+----+----+----+----+----+----+----+----+----+----+
 * Price  | 1  | 5  | 8  | 9  | 10 | 17 | 17 | 20 | 24 | 30 | 
 * -------+----+----+----+----+----+----+----+----+----+----+
 *
 * [Reference]: 
 * 1. http://cse.unl.edu/~goddard/Courses/CSCE310J/Lectures/Lecture8-DynamicProgramming.pdf
 * 2. https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/knapsack_01.html (some test cases)
 */

/* Utility function */
void printList(int*, int);

/* Test cases */
void testCase1();
void testCase2();
void testCase3();

/* The problem can be treated as 0/1 knapsack problem. */
int* knapsack(int maxWeight, int numItems, int** elements, int* resultLength);

int
main(void)
{
  testCase1(); 
  testCase2();
  testCase3();
  return 0;
}

/*
 * The procedure solves the problem above.
 *
 * [Return]:
 *  - The procedure returns an array of companies that maximizes
 *    our profit
 *
 * [Parameters]:
 *  - maxWeight : the amount of material we have
 *  - numItems  : the number of companies we receive the requests from
 *  - elements  : 2D array that have two rows: first row contains the
 *                amount of material that each company needs and second
 *                row contains the corresponding price that each company wants
 *                to pay
 *  - resultLength : number of companies we choose to maximize our profit
 */
int*
knapsack(int maxWeight, int numItems, int** elements, int* resultLength)
{
  int** table = malloc(sizeof(int*) * (numItems+1));
  int w, i, w_i, b_i, k;
  for(i = 0; i <= numItems; i++)
  {
    table[i] = malloc(sizeof(int) * (maxWeight+1));
  }
  for(w = 0; w <= maxWeight; w++)
    table[0][w] = 0;
  for(i = 1; i <= numItems; i++)
    table[i][0] = 0;
  for(i = 1; i <= numItems; i++)
  {
    for(w = 1; w <= maxWeight; w++)
    {
      w_i = elements[0][i-1];
      b_i = elements[1][i-1];
      if(w_i <= w)
      {
        if(b_i + table[i-1][w-w_i] > table[i-1][w])
          table[i][w] = b_i + table[i-1][w-w_i];
        else
          table[i][w] = table[i-1][w];
      }
      else
      {
        table[i][w] = table[i-1][w];
      }
    }
  }
  // max profit: table[numItems][maxWeight];
  // now we need to find out exactly composition of the knapsack items
  int *result = malloc(sizeof(int) * numItems);
  i = numItems;
  k = maxWeight;
  int counter = 0; 
  while (i > 0 && k > 0)
  {
    if(table[i][k] != table[i-1][k])
    {
      result[counter] = i; // we mark the ith item as in the knapsack
      counter++;
      i--;
      k -= elements[0][i];
    }
    else
    {
      i--;
    }
  }
  *resultLength = counter;
  for(i=0; i <= numItems; i++)
  {
    free(table[i]);
  }
  free(table);
  return result;
}

void
printList(int* result, int resultLength)
{
  int i;
  printf("Company we choose: ");
  for (i = resultLength-1; i >= 0; i--)
  {
    printf("%c, %s", result[i]+64, (i == 0) ? ("\n") : (""));
  }
}

/*
 * Company| A  | B  | C  | D  |
 * -------+----+----+----+----+
 * Amount | 2  | 3  | 4  | 5  |
 * -------+----+----+----+----+
 * Price  | 3  | 4  | 5  | 6  |
 * -------+----+----+----+----+
 */
void
testCase1()
{
  printf("%s\n", "TEST CASE 1");

  // Create test data
  int maxWeight, numItems;
  maxWeight = 5;
  numItems = 4;
  int** elements = malloc(sizeof(int*) * 2);
  int i, j;
  for(i = 0 ; i < 2; i++)
  {
    elements[i] = malloc(sizeof(int) * numItems);
  }
  elements[0][0] = 2;
  elements[0][1] = 3;
  elements[0][2] = 4;
  elements[0][3] = 5;
  elements[1][0] = 3;
  elements[1][1] = 4;
  elements[1][2] = 5;
  elements[1][3] = 6;

  // Get the solution
  int resultLength;
  int* result = knapsack(maxWeight,numItems,elements, &resultLength);

  // Add assertion
  assert(resultLength == 2);
  assert(result[0] == 2); // B
  assert(result[1] == 1); // A

  // Printout the solution
  printList(result, resultLength);
  printf("%s\n\n", "pass");
  
  // Cleanup
  free(result);
  for(i = 0; i < 2; i++)
  {
    free(elements[i]);
  }
  free(elements);
}

/*
 * Company| A  | B  | C  | D  | E  | F  | G  | H  | I  | J  |
 * -------+----+----+----+----+----+----+----+----+----+----+
 * Amount | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 10 |
 * -------+----+----+----+----+----+----+----+----+----+----+
 * Price  | 1  | 5  | 8  | 9  | 10 | 17 | 17 | 20 | 24 | 30 | 
 * -------+----+----+----+----+----+----+----+----+----+----+
 */
void
testCase2()
{
  printf("%s\n", "TEST CASE 2");

  // Create test data
  int maxWeight, numItems;
  maxWeight = 30;
  numItems = 10;
  int** elements = malloc(sizeof(int*) * 2);
  int i, j;
  for(i = 0 ; i < 2; i++)
  {
    elements[i] = malloc(sizeof(int) * numItems);
  }
  elements[0][0] = 1;
  elements[0][1] = 2;
  elements[0][2] = 3;
  elements[0][3] = 4;
  elements[0][4] = 5;
  elements[0][5] = 6;
  elements[0][6] = 7;
  elements[0][7] = 8;
  elements[0][8] = 9;
  elements[0][9] = 10;
  elements[1][0] = 1;
  elements[1][1] = 5;
  elements[1][2] = 8;
  elements[1][3] = 9;
  elements[1][4] = 10;
  elements[1][5] = 17;
  elements[1][6] = 17;
  elements[1][7] = 20;
  elements[1][8] = 24;
  elements[1][9] = 30;

  // Get the solution
  int resultLength;
  int* result = knapsack(maxWeight,numItems,elements, &resultLength);

  // Add assertion
  assert(resultLength == 5);
  assert(result[0] == 10); // B
  assert(result[1] == 9);  // C
  assert(result[2] == 6);  // F
  assert(result[3] == 3);  // I  
  assert(result[4] == 2);  // J
  
  // Printout the solution
  printList(result, resultLength);
  printf("%s\n\n", "pass");
  
  // Cleanup
  free(result);
  for (i = 0; i < 2; i++)
  {
    free(elements[i]);
  }
  free(elements);
}

/*
 * | Company | A  | B  | C  | D  | E  | F  | G  | H  | I  | J  |
 * |---------|----|----|----|----|----|----|----|----|----|----|
 * | Amount  | 23 | 31 | 29 | 44 | 53 | 38 | 63 | 85 | 89 | 82 |
 * | Price   | 92 | 57 | 49 | 68 | 60 | 43 | 67 | 84 | 87 | 72 |
 */
void
testCase3()
{
  printf("%s\n", "TEST CASE 2");

  // Create test data
  int maxWeight, numItems;
  maxWeight = 165;
  numItems = 10;
  int** elements = malloc(sizeof(int*) * 2);
  int i, j;
  for(i = 0 ; i < 2; i++)
  {
    elements[i] = malloc(sizeof(int) * numItems);
  }
  elements[0][0] = 23;
  elements[0][1] = 31;
  elements[0][2] = 29;
  elements[0][3] = 44;
  elements[0][4] = 53;
  elements[0][5] = 38;
  elements[0][6] = 63;
  elements[0][7] = 85;
  elements[0][8] = 89;
  elements[0][9] = 82;
  elements[1][0] = 92;
  elements[1][1] = 57;
  elements[1][2] = 49;
  elements[1][3] = 68;
  elements[1][4] = 60;
  elements[1][5] = 43;
  elements[1][6] = 67;
  elements[1][7] = 84;
  elements[1][8] = 87;
  elements[1][9] = 72;

  // Get the solution
  int resultLength;
  int* result = knapsack(maxWeight,numItems,elements, &resultLength);

  // Add assertion
  assert(resultLength == 5);
  assert(result[0] == 6); // F
  assert(result[1] == 4); // D
  assert(result[2] == 3); // C
  assert(result[3] == 2); // B
  assert(result[4] == 1); // A
  
  // Printout the solution
  printList(result, resultLength);
  printf("%s\n\n", "pass");
  
  // Cleanup
  free(result);
  for (i = 0; i < 2; i++)
  {
    free(elements[i]);
  }
  free(elements);
}
