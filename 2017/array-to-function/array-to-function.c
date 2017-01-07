#include <stdio.h>
#include <stdlib.h>

/* Print on console the array of int
 */
void print(int* array, int length);
void change(int *array,int length);
void change2(int *array, int length);

void test_change();
void test_change2();

int main(){

  test_change();  printf("\n");
  test_change2(); printf("\n");
  
  return 0;
}

void print(int* array, int length)
{
  int i;
  for(i = 0 ; i < length ; i++)
    printf("%d ", array[i]);
  printf("\n");
}

void change(int *array,int length)
{
  printf("array address inside function: %p\n", array);
  int i;
  for(i = 0 ; i < length ; i++)
    array[i] = 5;
}

void change2(int *array,int length)
{
  printf("array address inside function: %p\n", array);
  int i;
  int tmp[3] = {5,5,5};
  array = tmp;
}

void test_change()
{
  printf("TEST: change\n");
  int i, length = 3;
  int test[3] = {1,2,3};

  printf("Before:");
  print(test, length);
  printf("before change, test address: %p\n", test);
  change(test, 3);
  printf("After:");
  print(test, length);
  printf("after change, test address: %p\n", test);
}

void test_change2()
{
  printf("TEST: change2\n");
  int i, length = 3;
  int test[3] = {1,2,3};

  printf("Before:");
  print(test, length);
  printf("before change, test address: %p\n", test);
  change2(test, 3);
  printf("After:");
  print(test, length);
  printf("after change, test address: %p\n", test);
}