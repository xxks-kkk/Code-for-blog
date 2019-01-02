#include "mkl.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#define min(x, y) (((x) < (y)) ? (x) : (y))
#define NUMBER_OF_INPUT_CELLS 4

typedef struct Cell {
  double input[NUMBER_OF_INPUT_CELLS];
  double weight[NUMBER_OF_INPUT_CELLS];
  double output;
  double bias;
} Cell;

void setVals(Cell *c) {
  for (int i = 0; i < NUMBER_OF_INPUT_CELLS; i++) {
    c->input[i] = i + 1;
    c->weight[i] = i + 1;
    /* printf("c->input[%d] = %f\n", i, c->input[i]); */
    /* printf("c->weight[%d] = %f\n", i, c->weight[i]); */
  }
}

/* Vanila matrix multiplication */
void calc1(Cell *c) {
  c->output = 0;

  for (int i = 0; i < NUMBER_OF_INPUT_CELLS; i++) {
    c->output += c->input[i] * c->weight[i];
  }

  c->output /= NUMBER_OF_INPUT_CELLS; // normalize output (0-1)
  printf("Calc1 output: %f\n", c->output);
}

/* Calculate the same thing as Calc1 but using "mkl.h" instead */
void calc3(Cell *c) {
  double C[1];
  int m, n, k, i, j;
  double alpha, beta;

  m = 1, k = NUMBER_OF_INPUT_CELLS, n = 1;
  alpha = 1.0;
  beta = 0.0;

  C[0] = 0.0;
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha,
              c->input, k, c->weight, n, beta, C, n);

  printf("calc3: C[0]=%f\n", C[0]);
  printf("calc3: %f\n", C[0] / NUMBER_OF_INPUT_CELLS);
}

int main() {
  Cell c;
  setVals(&c);

  struct timeval c1_start, c1_end, c3_start, c3_end, c1_res, c3_res;
  gettimeofday(&c1_start, NULL);
  calc1(&c);
  gettimeofday(&c1_end, NULL);
  gettimeofday(&c3_start, NULL);
  calc3(&c);
  gettimeofday(&c3_end, NULL);
  timersub(&c1_end, &c1_start, &c1_res);
  timersub(&c3_end, &c3_start, &c3_res);
  printf("c1: %d s \t %d ms\n", c1_res.tv_sec, c1_res.tv_usec);
  printf("c3: %d s \t %d ms\n", c3_res.tv_sec, c3_res.tv_usec);
}
