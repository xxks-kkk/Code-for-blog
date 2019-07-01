#include <stdlib.h>

int main() {
  char *a = (char *)malloc(10 * sizeof(char));
  free(a);
  return 0;
}
