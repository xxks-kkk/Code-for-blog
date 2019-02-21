// vim: ts=2 sw=2 et

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/time.h>

#if defined(DEBUG) || defined(_DEBUG)
#define LOG(M, ...) \
  do { \
    fprintf(stderr, "[%s (%s:%d)]  " M "\n", __func__, __FILE__, __LINE__, ##__VA_ARGS__); \
  } while (0)
#else
#define LOG(M, ...) do {} while (0)
#endif

#define CHECK(EXPR, M, ...) \
  do { \
    if (!(EXPR)) { \
      fprintf(stderr, "[%s (%s:%d)] ERROR  " M "\n", __func__, __FILE__, __LINE__, ##__VA_ARGS__); \
      exit(EXIT_FAILURE); \
    } \
  } while (0)


#define DEFAULT_FILE_MODE 0644

struct timer_t {
  double start_time;
  double total_time;
};

void reset_timer(struct timer_t* timer) {
  timer->total_time = 0;
}

void start_timer(struct timer_t* timer, int reset) {
  struct timeval time;
  CHECK(gettimeofday(&time, NULL) == 0, "gettimeofday() failed");
  timer->start_time = (double)time.tv_sec + (double)time.tv_usec / 1e6;
  if (reset != 0) {
    reset_timer(timer);
  }
}

void stop_timer(struct timer_t* timer) {
  struct timeval time;
  CHECK(gettimeofday(&time, NULL) == 0, "gettimeofday() failed");
  timer->total_time += (double)time.tv_sec + (double)time.tv_usec / 1e6 - timer->start_time;
}

int randint(int lo, int hi) {
  return rand() % (hi - lo) + lo;
}

void random_bytes(char* buffer, int size) {
  for (int i = 0; i < size; i++) {
    buffer[i] = rand() & 255;
  }
}

void random_filename(char* buffer, int size) {
  for (int i = 0; i < size; i++) {
    buffer[i] = (char)randint('a', 'z'+1);
  }
  buffer[size] = '\0';
}

int pwrite_all(int fd, const char* buf, int count, int offset) {
  do {
    int ret = pwrite(fd, buf, count, offset);
    if (ret == -1) {
      return -1;
    }
    buf += ret;
    count -= ret;
    offset += ret;
  } while (count > 0);
  return 0;
}

int write_all(int fd, const char* buf, int count) {
  do {
    int ret = write(fd, buf, count);
    if (ret == -1) {
      return -1;
    }
    buf += ret;
    count -= ret;
  } while (count > 0);
  return 0;
}

char buffer[10 * 1024 * 1024];  // 10MB

void create_random_file(char* path, int size) {
  CHECK(size % sizeof(buffer) == 0, "Size should be multiple of 64KB");
  int fd = creat(path, DEFAULT_FILE_MODE);
  CHECK(fd != -1, "Failed to create file %s: %s", path, strerror(errno));
  int n_round = size / sizeof(buffer);
  for (int i = 0; i < n_round; i++) {
    random_bytes(buffer, sizeof(buffer));
    CHECK(write_all(fd, buffer, sizeof(buffer)) == 0,
          "Failed to write bytes to file %s: %s", path, strerror(errno));
  }
  close(fd);
}

const char* bench_dir;
char filename[512];
char filepath[512];

// Round 1: 10^5 4K random writes
void round1() {
  struct timer_t timer;
  printf("[4K random writes]\n");

  int filesize = 100 * 1024 * 1024;  // 100MB
  random_filename(filename, 20);
  sprintf(filepath, "%s/%s", bench_dir, filename);
  start_timer(&timer, 1);
  create_random_file(filepath, filesize);
  stop_timer(&timer);
  printf("creation: %.6f s\n", timer.total_time);
  int n_write = 100000;
  int write_size = 4096;  // 4KB

  random_bytes(buffer, write_size);

  start_timer(&timer, 1);
  int fd = open(filepath, O_WRONLY);
  stop_timer(&timer);
  printf("open: %.6f s\n", timer.total_time);
  CHECK(fd != -1, "Failed to open file %s: %s", filepath, strerror(errno));

  start_timer(&timer, 1);
  for (int i = 0; i < n_write; i++) {
    int offset = randint(0, filesize / write_size) * write_size;
    CHECK(pwrite_all(fd, buffer, write_size, offset) == 0,
          "Failed to write bytes to file %s: %s", filepath, strerror(errno));
  }
  stop_timer(&timer);
  printf("write: %.6f s\n", timer.total_time);

  start_timer(&timer, 1);
  close(fd);
  stop_timer(&timer);
  printf("close: %.6f s\n", timer.total_time);

  printf("\n");
}

// Round 2: 4KB sequential writes with fsync
void round2() {
  struct timer_t timer;
  printf("[4KB sequential writes with fsync]\n");

  int filesize = 500 * 1024 * 1024;  // 500MB
  random_filename(filename, 20);
  sprintf(filepath, "%s/%s", bench_dir, filename);
  int write_size = 4096;  // 4KB
  int n_write = filesize / write_size;

  random_bytes(buffer, write_size);

  start_timer(&timer, 1);
  int fd = creat(filepath, DEFAULT_FILE_MODE);
  stop_timer(&timer);
  printf("open: %.6f s\n", timer.total_time);
  CHECK(fd != -1, "Failed to open file %s: %s", filepath, strerror(errno));

  start_timer(&timer, 1);
  for (int i = 0; i < n_write; i++) {
    CHECK(write_all(fd, buffer, write_size) == 0,
          "Failed to write bytes to file %s: %s", filepath, strerror(errno));
    CHECK(fsync(fd) == 0, "Failed to fsync file %s: %s", filepath, strerror(errno));
  }
  stop_timer(&timer);
  printf("write: %.6f s\n", timer.total_time);

  start_timer(&timer, 1);
  close(fd);
  stop_timer(&timer);
  printf("close: %.6f s\n", timer.total_time);

  printf("\n");
}

// Round 3: create 100 4K random files
void round3() {
  printf("[create 4K random files]\n");

  int n_files = 100;
  int file_size = 4096;  // 4KB

  struct timer_t creat_timer;
  struct timer_t write_timer;
  struct timer_t close_timer;
  reset_timer(&creat_timer);
  reset_timer(&write_timer);
  reset_timer(&close_timer);

  for (int i = 0; i < n_files; i++) {
    random_filename(filename, 20);
    sprintf(filepath, "%s/%s", bench_dir, filename);

    start_timer(&creat_timer, 0);
    int fd = creat(filepath, DEFAULT_FILE_MODE);
    CHECK(fd != -1, "Failed to create file %s: %s", filepath, strerror(errno));
    stop_timer(&creat_timer);

    random_bytes(buffer, file_size);

    start_timer(&write_timer, 0);
    CHECK(write_all(fd, buffer, file_size) == 0,
          "Failed to write bytes to file %s: %s", filepath, strerror(errno));
    stop_timer(&write_timer);

    start_timer(&close_timer, 0);
    close(fd);
    stop_timer(&close_timer);
  }

  printf("creat: %.6f s\n", creat_timer.total_time);
  printf("write: %.6f s\n", write_timer.total_time);
  printf("close: %.6f s\n", close_timer.total_time);

  printf("\n");
}

int main(int argc, char* argv[]) {
  srand(time(NULL));
  CHECK(argc >= 2, "The first argument should be bench_dir");
  bench_dir = argv[1];

  round1();
  round2();
  round3();

  return 0;
}