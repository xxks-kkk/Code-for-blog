#define _XOPEN_SOURCE 500

#define FUSE_USE_VERSION 26

#include <errno.h>
#include <fuse.h>
#include <libgen.h>
#include <libssh/libssh.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef _BARFS_H
#define _BARFS_H

#if defined(DEBUG) || defined(_DEBUG)
#define LOG(M, ...)                                                            \
  do {                                                                         \
    fprintf(stderr, "[%s (%s:%d)]  " M "\n", __func__, __FILE__, __LINE__,     \
            ##__VA_ARGS__);                                                    \
  } while (0)
#else
#define LOG(M, ...)                                                            \
  do {                                                                         \
  } while (0)
#endif

#define CHECK(EXPR, M, ...)                                                    \
  do {                                                                         \
    if (!(EXPR)) {                                                             \
      fprintf(stderr, "[%s (%s:%d)] ERROR  " M "\n", __func__, __FILE__,       \
              __LINE__, ##__VA_ARGS__);                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#define DEFAULT_FILE_MODE 0644
#define DEFAULT_DIRECTORY_MODE 0755

int remote_exec(const char *, char *, int);
int mkpath(char *, mode_t);
int remote_download_file(const char *);
int remote_upload_file(const char *);
int remote_create_file(const char *);
char *parse_str(char *, const char *);
static int barfs_getattr(const char *, struct stat *);
static int barfs_readdir(const char *, void *, fuse_fill_dir_t, off_t,
                         struct fuse_file_info *);
int process_flags(int);
int flags_contain_write(int);
static int barfs_create(const char *, mode_t, struct fuse_file_info *);
static int barfs_open(const char *, struct fuse_file_info *);
static int barfs_read(const char *, char *, size_t, off_t,
                      struct fuse_file_info *);
static int barfs_write(const char *, const char *, size_t, off_t,
                       struct fuse_file_info *);
static int barfs_release(const char *path, struct fuse_file_info *);
static int barfs_fsync(const char *, int, struct fuse_file_info *);
static int barfs_mkdir(const char *, mode_t);
static int barfs_unlink(const char *);
static int barfs_rmdir(const char *);

#endif
