#include "barfs.h"

uid_t current_uid;
gid_t current_gid;

char remote_path[256];
char local_cache_path[256];
__thread char buffer[65536];
__thread char command[1024];

ssh_session session;
pthread_mutex_t session_mutex;

int remote_exec(const char *command, char *buf, int len) {
  LOG("Execute remote command: %s", command);

  if (pthread_mutex_lock(&session_mutex) != 0) {
    LOG("Failed to acquire mutex");
    return -1;
  }

  ssh_channel channel = ssh_channel_new(session);
  if (channel == NULL) {
    LOG("Failed to create SSH channel");
    goto error;
  }
  if (ssh_channel_open_session(channel) != SSH_OK) {
    LOG("Failed to open session on SSH channel");
    goto error;
  }

  if (ssh_channel_request_exec(channel, command) != SSH_OK) {
    LOG("Failed to execute stat command on remote host");
    goto error;
  }

  int status = ssh_channel_get_exit_status(channel);
  if (status != 0) {
    LOG("Failed to get exit status of SSH channel");
    goto error;
  }

  int nbytes = 0;
  int ret;
  while (1) {
    ret = ssh_channel_read(channel, buf + nbytes, len - nbytes, 0);
    if (ret == 0) {
      break;
    }
    if (ret < 0) {
      LOG("Failed to read results from SSH channel");
      goto error;
    }
    nbytes += ret;
  }

  ssh_channel_free(channel);

  if (pthread_mutex_unlock(&session_mutex) != 0) {
    LOG("Failed to release mutex");
    return -1;
  }

  return nbytes;

error:
  if (channel != NULL) {
    ssh_channel_free(channel);
  }
  pthread_mutex_unlock(&session_mutex);
  return -1;
}

int mkpath(char *dir, mode_t mode) {
  struct stat sb;
  if (stat(dir, &sb) == 0) {
    return 0;
  }
  mkpath(dirname(strdupa(dir)), mode);
  return mkdir(dir, mode);
}

int remote_download_file(const char *path) {
  if (pthread_mutex_lock(&session_mutex) != 0) {
    LOG("Failed to acquire mutex");
    return -1;
  }

  sprintf(buffer, "%s%s", remote_path, path);
  ssh_scp scp = ssh_scp_new(session, SSH_SCP_READ, strdupa(buffer));
  if (scp == NULL) {
    LOG("Failed to create SCP session");
    goto error;
  }

  if (ssh_scp_init(scp) != SSH_OK) {
    LOG("Failed to initialize SCP session");
    goto error;
  }

  if (ssh_scp_pull_request(scp) != SSH_SCP_REQUEST_NEWFILE) {
    LOG("Unexpected SCP pull request");
    goto error;
  }

  sprintf(buffer, "%s%s", local_cache_path, path);
  char *local_path = strdupa(buffer);
  mkpath(dirname(strdupa(local_path)), DEFAULT_DIRECTORY_MODE);
  int fd = creat(local_path, DEFAULT_FILE_MODE);
  if (fd == -1) {
    LOG("Failed to create file: %s", local_path);
    goto error;
  }

  size_t size = ssh_scp_request_get_size(scp);
  size_t received_bytes = 0;
  ssh_scp_accept_request(scp);
  while (1) {
    int nbytes = ssh_scp_read(scp, buffer, sizeof(buffer));
    if (nbytes == SSH_ERROR) {
      LOG("Failed to read from SCP session");
      goto error;
    }
    if (write(fd, buffer, nbytes) == -1) {
      LOG("Failed to write to local file");
      goto error;
    }
    received_bytes += nbytes;
    if (received_bytes == size) {
      break;
    }
  }

  if (ssh_scp_pull_request(scp) != SSH_SCP_REQUEST_EOF) {
    LOG("Unexpected SCP pull request");
    goto error;
  }

  ssh_scp_free(scp);

  if (pthread_mutex_unlock(&session_mutex) != 0) {
    LOG("Failed to release mutex");
    return -1;
  }

  return 0;

error:
  if (scp != NULL) {
    ssh_scp_free(scp);
  }
  pthread_mutex_unlock(&session_mutex);
  return -1;
}

int remote_upload_file(const char *path) {
  if (pthread_mutex_lock(&session_mutex) != 0) {
    LOG("Failed to acquire mutex");
    return -1;
  }

  sprintf(buffer, "%s%s", remote_path, path);
  char *remote_path = strdupa(buffer);
  ssh_scp scp =
      ssh_scp_new(session, SSH_SCP_WRITE, dirname(strdupa(remote_path)));
  if (scp == NULL) {
    LOG("Failed to create SCP session");
    goto error;
  }

  if (ssh_scp_init(scp) != SSH_OK) {
    LOG("Failed to initialize SCP session");
    goto error;
  }

  sprintf(buffer, "%s%s", local_cache_path, path);
  char *local_path = strdupa(buffer);
  struct stat st;
  if (stat(local_path, &st) != 0) {
    LOG("Failed to get stat of local file");
    goto error;
  }
  if (ssh_scp_push_file(scp, basename(strdupa(local_path)), st.st_size,
                        DEFAULT_FILE_MODE) != SSH_OK) {
    LOG("Failed to open remote file");
    goto error;
  }

  int fd = open(local_path, O_RDONLY);
  if (fd == -1) {
    LOG("Failed to open local file");
    goto error;
  }

  while (1) {
    int nbytes = read(fd, buffer, sizeof(buffer));
    if (nbytes == 0) {
      break;
    }
    if (nbytes == -1) {
      LOG("Failed to read local file");
      goto error;
    }
    if (ssh_scp_write(scp, buffer, nbytes) != SSH_OK) {
      LOG("Failed to write remote file");
      goto error;
    }
  }

  ssh_scp_free(scp);

  if (pthread_mutex_unlock(&session_mutex) != 0) {
    LOG("Failed to release mutex");
    return -1;
  }

  return 0;

error:
  if (scp != NULL) {
    ssh_scp_free(scp);
  }
  pthread_mutex_unlock(&session_mutex);
  return -1;
}

int remote_create_file(const char *path) {
  if (pthread_mutex_lock(&session_mutex) != 0) {
    LOG("Failed to acquire mutex");
    return -1;
  }

  sprintf(buffer, "%s%s", remote_path, path);
  char *remote_path = strdupa(buffer);
  ssh_scp scp =
      ssh_scp_new(session, SSH_SCP_WRITE, dirname(strdupa(remote_path)));
  if (scp == NULL) {
    LOG("Failed to create SCP session");
    goto error;
  }

  if (ssh_scp_init(scp) != SSH_OK) {
    LOG("Failed to initialize SCP session");
    goto error;
  }

  if (ssh_scp_push_file(scp, basename(strdupa(remote_path)), 0,
                        DEFAULT_FILE_MODE) != SSH_OK) {
    LOG("Failed to open remote file");
    goto error;
  }

  ssh_scp_free(scp);

  if (pthread_mutex_unlock(&session_mutex) != 0) {
    LOG("Failed to release mutex");
    return -1;
  }

  return 0;

error:
  if (scp != NULL) {
    ssh_scp_free(scp);
  }
  pthread_mutex_unlock(&session_mutex);
  return -1;
}

char *parse_str(char *str, const char *pattern) {
  char *p = strstr(str, pattern);
  if (p == NULL) {
    return NULL;
  } else {
    return p + strlen(pattern);
  }
}

static int barfs_getattr(const char *path, struct stat *stbuf) {
  LOG("Get attributes of path: %s", path);

  sprintf(command, "stat \"%s%s\"", remote_path, path);
  int nbytes = remote_exec(command, buffer, sizeof(buffer));
  if (nbytes == -1) {
    return -ENOENT;
  }

  buffer[nbytes] = '\0';
  LOG("Execution result: %s", buffer);

  memset(stbuf, 0, sizeof(struct stat));
  stbuf->st_uid = current_uid;
  stbuf->st_gid = current_gid;

  char *p = buffer;
  char *np;

  np = parse_str(p, "Size:");
  if (np != NULL) {
    p = np;
    long size;
    sscanf(p, "%ld", &size);
    stbuf->st_size = size;
    LOG("size = %ld", size);
  }

  np = parse_str(p, "Blocks:");
  if (np != NULL) {
    p = np;
    long blocks;
    sscanf(p, "%ld", &blocks);
    stbuf->st_blocks = blocks;
    LOG("blocks = %ld", blocks);
  }

  np = parse_str(p, "directory");
  if (np != NULL) {
    p = np;
    stbuf->st_mode = S_IFDIR | DEFAULT_DIRECTORY_MODE;
    LOG("Current file is directory");
  }

  np = parse_str(p, "regular file");
  if (np != NULL) {
    p = np;
    stbuf->st_mode = S_IFREG | DEFAULT_FILE_MODE;
    LOG("Current file is regular file");
  }

  np = parse_str(p, "regular empty file");
  if (np != NULL) {
    p = np;
    stbuf->st_mode = S_IFREG | DEFAULT_FILE_MODE;
    LOG("Current file is regular empty file");
  }

  np = parse_str(p, "Links:");
  if (np != NULL) {
    p = np;
    long nlink;
    sscanf(p, "%ld", &nlink);
    stbuf->st_nlink = nlink;
    LOG("nlink = %ld", nlink);
  }

  return 0;
}

static int barfs_readdir(const char *path, void *buf, fuse_fill_dir_t filler,
                         off_t offset, struct fuse_file_info *fi) {
  LOG("Read directory: %s", path);

  struct stat st;
  int ret = barfs_getattr(path, &st);
  if (ret != 0) {
    return ret;
  }
  if ((st.st_mode | S_IFDIR) == 0) {
    return -ENOTDIR;
  }

  sprintf(command, "ls -1a \"%s%s\"", remote_path, path);
  int nbytes = remote_exec(command, buffer, sizeof(buffer));
  if (nbytes == -1) {
    return -ENOENT;
  }

  buffer[nbytes] = '\0';
  LOG("Execution result: %s", buffer);

  char *p = buffer;
  char *np;
  while (1) {
    np = strchr(p, '\n');
    if (np == NULL) {
      break;
    }
    *np = '\0';
    if (filler(buf, p, NULL, 0) != 0) {
      LOG("Filler failed");
      return -EINVAL;
    }
    p = np + 1;
  }

  return 0;
}

int process_flags(int fi_flags) {
  int flags = 0;
  if ((fi_flags & O_RDONLY) != 0) {
    flags |= O_RDONLY;
  }
  if ((fi_flags & O_WRONLY) != 0) {
    flags |= O_WRONLY;
  }
  if ((fi_flags & O_RDWR) != 0) {
    flags |= O_RDWR;
  }
  return flags;
}

int flags_contain_write(int flags) {
  if ((flags & O_WRONLY) != 0) {
    return 1;
  }
  if ((flags & O_RDWR) != 0) {
    return 1;
  }
  return 0;
}

static int barfs_create(const char *path, mode_t mode,
                        struct fuse_file_info *fi) {
  LOG("Create file: %s", path);

  struct stat st;
  int ret = barfs_getattr(path, &st);
  if ((fi->flags | O_EXCL) != 0 && ret == 0) {
    return -EEXIST;
  }

  if (ret != 0) {
    if (remote_create_file(path) != 0) {
      return -1;
    }
  }

  if (remote_download_file(path) != 0) {
    return -ENOENT;
  }

  sprintf(buffer, "%s%s", local_cache_path, path);
  char *local_path = strdupa(buffer);
  LOG("Original flags: %x", fi->flags);
  LOG("Processed flags: %x", process_flags(fi->flags));
  int fd = open(local_path, process_flags(fi->flags));
  if (fd == -1) {
    LOG("Failed to open local file");
    return -errno;
  }
  fi->fh = fd;
  return 0;
}

static int barfs_open(const char *path, struct fuse_file_info *fi) {
  LOG("Open file: %s", path);
  if (remote_download_file(path) != 0) {
    return -ENOENT;
  }
  sprintf(buffer, "%s%s", local_cache_path, path);
  char *local_path = strdupa(buffer);
  int fd = open(local_path, process_flags(fi->flags));
  if (fd == -1) {
    LOG("Failed to open local file");
    return -errno;
  }
  fi->fh = fd;
  return 0;
}

static int barfs_read(const char *path, char *buf, size_t size, off_t offset,
                      struct fuse_file_info *fi) {
  LOG("Read file: %s", path);
  int ret = pread(fi->fh, buf, size, offset);
  if (ret == -1) {
    LOG("Failed to read from local file");
    return -errno;
  }
  return ret;
}

static int barfs_write(const char *path, const char *buf, size_t size,
                       off_t offset, struct fuse_file_info *fi) {
  LOG("Write file: %s", path);
  int ret = pwrite(fi->fh, buf, size, offset);
  if (ret == -1) {
    LOG("Failed to write to local file");
    return -errno;
  }
  return ret;
}

static int barfs_release(const char *path, struct fuse_file_info *fi) {
  LOG("Release file: %s", path);
  close(fi->fh);
  remote_upload_file(path);
  return 0;
}

static int barfs_fsync(const char *path, int datasync,
                       struct fuse_file_info *fi) {
  LOG("Fsync file: %s", path);
  if (fsync(fi->fh) != 0) {
    LOG("Failed to fsync local file");
    return -errno;
  }
  return 0;
}

static int barfs_mkdir(const char *path, mode_t mode) {
  LOG("Create directory: %s", path);

  sprintf(command, "mkdir -m 0%o \"%s%s\"", DEFAULT_DIRECTORY_MODE, remote_path,
          path);
  int nbytes = remote_exec(command, buffer, sizeof(buffer));
  if (nbytes == -1) {
    return -1;
  }

  return 0;
}

static int barfs_unlink(const char *path) {
  LOG("Remove file: %s", path);

  sprintf(command, "rm \"%s%s\"", remote_path, path);
  int nbytes = remote_exec(command, buffer, sizeof(buffer));
  if (nbytes == -1) {
    return -1;
  }

  return 0;
}

static int barfs_rmdir(const char *path) {
  LOG("Remove directory: %s", path);

  sprintf(command, "rm -d \"%s%s\"", remote_path, path);
  int nbytes = remote_exec(command, buffer, sizeof(buffer));
  if (nbytes == -1) {
    return -1;
  }

  return 0;
}

static struct fuse_operations barfs_oper = {
    .getattr = barfs_getattr,
    .readdir = barfs_readdir,
    .create = barfs_create,
    .open = barfs_open,
    .read = barfs_read,
    .write = barfs_write,
    .release = barfs_release,
    .fsync = barfs_fsync,
    .mkdir = barfs_mkdir,
    .unlink = barfs_unlink,
    .rmdir = barfs_rmdir,
};

int main(int argc, char *argv[]) {
  CHECK(argc >= 5,
        "The first four arguments should be remote_user, remote_host, "
        "remote_path and local_cache_path.");

  current_uid = geteuid();
  current_gid = getegid();

  const char *remote_user = argv[1];
  const char *remote_host = argv[2];
  strcpy(remote_path, argv[3]);
  strcpy(local_cache_path, argv[4]);
  for (int i = 5; i < argc; i++) {
    argv[i - 4] = argv[i];
  }
  argc -= 4;

  CHECK(pthread_mutex_init(&session_mutex, NULL) == 0,
        "Failed to create mutex for SSH session");

  session = ssh_new();
  ssh_options_set(session, SSH_OPTIONS_HOST, remote_host);
  CHECK(ssh_connect(session) == SSH_OK, "Failed to connect to %s", remote_host);

  CHECK(ssh_userauth_publickey_auto(session, remote_user, NULL) ==
            SSH_AUTH_SUCCESS,
        "Failed to authenticate user %s", remote_user);

  return fuse_main(argc, argv, &barfs_oper, NULL);
}
