"""One output authority for primary files, companions and resource reads.

Servers supply a trusted root. An explicit in-process client may instead use
trusted local destinations. Directory-fd traversal and atomic replacement do
not follow request-controlled symlinks, including final companion targets.
"""
import contextlib
import json
import os
from pathlib import Path
import stat
import uuid


class OutputPolicy:
    def __init__(self, root=None, max_bytes=32 * 1024 * 1024):
        self.root = Path(root).expanduser().resolve() if root is not None else None
        self.max_bytes = max_bytes

    def path(self, requested):
        if not isinstance(requested, (str, os.PathLike)) or not str(requested):
            raise ValueError("an output path is required")
        path = Path(requested)
        if self.root is not None:
            if ".." in path.parts:
                raise ValueError("output traversal is not permitted")
            path = path if path.is_absolute() else self.root / path
            try:
                relative = path.relative_to(self.root)
            except ValueError:
                raise ValueError("output must remain inside the server output root") from None
            if not relative.parts:
                raise ValueError("output must name a file")
        else:
            path = Path(os.path.abspath(path))
            if path.is_symlink():
                raise ValueError("symlink output targets are not permitted")
            path = path.parent.resolve() / path.name
        if path.name in ("", ".", ".."):
            raise ValueError("output must name a file")
        # Check existing components before any directory creation/publication.
        for component in reversed((path, *path.parents)):
            if component.is_symlink():
                raise ValueError("symlink output components are not permitted")
        return path

    def admit(self, *requested):
        return [self.path(path) for path in requested]

    @contextlib.contextmanager
    def _parent(self, path, create):
        path = self.path(path)
        anchor = self.root if self.root is not None else Path(path.anchor)
        if self.root is not None and create:
            self.root.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        fd = os.open(anchor, flags)
        try:
            relative = path.relative_to(anchor)
            for part in relative.parts[:-1]:
                if create:
                    try:
                        os.mkdir(part, mode=0o755, dir_fd=fd)
                    except FileExistsError:
                        pass
                child = os.open(part, flags, dir_fd=fd)
                os.close(fd)
                fd = child
            yield fd, relative.name
        finally:
            os.close(fd)

    @staticmethod
    def _check_target(fd, name):
        try:
            mode = os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode
        except FileNotFoundError:
            return
        if not stat.S_ISREG(mode):
            raise ValueError("output target must be a regular non-symlink file")

    @contextlib.contextmanager
    def writer(self, requested):
        with self._parent(requested, True) as (parent, name):
            self._check_target(parent, name)
            temporary = ".pymcp-" + uuid.uuid4().hex
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=parent)
            try:
                with os.fdopen(fd, "wb") as stream:
                    yield stream
                    if stream.tell() > self.max_bytes:
                        raise ValueError("output exceeds the configured byte budget")
                self._check_target(parent, name)
                os.replace(temporary, name, src_dir_fd=parent, dst_dir_fd=parent)
            finally:
                try:
                    os.unlink(temporary, dir_fd=parent)
                except FileNotFoundError:
                    pass

    def save_figure(self, figure, requested, **kwargs):
        path = self.path(requested)
        extension = path.suffix.lstrip(".").lower()
        if extension not in ("png", "svg", "pdf", "jpg", "jpeg"):
            raise ValueError("unsupported figure output format")
        with self.writer(path) as stream:
            figure.savefig(stream, format=extension, **kwargs)
        return str(path)

    def write_json(self, requested, value):
        encoded = json.dumps(value, allow_nan=False, indent=2).encode("utf-8")
        if len(encoded) > self.max_bytes:
            raise ValueError("JSON output exceeds the configured byte budget")
        with self.writer(requested) as stream:
            stream.write(encoded)
        return str(self.path(requested))

    def read(self, requested):
        with self._parent(requested, False) as (parent, name):
            self._check_target(parent, name)
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent)
            with os.fdopen(fd, "rb") as stream:
                value = stream.read(self.max_bytes + 1)
            if len(value) > self.max_bytes:
                raise ValueError("resource exceeds the configured byte budget")
            return value
