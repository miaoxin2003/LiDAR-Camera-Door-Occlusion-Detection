#!/usr/bin/env python3
"""Build the current ROS Noetic robot ZIP, excluding large recorded bags."""
import hashlib
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'c32_oak_ros1_bundle.zip'
MANIFEST = ROOT / 'bundle_contents.sha256'


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    directories = ('image_pointcloud_fusion', 'deploy', 'tools', 'models', 'inspection')
    files = []
    for directory in directories:
        for path in (ROOT / directory).rglob('*'):
            if (path.is_file() and not path.is_symlink() and
                    '__pycache__' not in path.parts and
                    path.suffix not in ('.pyc', '.bag')):
                files.append(path)
    files.extend(ROOT / name for name in (
        'README.md', '真机实时部署教程.md', '实时检测精简手册.md',
        '小车复现操作手册.md', '下午执行速查.md', 'YOLO点云定位改动说明.md',
        'calibration_2026_9_13_3_27_59.txt', 'best.pt'))
    files = sorted(set(files), key=lambda path: path.relative_to(ROOT).as_posix())
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError('Missing bundle inputs: ' + ', '.join(missing))
    MANIFEST.write_text(''.join(
        sha256(path) + '  ' + path.relative_to(ROOT).as_posix() + '\n'
        for path in files), encoding='utf-8')
    files.append(MANIFEST)
    with zipfile.ZipFile(ARCHIVE, 'w', compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(ARCHIVE) as bundle:
        corrupt = bundle.testzip()
        if corrupt:
            raise RuntimeError('ZIP CRC validation failed: ' + corrupt)
        names = set(bundle.namelist())
        required = {'best.pt', '真机实时部署教程.md',
                    'image_pointcloud_fusion/scripts/c32_oak_fusion.py',
                    'image_pointcloud_fusion/scripts/c32_common.py',
                    'image_pointcloud_fusion/config/c32_oak.yaml',
                    'image_pointcloud_fusion/rviz/c32_live.rviz'}
        if not required <= names or any(name.endswith('.bag') for name in names):
            raise RuntimeError('ZIP content validation failed')
    digest = sha256(ARCHIVE)
    (ROOT / (ARCHIVE.name + '.sha256')).write_text(
        digest + '  ' + ARCHIVE.name + '\n', encoding='utf-8')
    print('Archive:', ARCHIVE)
    print('Files:', len(files))
    print('Bytes:', ARCHIVE.stat().st_size)
    print('SHA256:', digest)


if __name__ == '__main__':
    main()
