import os, shutil, random

# Paths to the flat directories (choose one as reference; we'll use FD as the source)
fd_root = r'C:/Users/HP/Documents/COVID-19-EXPIREMENT/FD_1mm_flat'
qd_root = r'C:/Users/HP/Documents/COVID-19-EXPIREMENT/QD_1mm_flat'

# Destination sub‑folders for the split (mirrored in both FD and QD)
split_names = ['train', 'val', 'test']
for split in split_names:
    split_fd_path = os.path.join(fd_root, split)
    split_qd_path = os.path.join(qd_root, split)
    # Remove existing split folder if it exists to avoid leftover files
    if os.path.isdir(split_fd_path):
        shutil.rmtree(split_fd_path)
    if os.path.isdir(split_qd_path):
        shutil.rmtree(split_qd_path)
    os.makedirs(split_fd_path, exist_ok=True)
    os.makedirs(split_qd_path, exist_ok=True)

# Get sorted file lists
fd_files = sorted([f for f in os.listdir(fd_root) if os.path.isfile(os.path.join(fd_root, f))])
qd_files = sorted([f for f in os.listdir(qd_root) if os.path.isfile(os.path.join(qd_root, f))])

if len(fd_files) != len(qd_files):
    raise ValueError(f"FD ({len(fd_files)}) and QD ({len(qd_files)}) file counts differ – cannot guarantee pairing.")

# Deterministic split – keep original order
N = len(fd_files)
train_end = int(0.80 * N)
val_end = train_end + int(0.10 * N)

splits = {
    'train': list(range(0, train_end)),
    'val': list(range(train_end, val_end)),
    'test': list(range(val_end, N))
}


# Helper to copy a set of paired files into a given split folder
def copy_pair(idx, split):
    fd_src = os.path.join(fd_root, fd_files[idx])
    qd_src = os.path.join(qd_root, qd_files[idx])
    fd_dst = os.path.join(fd_root, split, fd_files[idx])
    qd_dst = os.path.join(qd_root, split, qd_files[idx])
    shutil.copy2(fd_src, fd_dst)
    shutil.copy2(qd_src, qd_dst)

# Perform copying for each split
for split_name, idx_list in splits.items():
    for idx in idx_list:
        copy_pair(idx, split_name)

print('Partitioning complete.')
print(f'Total pairs: {N}')
for s, lst in splits.items():
    print(f"{s.capitalize()}: {len(lst)} pairs")
