from os import walk

filenames = next(walk("./searchable_params"), (None, None, []))[2]  # [] if no file
for idx, filename in enumerate(filenames):
    if filename == '.gitkeep':
        continue
    print(f'export NAME=search_params_{idx}; tmux new -s $NAME -d;'
          f'tmux send-keys -t $NAME "conda activate PAS_sim;'
          f'python search_params.py {filename} {idx}" enter;\\')