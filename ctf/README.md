# Lösung 

```bash
docker compose -f ../docker-compose.yml up -d 
docker compose -f ../docker-compose.yml exec -it lel bash
```

```bash
cd /app

# get labels and model 
git clone https://github.com/prash29/Hotdog-Not-Hotdog /tmp/model
cp /tmp/model/graph_hotdog.pb .
cp /tmp/model/labels_hotdog.txt .

# old hot or not jpg picture
# curl "https://drive.usercontent.google.com/uc?id=10RxJA5F0d037lNdrZ29fvIaaaBHuaulD&export=download" -L -o hotornot.jpg

# tph specific image
# alternative download it via browser here: https://drive.google.com/file/d/1osWuQ7Y1B91la9BqpZardKvMtdRG_PUB/view
curl "https://drive.usercontent.google.com/download?id=1osWuQ7Y1B91la9BqpZardKvMtdRG_PUB&export=download&confirm=t" -L -o hotornot.jpg


mkdir -p pictures/ processed/
cd pictures

# Split into parts
convert ../hotornot.jpg -crop 224x224 +repage %08d.jpg
echo "Split into $(ls | wc -l) pictures!"
cd ..

# run solver.py
python3 solve.py 
```

Original DataSet: https://www.kaggle.com/datasets/thedatasith/hotdog-nothotdog


### Solve

```bash
➜  cd ctf/
➜  sudo docker compose -f ../docker-compose.yml up -d 
➜  sudo docker compose -f ../docker-compose.yml exec -it lel bash


________                               _______________
___  __/__________________________________  ____/__  /________      __
__  /  _  _ \_  __ \_  ___/  __ \_  ___/_  /_   __  /_  __ \_ | /| / /
_  /   /  __/  / / /(__  )/ /_/ /  /   _  __/   _  / / /_/ /_ |/ |/ /
/_/    \___//_/ /_//____/ \____//_/    /_/      /_/  \____/____/|__/


WARNING: You are running this container as root, which can cause new files in
mounted volumes to be created as the root user on your host machine.

To avoid this, run the container by specifying your user's userid:

$ docker run -u $(id -u):$(id -g) args...

root@a82717ea7166:/# cd app/
root@a82717ea7166:/app# ls
README.md  create_new_challenge.py  pictures  policy.xml  solve.py
root@a82717ea7166:/app# git clone https://github.com/prash29/Hotdog-Not-Hotdog /tmp/model
Cloning into '/tmp/model'...
remote: Enumerating objects: 797, done.
remote: Total 797 (delta 0), reused 0 (delta 0), pack-reused 797 (from 1)
Receiving objects: 100% (797/797), 141.40 MiB | 6.22 MiB/s, done.
Resolving deltas: 100% (17/17), done.
root@a82717ea7166:/app# cp /tmp/model/graph_hotdog.pb .
cp /tmp/model/labels_hotdog.txt .
root@a82717ea7166:/app# curl "https://drive.usercontent.google.com/uc?id=10RxJA5F0d037lNdrZ29fvIaaaBHuaulD&export=download" -L -o hotornot.jpg
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
  0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
100 68.1M  100 68.1M    0     0  4871k      0  0:00:14  0:00:14 --:--:-- 5527k
root@a82717ea7166:/app# mkdir -p pictures/ processed/
cd pictures
root@a82717ea7166:/app/pictures# convert ../hotornot.jpg -crop 224x224 +repage %08d.jpg
root@a82717ea7166:/app/pictures# ls | wc -l
7569
root@a82717ea7166:/app/pictures# cd ..
root@a82717ea7166:/app# python3 solve.py
root@a82717ea7166:/app# python3 solve.py 
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1789823597.813905      57 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
I0000 00:00:1789823597.894566      57 cpu_feature_guard.cc:227] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
To enable the following instructions: AVX2 AVX_VNNI FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1789823599.719725      57 port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
W0000 00:00:1789823602.023167      57 op_def_util.cc:371] Op BatchNormWithGlobalNormalization is deprecated. It will cease to work in GraphDef version 9. Use tf.nn.batch_normalization().
I0000 00:00:1789823602.742558      57 mlir_graph_optimization_pass.cc:437] MLIR V1 optimization pass is not enabled
[200/7569] 00000199.jpg: not hotdog
[400/7569] 00000399.jpg: not hotdog
[600/7569] 00000599.jpg: not hotdog

[.........]

```