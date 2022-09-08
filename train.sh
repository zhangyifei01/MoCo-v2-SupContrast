export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
CUDA_VISIBLE_DEVICES=0,1 python main_moco.py \
  -a resnet50 \
  --lr 0.12 \
  --batch-size 256 --moco-k 4096 \
  --mlp --moco-t 0.2 --aug-plus --cos \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  data/

CUDA_VISIBLE_DEVICES=0,1 python main_lincls.py \
  -a resnet50 \
  --lr 1.0 \
  --batch-size 256 \
  --pretrained checkpoint_0999.pth.tar \
  --dist-url 'tcp://localhost:10002' --multiprocessing-distributed --world-size 1 --rank 0 \
  data/

