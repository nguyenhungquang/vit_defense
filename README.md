Put data in ```data/imagenet```

Example commands
```
python main.py --attack simba_dct --eps 1 --defense random_noise-def_position logits --dataset imagenet --n_ex 1000 --n_iter 1000

python main.py --attack square_l2 --eps 10 --p 0.1 --defense random_noise --def_position pre_att_cls --dataset imagenet --n_ex 1000 --n_iter 1000 --gpu 1 --noise_list 0.03 0.01 0.005

python main.py --attack square_linf --defense random_noise --def_position pre_att_cls --dataset cifar10 --n_ex 1000 --n_iter 1000 --gpu 3
```