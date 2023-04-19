python main.py --attack simba_dct --eps 1 --defense random_noise --def_position last_cls --dataset cifar10 --n_ex 1000 --n_iter 1000 --exp_folder exps_mean --gpu 3

python main.py --attack simba_dct --eps 1 --defense random_noise --def_position input_noise --dataset cifar10 --n_ex 1000 --n_iter 1000 --exp_folder exps --stop_criterion single --gpu 1 --noise_list 0.03 
python main.py --attack square_l2 --eps 5 --p 0.1 --defense random_noise --def_position pre_att_all --dataset imagenet --n_ex 1000 --n_iter 1000 --gpu 2  --noise_list 0.05
python main.py --attack square_linf --eps 0.05 --defense random_noise --def_position pre_att_all --dataset imagenet --n_ex 1000 --n_iter 1000 --gpu 2 --stop_criterion single --exp_folder single --noise_list 0.05 --layer_index 7

python main.py --attack bandit_l2 --eps 5 --defense random_noise --def_position pre_att_all --dataset cifar10 --n_ex 1000 --n_iter 1000 --exp_folder exps_mean --gpu 0
python main.py --attack bandit_l2 --eps 5 --defense identical --def_position baseline --dataset cifar10 --n_ex 1000 --n_iter 1000 --exp_folder exps_mean --gpu 0
python main.py --attack signhunt_l2 --eps 5 --defense random_noise --def_position pre_att_all --dataset cifar10 --n_ex 1000 --n_iter 1000 --gpu 0
python main.py --attack signhunt_l2 --eps 5 --defense random_noise --def_position input_noise --dataset imagenet --n_ex 1000 --n_iter 1000 --gpu 2 --noise_list 0.03 --stop_criterion single --exp_folder single
python main.py --attack simba_dct --eps 0.2 --defense random_noise --def_position input_noise --dataset cifar10 --n_ex 1000 --n_iter 1000 --gpu 1

python main.py --attack bandit_linf --eps 0.1 --defense random_noise --def_position pre_att_cls --dataset imagenet --n_ex 1000 --n_iter 1000 --gpu 1 --exp_folder single --stop_criterion single --noise_list 0.05

python main.py --attack square_linf --eps 0.05 --defense random_noise --def_position pre_att_cls --dataset imagenet --n_ex 1000 --n_iter 1000 --exp_folder all --gpu 3 --stop_criterion none --noise_list 0.05

python main.py --attack nes_linf --eps 0.05 --defense random_noise --def_position pre_att_all --dataset imagenet --n_ex 1000 --n_iter 1000 --gpu 0  --noise_list 0.05

python main.py --attack square_linf --eps 0.1 --defense random_noise --def_position input_noise --dataset imagenet --n_ex 1000 --n_iter 1000 --gpu 2 --noise_list 0.05

python main.py --attack hsja_l2 --eps 1 --defense random_noise --def_position input_noise --dataset imagenet --n_ex 1000 --n_iter 1000 --gpu 3 --stop_criterion single --exp_folder hard_single --noise_list 0.05 --layer_index 7


python main.py --attack square_l2 --eps 10 --p 0.1 --defense random_noise --def_position pre_att_cls --dataset imagenet --n_ex 1000 --n_iter 1000 --gpu 1 --stop_criterion single --exp_folder single_2 
python main.py --attack nes_l2 --eps 5 --defense random_noise --def_position pre_att_cls --dataset imagenet --n_ex 1000 --n_iter 1000  --stop_criterion single --exp_folder single_2  --gpu 0
python main.py --attack bandit_linf --eps 0.05 --defense random_noise --def_position input_noise --dataset imagenet --n_ex 1000 --n_iter 1000  --stop_criterion single --exp_folder single_2  --gpu 3
python main.py --attack nes_linf --eps 0.05 --defense random_noise --def_position input_noise --dataset cifar10 --n_ex 1000 --n_iter 1000 --gpu 1 --stop_criterion single --exp_folder single_2
python main.py --attack nes_l2 --eps 5 --defense random_noise --def_position mix_all_first --dataset cifar10 --n_ex 1000 --n_iter 1000 --gpu 3
python main.py --attack square_linf --eps 0.1 --p 0.05 --defense random_noise --def_position mix_all_first --dataset imagenet --n_ex 1000 --n_iter 1000 --gpu 0 --noise_list 0.05
python main.py --attack square_l2 --eps 10 --p 0.1 --defense random_noise --def_position mix_all_first --dataset imagenet --n_ex 1000 --n_iter 1000 --gpu 3
python main.py --attack signhunt_l2 --eps 5 --defense random_noise --def_position input_noise --dataset imagenet --n_ex 1000 --n_iter 1000 --gpu 1
python main.py --attack square_linf --eps 0.1 --p 0.05 --defense random_noise --def_position pre_att_all --dataset imagenet --n_ex 1000 --n_iter 1000 --gpu 3 --stop_criterion single --exp_folder single_2 --noise_list 0.03

python main.py --attack square_linf --eps 0.05 --defense random_noise --def_position pre_att_all --dataset imagenet --n_ex 1000 --n_iter 1000 --gpu 2 --noise_list 0.05

python main.py --attack nes_linf --eps 0.05 --defense identical --def_position baseline --dataset cifar10 --n_ex 1000 --n_iter 1000 --gpu 3 --noise_list 0

python main.py --attack nes_adapt_l2 --eps 5 --defense random_noise --def_position pre_att_cls --dataset imagenet --n_ex 1000 --n_iter 5000 --exp_folder adaptive --gpu 2 --noise_list 0.1
python main.py --attack nes_adapt_linf --eps 0.05 --defense random_noise --def_position pre_att_cls --dataset cifar10 --n_ex 1000 --n_iter 5000 --exp_folder adaptive --gpu 0 --noise_list 0.1

python main.py --attack nes_linf --eps 0.05 --defense identical --def_position baseline --dataset cifar10 --n_ex 1000 --n_iter 1000 --gpu 0 --exp_folder single --stop_criterion single

python main.py --attack simba_dct --eps 0.1 --defense random_noise --def_position input_noise --dataset cifar10 --n_ex 1000 --n_iter 1000 --exp_folder tuning --gpu 2
python main.py --attack square_linf --eps 0.05 --p 0.3 --defense random_noise --def_position pre_att_cls --dataset imagenet --n_ex 1000 --n_iter 1000 --gpu 2 --exp_folder tuning --noise_list 0.1
python main.py --attack square_l2 --eps 5 --p 0.3 --defense random_noise --def_position pre_att_cls --dataset imagenet --n_ex 1000 --n_iter 1000 --gpu 2 --exp_folder tuning --noise_list 0.1

python main.py --attack square_linf --eps 0.1 --p 0.05 --defense random_noise --def_position pre_att_cls --dataset cifar10 --n_ex 1000 --n_iter 5000 --gpu 0 --exp_folder adaptive --noise_list 0.1 --num_adapt 5
python main.py --attack square_l2 --eps 10 --p 0.1 --defense random_noise --def_position input_noise --dataset cifar10 --n_ex 1000 --n_iter 5000 --gpu 1 --exp_folder adaptive --noise_list 0.05 --num_adapt 5
python main.py --attack bandit_linf --eps 0.05 --defense random_noise --def_position input_noise --dataset imagenet --n_ex 1000 --n_iter 1000  --exp_folder tuning  --gpu 0 --noise_list 0.05
# resnet 50
python main.py --attack square_linf --eps 0.05 --p 0.05 --model resnet50 --defense random_noise --def_position hidden_feature --dataset imagenet_baseline --n_ex 1000 --n_iter 10000 --gpu 3 --exp_folder single --stop_criterion single
python main.py --attack square_linf --eps 0.05 --p 0.05 --model vgg19_bn --defense random_noise --def_position input_noise --dataset imagenet --n_ex 1000 --n_iter 1000 --gpu 3 --exp_folder mean 
