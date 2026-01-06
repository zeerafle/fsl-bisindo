# 1000 episodes
# autsl
uv run tools/train_fewshot.py --config configs/fewshot/train_protonet.yaml \
--eval_only \
--resume_artifact "zeerafle-sivas-cumhuriyet-university/wl-bisindo-fsl/protonet-best:v0" \
--n_test_episodes 1000

# csl
uv run tools/train_fewshot.py --config configs/fewshot/train_protonet.yaml \
--eval_only \
--resume_artifact "zeerafle-sivas-cumhuriyet-university/wl-bisindo-fsl/protonet-best:v1" \
--n_test_episodes 1000

# lsa64
uv run tools/train_fewshot.py --config configs/fewshot/train_protonet.yaml \
--eval_only \
--resume_artifact "zeerafle-sivas-cumhuriyet-university/wl-bisindo-fsl/protonet-best:v2" \
--n_test_episodes 1000

# 1-shot
# autsl
uv run tools/train_fewshot.py --config configs/fewshot/train_protonet.yaml \
--eval_only \
--resume_artifact "zeerafle-sivas-cumhuriyet-university/wl-bisindo-fsl/protonet-best:v0" \
--n_test_episodes 1000 \
--k_shot 1

# csl
uv run tools/train_fewshot.py --config configs/fewshot/train_protonet.yaml \
--eval_only \
--resume_artifact "zeerafle-sivas-cumhuriyet-university/wl-bisindo-fsl/protonet-best:v1" \
--n_test_episodes 1000 \
--k_shot 1

# lsa64
uv run tools/train_fewshot.py --config configs/fewshot/train_protonet.yaml \
--eval_only \
--resume_artifact "zeerafle-sivas-cumhuriyet-university/wl-bisindo-fsl/protonet-best:v2" \
--n_test_episodes 1000 \
--k_shot 1

# train from scratch and eval
uv run tools/train_fewshot.py --config configs/fewshot/train_protonet.yaml \
--encoder configs/backbones/random_slgcn.yaml \
--unfreeze_encoder \
--lr 0.001 \
--n_epochs 200

# eval from from-scratch-trained model for 1-shot
uv run tools/train_fewshot.py --config configs/fewshot/train_protonet.yaml \
--eval_only \
--resume_artifact "zeerafle-sivas-cumhuriyet-university/wl-bisindo-fsl/protonet-best:v3" \
--n_test_episodes 1000 \
--k_shot 1
