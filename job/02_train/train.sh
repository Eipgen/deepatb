export PYTHONUNBUFFERED=TRUE
deepks train train_input.yaml -d train.raw -t valid.raw  -o model.pth > log.iter 2> err.iter
