# CVpipeline

## Recommended Docker Image
```
docker pull hirune924/cuda-conda-deep:latest
```
## Get Started
```
git clone https://github.com/hirune924/CVpipeline.git
cd CVpipeline
bash docker-run.sh
``` 
## TODO
- [x] Apex
- [x] DALI (Limited function version)
- [x] lmdb
- [x] add tensorboard( and watch visdom)
- [x] add dataparallel
- [x] save model parameter
- [ ] add DataAugmentation
- [ ] lr scheduler(optimize for model params)
- [ ] error handling
- [ ] logger
- [ ] inference script(confusion matrix)
- [ ] grad cam
- [ ] add more criterion
- [ ] add more method of validation split (adjust distribution)
- [ ] add more metrics
- [ ] add more optimizer
- [ ] add more models
- [ ] RAPIDS
- [ ] optuna, BoTorch
- [ ] add data distribution data parallel

