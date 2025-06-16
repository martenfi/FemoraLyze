Docker image build command in this dir:
```
docker build -t femoralyze:latest .
```
Docker container run command in your FemoraLyze repo dir:
```
docker run -it --rm --gpus all -v $(pwd):/workspace   femoralyze:latest   bash
```
