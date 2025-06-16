Docker image build command in this dir:
```
docker build -t FemoraLyze:latest .
```
Docker container run command in your FemoraLyze repo dir:
```
docker run -it --rm --gpus all -v $(pwd):/workspace   FemoraLyze:latest   bash
```
