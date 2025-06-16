Docker image build command:
```
docker build -t FemoraLyze:latest .
```
Docker container run command:
```
docker run -it --rm --gpus all -v $(pwd):/workspace   FemoraLyze:latest   bash
```
