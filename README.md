# entrywatch

process

activate env
```python
source /home/squidward/hailo-apps/venv_hailo_apps/bin/activate
```

inference
```python
python3 app.py --input rpi

```


build your dataset
```python
python -m scripts.capture_images --name oscar
```


training
```
python -m scripts.train
```

