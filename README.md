# Road Extraction with Satellite Images and Partial Road Maps
> Accepted by [TGRS'23](https://ieeexplore.ieee.org/document/10081487).

## Requirements
- PyTorch 1.7.0

## Dataset
- The SpaceNet dataset could be downloaded from [this link](https://entuedu-my.sharepoint.com/:u:/g/personal/qianxion001_e_ntu_edu_sg/EcB__mET9yFJnVGnkk3X6tEBdsZT8yCWypuImi62KL7daQ?e=qYlg4L).
- The OSM dataset could be downloaded from [this link](https://entuedu-my.sharepoint.com/:u:/g/personal/qianxion001_e_ntu_edu_sg/EetY7naFINVCl1N5CTwlGXoBzEhidlDqI0Y_eC7yYMjOfg?e=ItIRku).
```
> cd ..
> mkdir data
> cd data
> mkdir osm
> mkdir spacenet
# Download the datasets into corresponding folders and unzip.
```

## Pretrained Models
- `P2CNet` trained on mix SpaceNet and OSM datasets could be downloaded from [this link](https://entuedu-my.sharepoint.com/:u:/g/personal/qianxion001_e_ntu_edu_sg/ESjmO648eS1GjSpC-MPDGS0BMh23vlaTChPktNmIPoP0dw?e=giPsz7).

## Test
```python
> python test_deeplabv3plus_mix_mp_sat_gsam_{spacenet,osm}.py
```

## Train
```python
> python train_deeplabv3plus_mix_mp_sat_gsam_{spacenet,osm}.py
```
