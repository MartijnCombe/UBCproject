# Applying Transfer Learning to PriSTI for Spatiotemporal Imputation on new Data

## Requirement
Create a conda or python environment
See `requirements.txt` for the list of packages.
To install:
pip install -r requirements.txt

## Experiments
### Training of PriSTI

To train PriSTI on different datasets, you can run the scripts `exe_{dataset_name}.py` such as:
```
python exe_aqi36.py --device 'cuda:0' --num_workers 16

```
### Inference by the trained PriSTI
You can directly use our provided trained model for imputation:
```
python exe_aqi36.py --device 'cuda:0' --num_workers 16 --modelfolder 'aqi36'
python exe_metrla.py --device 'cuda:0' --num_workers 16 --modelfolder 'metr_la'
python exe_pemsbay.py --device 'cuda:0' --num_workers 16 --modelfolder 'pems_bay'
```
### For running Transfer learning experiments
python exe_pems08_TL.py --device 'cuda:0' --num_workers 16 --modelfolder 'pems_bay'
python exe_solar_TL.py --device 'cuda:0' --num_workers 16 --modelfolder 'aqi36'

## Citation of the original work

```bibtex
@article{liu2023pristi,
  title={PriSTI: A Conditional Diffusion Framework for Spatiotemporal Imputation},
  author={Liu, Mingzhe and Huang, Han and Feng, Hao and Sun, Leilei and Du, Bowen and Fu, Yanjie},
  journal={arXiv preprint arXiv:2302.09746},
  year={2023}
}
```




