# CS229-final-project

Final project for CS229

- `run_resnet50.py` is the python script used to train, validate, and test the ResNet-50 baseline model.

- `eval_resnet50.py` is the python script used to evaluate the ResNet-50 baseline model on the test set.

- `run_bagnet33.py` is the python script used to train, validate, and test the BagNet-33 baseline model.

- `eval_bagnet33.py` is the python script used to evaluate the BagNet-33 baseline model on the test set.

- `bagnet33_experiments.py` is the python script used to run and evaluate the patch blackout experiments.

- Performance of each model is stored in the `model_performance_results` directory, including loss_acc_plots, terminal output, and model checkpoints.

- The 'flowers_original' directory contains the original downloaded flowers dataset, downloaded from Kaggle at: https://www.kaggle.com/alxmamaev/flowers-recognition.

- The 'flowers_tvtsplit' directory contains the flowers data split into 70\% training, 20\% validation, and 10\% test data subsets.


References:
- BagNets: "APPROXIMATING CNNS WITH BAG-OF-LOCAL FEATURES MODELS WORKS SURPRISINGLY WELL ON IMAGENET" - https://arxiv.org/pdf/1904.00760.pdf
- Datasets: https://www.kaggle.com/alxmamaev/flowers-recognition