# Training Asccl on Shengteng server

This has enabled the training of asccl on Shengteng NPU.

## Requirements

- Install PyTorch ([pytorch.org](http://pytorch.org)).
- `pip install -r requirements.txt`
- Download the MEAD dataset from ([here](https://wywu.github.io/projects/MEAD/MEAD.html)).
- Download the pre-trained weights ([here](https://drive.google.com/file/d/1W_qa9xxXTCXo_44PX_oRDLlJQ3F8uXJk/view?usp=sharing)) (" backbone.pth ") and place it under "./pretrain/backbone.pth"

## Environmental configuration
Before running the code, it is necessary to configure the environment through 'env. sh':

```bash
source env.sh
```

## Preprocessing
The obtained MEAD dataset is first preprocessed with 'dataloader/align_face.py':

```bash
python ./dataloader/align_face.py
```

## Training

To train the model, run './trainer/train_asccl.py' with the preprocessed dataset path configured:

```bash
python ./trainer/train_asccl.py
```



