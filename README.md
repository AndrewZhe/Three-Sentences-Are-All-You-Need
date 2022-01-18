# Three-Sentences-Are-All-You-Need
Codes for "Three Sentences Are All You Need: Local Path Enhanced DocumentRelation Extraction"

## Prepare Data
Step1: Download folder from https://drive.google.com/drive/folders/1dINWCYhOhqt_s2AWflMGT2FWP37RZWMM?usp=sharing and put it into ./dataset/DocRED_baseline_metadata

Step2: Download train_annotated.json, dev.json, test.json from https://drive.google.com/drive/folders/1c5-0YwnoJx8NS6CV2f-NoTHR__BdkNqw and put them into ./dataset

Step3: Go to ./src/preprocess/ and run make_dataset_doc_path.py. The preprocessed data are stored in ./dataset/100_lower-doc-new-path

## Train Model
Go to src/script and run local_path.sh. The log file will be stored in ./log/100_lower-doc-new-path/local_path/.

I train this model on 3090 with 24G memory. If you want to train this model on a gpu with smaller vm, you need to reduce the negative samples in one document. (I will release a version to support it soon)
