# Three-Sentences-Are-All-You-Need
Codes for "Three Sentences Are All You Need: Local Path Enhanced DocumentRelation Extraction"

## Prepare Data
Step1: Download folder from https://drive.google.com/drive/folders/1dINWCYhOhqt_s2AWflMGT2FWP37RZWMM?usp=sharing and put it into ./dataset/DocRED_baseline_metadata
Step2: Download train_annotated.json, dev.json, test.json and put them into ./dataset
Step3: Go to ./src/preprocess/ and run make_dataset_doc_path.py. The preprocessed data are stored in ./dataset/100_lower-doc-new-path

## Train Model
Go to src/script and run local_path.sh. The log file will be stored in ./log/100_lower-doc-new-path/local_path/.
