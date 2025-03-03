# ECG Heartbeat Categorization Prediction Project documentation!

## Description

ECG Heartbeat Categorization Prediction

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `gsutil rsync` to recursively sync files in `data/` up to `gs://sickkids-ecg-classification/data/`.
* `make sync_data_down` will use `gsutil rsync` to recursively sync files in `gs://sickkids-ecg-classification/data/` to `data/`.


