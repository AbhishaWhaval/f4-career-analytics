PY ?= python
export PYTHONPATH := src

.PHONY: setup data build train all test clean

setup:         ## install dependencies
	$(PY) -m pip install -r requirements.txt

data:          ## download Wikipedia season pages into data/raw (cached, ~15 min first time)
	$(PY) -m f4ca fetch

build:         ## parse cached pages into data/processed/*.csv
	$(PY) -m f4ca build

train:         ## features, models, figures and reports/results.md
	$(PY) -m f4ca train

all: data build train

test:
	$(PY) -m pytest -q

clean:         ## remove processed data and reports (keeps the download cache)
	rm -rf data/processed reports/figures/*.png reports/*.csv reports/*.json reports/results.md
