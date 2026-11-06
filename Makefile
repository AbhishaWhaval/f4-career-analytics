PY ?= python
export PYTHONPATH := src

.PHONY: setup data build bios train all test app clean

setup:         ## install dependencies
	$(PY) -m pip install -r requirements.txt

data:          ## download Wikipedia season pages into data/raw (cached, ~15 min first time)
	$(PY) -m f4ca fetch

build:         ## parse cached pages into data/processed/*.csv
	$(PY) -m f4ca build

bios:          ## look up driver birth dates (cached)
	$(PY) -m f4ca bios

train:         ## features, models, figures, reports/results.md and app/data
	$(PY) -m f4ca train

all: data build bios train

test:
	$(PY) -m pytest -q

app:           ## run the Streamlit app locally
	streamlit run app/streamlit_app.py

clean:         ## remove processed data and reports (keeps the download cache)
	rm -rf data/processed reports/figures/*.png reports/*.csv reports/*.json reports/results.md
