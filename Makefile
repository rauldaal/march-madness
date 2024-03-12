# Create virtualenvironment and install requiremnts
setup:
	python3 -m venv .venv
	. .venv/bin/activate; pip install -r requirements.txt

kaggle_data:
	mkdir -p data
	kaggle competitions download -c march-machine-learning-mania-2024 -p data
	unzip data/march-machine-learning-mania-2024.zip -d data
	rm data/march-machine-learning-mania-2024.zip