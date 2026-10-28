# Smart Agriculture AI Irrigation Prediction System

A college demonstration for IoT and machine learning. A Random Forest predicts whether a field needs irrigation from simulated soil and weather readings. The model is trained, registered, and deployed in Azure Machine Learning. The Streamlit dashboard sends each reading to that online endpoint and shows the prediction that comes back. No real irrigation hardware is used.

## Problem statement

Deciding when to irrigate depends on soil moisture, temperature, humidity, and recent rainfall. This project asks a simple classification question:

**Given the current simulated sensor readings, is irrigation required?**

The answer is a recommendation only. The program does not open a valve, start a pump, or connect to a farm.

## Objective

- Build a simulated agricultural dataset and label it with clear watering rules.
- Train a Random Forest classifier and measure it on held-out rows.
- Log the run and register the model with local MLflow.
- Simulate live IoT readings and score them on the laptop (edge inference).
- Show the result in a small dashboard that can be demonstrated in about five minutes.

## System architecture

```
generate_dataset.py
    writes data/agriculture_sensor_data.csv
            |
            +--> local: train_model.py writes models/irrigation_rf_model.joblib
            |           and logs to local MLflow (mlflow.db)
            |
            +--> Azure: az ml job create runs the same train_model.py
                        in the Azure Machine Learning workspace
                        and registers the model there
            |
            v
sensor_simulator.py  --->  Azure ML endpoint  --->  app.py
(new reading / 2 sec)      (model stays deployed)   (shows the reply)
```

The dashboard does not download the model and does not load `models/irrigation_rf_model.joblib`. Each simulated reading is posted to online endpoint `aeal-irrig-score`, deployment `blue`, which serves registered model `smart-agri-irrigation-predictor` version 1. The test scores still come from Azure ML job `sleepy_hamster_zj42xxk8q9`.

## Technologies

| Piece | Role |
| --- | --- |
| Python 3.13 | Tested language version |
| pandas, NumPy | Dataset and feature tables |
| scikit-learn | Random Forest training and metrics |
| joblib | Save and load the model file |
| MLflow 2.22 | Experiment tracking and model registry |
| SQLite | Local MLflow database (`mlflow.db`) |
| Azure CLI and Azure Machine Learning | Training, model registry, and the online scoring endpoint |
| Streamlit | Dashboard |
| Plotly | Sensor history and confusion matrix |

SQLAlchemy is pinned to 2.0.55 because MLflow 2.22 cannot import a connection-pool class that SQLAlchemy 2.1 removed. The local MLflow database uses that pin. Live predictions go to the Azure endpoint.

## Project layout

```
generate_dataset.py      Create the simulated CSV
train_model.py           Train, evaluate, save, and log the model
sensor_simulator.py      Fake sensor readings for three scenarios
azure_source.py          Send each reading to the Azure ML online endpoint
azure/score/score.py     Scoring script that runs inside that endpoint
app.py                   Streamlit dashboard
edge_inference.py        Column order and the local training helper
requirements.txt         Pinned packages
data/                    CSV and a short description of the simulation
models/                  joblib model and metrics.json
mlflow.db                Local MLflow tracking and registry database
mlartifacts/             Files logged by local MLflow
azure/job.yml            Azure ML training job
azure/conda.yml          Training environment used in Azure
azure/endpoint.yml       Online endpoint
azure/deployment.yml     Deploys model version 1 to that endpoint
```

## Dataset

The file `data/agriculture_sensor_data.csv` has 4,000 rows created by `generate_dataset.py` with random seed 42.

**This is simulated data. It was not collected from a real farm, weather station, or soil probe.** Every row includes `data_source = simulated_not_real_farm`.

| Column | Meaning | Unit |
| --- | --- | --- |
| soil_moisture | How wet the soil is | % |
| temperature | Air temperature | °C |
| humidity | Relative humidity | % |
| rainfall | Recent rainfall | mm |
| irrigation_required | Target label | 1 = required, 0 = not required |

Rows are drawn from six weather and soil situations: normal, dry, hot and dry, rainy, cool wet soil, and recent rain on soil that has not soaked through yet.

Labels follow these rules, in order:

1. If soil moisture is at least 60% **or** rainfall is at least 5 mm, irrigation is not required.
2. Otherwise, if soil moisture is below 30%, irrigation is required.
3. Otherwise, if soil moisture is below 45%, temperature is at least 34°C, humidity is below 45%, and rainfall is below 1 mm, irrigation is required.
4. Otherwise, irrigation is not required.

After those rules, about 2% of labels are flipped at random. That stands in for an uncertain field decision and keeps the task from being perfectly separable. The seed is fixed, so generating the file again produces the same rows.

On this seeded file, 2,606 rows are labeled not required and 1,394 rows are labeled required.

The model is trained only on the four sensor columns. `sample_id`, `regime`, and `data_source` are not features, so they cannot leak the label.

## Machine learning

Algorithm: `RandomForestClassifier` from scikit-learn.

| Setting | Value |
| --- | --- |
| Trees | 150 |
| Max depth | 8 |
| Min samples per leaf | 2 |
| Random seed | 42 |
| Split | 80% train (3,200 rows), 20% test (800 rows), stratified |

The forest is fit on the training split only. Accuracy, precision, recall, F1, and the confusion matrix below are from the test split. Precision and recall use class 1 (irrigation required).

Features are not scaled. A Random Forest does not need it, and scaling would not be a reason to change these scores.

### Test metrics from the seeded run

| Metric | Score |
| --- | --- |
| Accuracy | 0.9775 |
| Precision | 0.9815 |
| Recall | 0.9534 |
| F1 score | 0.9673 |

Confusion matrix (rows are the actual label, columns are the prediction):

|  | Predicted 0 | Predicted 1 |
| --- | ---: | ---: |
| Actual 0 | 516 | 5 |
| Actual 1 | 13 | 266 |

Feature importances on this run: humidity 0.6020, temperature 0.2558, soil moisture 0.1079, rainfall 0.0343. Humidity and temperature rank high because dry, hot conditions and rainy conditions differ strongly on those readings. Rainfall is partly redundant with humidity in the simulated weather, so its importance is smaller even though the labeling rules use it.

These numbers measure how well the forest reproduces the simulated labeling rules. They are **not** real-world irrigation accuracy.

The dashboard shows the same metrics under **Model evaluation on the test set**, read from the Azure ML job rather than from the local `metrics.json` file.

## MLflow model management

Tracking stays on this computer:

- Database: `mlflow.db` (SQLite)
- Artifacts: `mlartifacts/`
- Experiment: `smart_agriculture_irrigation`
- Registered model: `smart-agri-irrigation-predictor`
- Version from the latest seeded run: **2**
- Alias: **champion**

Each training run logs the model parameters, accuracy, precision, recall, F1, confusion-matrix counts, feature importances, the dataset, and the fitted model. The Model Registry entry is created in the same local database. No Databricks account or hosted tracking server is used.

MLflow 2.22 supports this registry because the tracking store is SQLite. A plain files-only store would not. If a future MLflow install cannot register models, `train_model.py` still saves the joblib file and prints the registry error. The dashboard does not use that local file. It sends readings to the Azure endpoint.

The dashboard does **not** need the MLflow UI process to stay running.

## Online scoring

`azure_source.py` posts each simulated reading to the Azure ML endpoint `aeal-irrig-score`. The model stays in the workspace. `azure/score/score.py` runs there, calls `predict` and `predict_proba`, and returns the label and probability. Nothing in this project sends a message to equipment.

`sensor_simulator.py` is clearly separate from inference. It invents readings. The dashboard labels every reading with source `simulated`.

Three manual scenarios keep the demo easy to follow:

| Scenario | What it imitates | Usual prediction |
| --- | --- | --- |
| Normal conditions | Comfortable moisture, mild weather, little rain | NOT REQUIRED |
| Dry soil | Low moisture, hot dry air, almost no rain | REQUIRED |
| High moisture / rainfall | Wet soil and recent rain | NOT REQUIRED |

A new reading is produced every 2 seconds while the simulation is running. Changing the scenario jumps into the new range on the next reading.

## Azure Machine Learning

Submission training uses the **Azure for Students** subscription only. Do not run these commands while the Azure CLI is pointed at another subscription.

The student policy allows only some regions. This workspace is in **UAE North**, resource group `AEAL_HUB`, workspace `mlw-aeal-irrigation`. The job uses serverless compute `Standard_DS2_v2` for one short training run, so there is no virtual machine left running afterward.

```powershell
az account set --subscription "Azure for Students"
az account show --query "{subscription:name, user:user.name}" --output table
az extension add --name ml
az ml job create --file azure/job.yml --resource-group AEAL_HUB --workspace-name mlw-aeal-irrigation
az ml job list --resource-group AEAL_HUB --workspace-name mlw-aeal-irrigation --output table
az ml model list --resource-group AEAL_HUB --workspace-name mlw-aeal-irrigation --output table
```

`train_model.py` detects that Azure ML started the run and logs the experiment to that workspace. On a laptop, the same script still uses the local `mlflow.db` file. The Streamlit dashboard does not use that local file. It posts each reading to the deployed endpoint.

The completed submission job is `sleepy_hamster_zj42xxk8q9`. It registered `smart-agri-irrigation-predictor` version 1. That version is deployed as `blue` on endpoint `aeal-irrig-score` in UAE North. Test accuracy on the training job was 0.9775. Azure ML did not accept the MLflow `champion` alias, so the deployment uses version 1.

The endpoint uses one `Standard_DS2_v2` instance and keeps running until you delete it. That uses student subscription credit. Delete it after the demonstration with:

```powershell
az ml online-endpoint delete --name aeal-irrig-score --resource-group AEAL_HUB --workspace-name mlw-aeal-irrigation --yes
```

Studio links for the student workspace:

- [Workspace](https://ml.azure.com/home?wsid=/subscriptions/9cfb5a12-6909-4d08-8336-d7e9ce27e33d/resourceGroups/AEAL_HUB/providers/Microsoft.MachineLearningServices/workspaces/mlw-aeal-irrigation)
- [Training job](https://ml.azure.com/runs/sleepy_hamster_zj42xxk8q9?wsid=/subscriptions/9cfb5a12-6909-4d08-8336-d7e9ce27e33d/resourceGroups/AEAL_HUB/providers/Microsoft.MachineLearningServices/workspaces/mlw-aeal-irrigation)
- [Registered model](https://ml.azure.com/model/smart-agri-irrigation-predictor/1/details?wsid=/subscriptions/9cfb5a12-6909-4d08-8336-d7e9ce27e33d/resourceGroups/AEAL_HUB/providers/Microsoft.MachineLearningServices/workspaces/mlw-aeal-irrigation)
- [Online endpoint](https://ml.azure.com/endpoints/realtime/aeal-irrig-score?wsid=/subscriptions/9cfb5a12-6909-4d08-8336-d7e9ce27e33d/resourceGroups/AEAL_HUB/providers/Microsoft.MachineLearningServices/workspaces/mlw-aeal-irrigation)
- [Online endpoint](https://ml.azure.com/endpoints/realtime/aeal-irrig-score?wsid=/subscriptions/9cfb5a12-6909-4d08-8336-d7e9ce27e33d/resourceGroups/AEAL_HUB/providers/Microsoft.MachineLearningServices/workspaces/mlw-aeal-irrigation)

## Installation and execution

Tested on Windows with Python 3.13. If `python` starts Python 3.15, use `py -3.13` instead. Python 3.15 is too new for these pinned packages.

Clone the repository and enter the folder:

```powershell
git clone https://github.com/Sudesh-J/predictive_maintenance_aeal.git
cd predictive_maintenance_aeal
```

Create the environment and install the pinned packages:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Sign in to the student subscription before starting the dashboard. The page refuses to open on any other Azure account:

```powershell
az login
az account set --subscription "Azure for Students"
az account show --query "{subscription:name, user:user.name}" --output table
```

Then start the dashboard:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

To rebuild the dataset, retrain the forest, and log a new MLflow run, use this order instead:

```powershell
.\.venv\Scripts\python.exe generate_dataset.py
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the dashboard at [http://localhost:8501](http://localhost:8501).

If PowerShell blocks `Activate.ps1`, you can ignore activation and keep using `.\.venv\Scripts\python.exe`, as above. To activate the environment instead:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### MLflow UI

Use a second terminal in the same folder. Leave it open only while you want to browse the experiment.

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

- Experiment: `smart_agriculture_irrigation`
- Model registry: `smart-agri-irrigation-predictor`, version 2, alias `champion`

Stop that terminal when you are finished. The dashboard reads Azure ML, so this local UI can stay closed.

### If a file is missing

- Dataset missing: run `generate_dataset.py`. The trainer and the dashboard both say this.
- Dashboard says the wrong subscription: run `az account set --subscription "Azure for Students"` and refresh the page.
- MLflow page is empty: run `train_model.py` from this folder, then start the UI with the command above.
- After you move the project to another folder, run `train_model.py` again if you want the local MLflow database paths to match. The dashboard does not use those local files.

Optional check of the saved model, without the dashboard:

```powershell
.\.venv\Scripts\python.exe edge_inference.py
```

## Five-minute demonstration

Have the environment installed before you start. Training can already be done. Open two terminals only if you want MLflow on screen.

1. **Problem (about 30 seconds).** Say that the system recommends irrigation from soil moisture, temperature, humidity, and rainfall, and that the recommendation does not control equipment.
2. **Data (about 45 seconds).** Open `data/agriculture_sensor_data.csv` or `data/dataset_info.json`. Say the rows are simulated, show the four sensor columns and `irrigation_required`, and state the labeling rules in one sentence: wet soil or meaningful rain means no irrigation; dry soil, or moderately dry soil in hot dry air, means irrigate.
3. **Training (about 60 seconds).** Scroll the `train_model.py` output, or open **Model evaluation on the test set** in the dashboard. Read accuracy 0.9775, precision 0.9815, recall 0.9534, and the confusion matrix. Say these scores are on simulated test rows, not a real farm.
4. **Azure ML (about 45 seconds).** In the dashboard sidebar, show subscription **Azure for Students**, endpoint `aeal-irrig-score`, and model `smart-agri-irrigation-predictor` version 1. Say each reading is sent to that endpoint.
5. **Live demo (about 90 seconds).** Turn on **Run live simulation**. With **Normal conditions**, the status stays NOT REQUIRED. Switch to **Dry soil** and point at the red IRRIGATION REQUIRED alert, the probability, and the updating chart. Switch to **High moisture / rainfall** and show the status return to NOT REQUIRED. Say the readings are simulated and the prediction came back from Azure.
6. **Close (about 30 seconds).** State the limits below: simulated sensors, rule-based labels, and no hardware.

## Limitations

- The dataset is simulated. It is not a record of a real field.
- Labels come from written rules plus a little noise. A high test score means the model learned those rules, not that it would match an agronomist's decision on a real farm.
- The sensor feed is generated in Python. No microcontroller, radio, or soil probe is attached.
- Nothing in this project switches irrigation equipment.
- The live scenario ranges are chosen so a short demo is easy to see. They are not a full climate model.
- MLflow artifact paths are stored for this computer. Moving the folder means training once more if you want the UI to open old artifacts.
- The Model Registry used here depends on MLflow 2.22 with SQLite and SQLAlchemy 2.0.

## Future improvements

- Replace the simulator with a real sensor node after the demo model is understood.
- Collect labels from crop observations instead of fixed rules.
- Predict how much water to apply, not only yes or no.
- Compare another simple classifier against this Random Forest in a second MLflow run.
- Keep the dashboard on the laptop even if a later version reads a local sensor, so inference stays at the edge.

## What this project does not claim

It does not claim field-trial accuracy, a connection to real IoT hardware, or any automatic control of water.
