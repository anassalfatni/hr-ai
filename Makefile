install: ; pip install -r requirements.txt
data:    ; python scripts/make_synthetic_salary.py
train:   ; python -m src.train
test:    ; pytest -q
api:     ; python -m app.api
ui:      ; streamlit run app/streamlit_app.py
all: data train test
