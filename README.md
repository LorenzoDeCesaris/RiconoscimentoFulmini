# RiconoscimentoFulmini
Due notebook con funzioni allegate per permettere la lettura dei file di magnetometro e addestrare una rete neurale a riconoscere segnali anomali, in questo caso fulmini.

Dati: magnetometro EMMA2013 di L'Aquila (AQU), campionamento 1 Hz, anno 2025.
Catalogo fulmini entro 50 km. I dati non sono inclusi nel repository, sono dati privati.

## Contenuto
- `lightning_pipeline.ipynb` — lettura dei dati, verifica del segnale, costruzione del dataset
- `lightning_nn.ipynb` — rete convoluzionale 1D, valutazione, confronto con una baseline (rapporto segnale/rumore) e verifiche statistiche.
- `lightning_aligner.py`, `emma2013_reader.py` — lettura dei file e costruzione del dataset.
- `environment.yml` — ambiente conda.

## Risultati principali (test set, normalizzazione per rumore locale)
- AUC-PR: rete ≈ 0.46, baseline ≈ 0.26.
- A 1 falso allarme al giorno: recall rete ≈ 0.53 contro baseline ≈ 0.35.
- Vantaggio statisticamente significativo (IC 95% del Δ recall: [+0.06, +0.23]; McNemar p ≈ 0.002),
  stabile su 5 addestramenti con seed diversi.
- I rilevamenti senza fulmine in catalogo si concentrano entro pochi minuti dai temporali:
  probabili fulmini non catalogati.

## Come rieseguire
1. `conda env create -f environment.yml`
2. Impostare i percorsi nella cella 1 della pipeline e generare il dataset.
3. Eseguire il notebook della rete utilizzando il dataset generato.

## Ulteriori contenuti
I notebook eseguiti in formato pdf
