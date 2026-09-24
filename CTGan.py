import pandas as pd
from sdv.constraints import FixedCombinations,ScalarInequality,Constraint
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

file_path = r"C:\Users\110095\Desktop\Materials\MASSET MASTER.csv"
data = pd.read_csv(file_path, encoding='utf-8', encoding_errors='replace')

metadata = SingleTableMetadata()
metadata.detect_from_dataframe(data=data)
metadata.update_column(column_name='NATIVE_STATE', sdtype='categorical')
metadata.update_column(column_name='NUMBER_OF_PROMOTIONS', sdtype='numerical')

synthesizer = CTGANSynthesizer(metadata,
    enforce_min_max_values=True,
    epochs=600)

synthesizer.fit(data)

synthetic_data = synthesizer.sample(num_rows=400)